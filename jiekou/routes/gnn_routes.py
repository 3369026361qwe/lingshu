"""GNN 路由 — 产业链图数据 + 推理端点。"""
import json
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api", tags=["gnn"])

# 模型数据路径
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PRED_PATH = _MODEL_DIR / "gnn_predictions.json"
_CONFIG_PATH = _MODEL_DIR / "gnn_config.json"


@router.get("/gnn/graph")
async def get_gnn_graph(top_n: int = Query(default=50, ge=10, le=200)):
    """返回 GNN 产业链图数据（节点 + 边），供前端力导向图渲染。

    从训练好的 GNN 预测中取 Top-N 股票作为核心节点，
    再根据 k-NN 图边关系补上相邻节点。
    """
    import numpy as np

    # 1. 加载预测数据
    if not _PRED_PATH.exists():
        return {"nodes": [], "edges": [], "error": "no predictions yet — run train_gnn_model.py first"}

    all_preds: dict[str, float] = json.loads(_PRED_PATH.read_text())
    if not all_preds:
        return {"nodes": [], "edges": [], "error": "empty predictions"}

    # 2. 找出最新日期的预测
    from collections import defaultdict
    by_date = defaultdict(dict)
    for key, score in all_preds.items():
        date_str, code = key.split("|")
        by_date[date_str][code] = score

    if not by_date:
        return {"nodes": [], "edges": []}

    latest_date = max(by_date.keys())
    latest_preds = by_date[latest_date]

    # 3. Top-N 核心节点
    sorted_codes = sorted(latest_preds.items(), key=lambda x: x[1], reverse=True)
    top_codes = set(c for c, _ in sorted_codes[:top_n])

    # 4. 加载边数据 (从 gnn_config.json 读 k-NN 参数, 从缓存读 edge_index)
    edges = []
    if _CONFIG_PATH.exists():
        try:
            import torch
            # 尝试从 .pt 文件加载边
            pt_path = _MODEL_DIR / "gnn_model.pt"
            if pt_path.exists():
                ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                ei = ckpt.get("edge_index", None)
                stock_codes = ckpt.get("stock_codes", [])
                if ei is not None and stock_codes:
                    ei_np = ei.cpu().numpy() if hasattr(ei, 'cpu') else np.array(ei)
                    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
                    # 只保留涉及 Top-N 或它们的邻居的边
                    seen_edges = set()
                    for s, d in ei_np.T:
                        sc = stock_codes[int(s)] if int(s) < len(stock_codes) else None
                        dc = stock_codes[int(d)] if int(d) < len(stock_codes) else None
                        if sc and dc and (sc in top_codes or dc in top_codes):
                            edge_key = tuple(sorted([sc, dc]))
                            if edge_key not in seen_edges:
                                seen_edges.add(edge_key)
                                # 边的粗细反比于得分差距
                                w = abs(latest_preds.get(sc, 0) - latest_preds.get(dc, 0))
                                edges.append({
                                    "source": sc, "target": dc,
                                    "weight": round(0.5 + w * 5, 2),
                                })
        except Exception:
            pass

    # 5. 收集所有涉及的节点
    all_nodes_in_edges = set()
    for e in edges:
        all_nodes_in_edges.add(e["source"])
        all_nodes_in_edges.add(e["target"])

    all_codes = top_codes | all_nodes_in_edges

    # 6. 构建节点列表
    scores = [latest_preds.get(c, 0) for c in all_codes]
    if scores:
        smin, smax = min(scores), max(scores)
        rng = max(smax - smin, 1e-6)
    else:
        smin, smax, rng = 0, 1, 1

    nodes = []
    for code in all_codes:
        score = latest_preds.get(code, 0)
        norm = (score - smin) / rng
        # 颜色: 得分高→金色, 得分低→蓝色
        if norm > 0.7:
            color = "#D4AF37"  # gold
            symbol_size = 28
        elif norm > 0.4:
            color = "#2ECC71"  # green
            symbol_size = 20
        else:
            color = "#4198FF"  # blue
            symbol_size = 14

        nodes.append({
            "id": code,
            "name": code,
            "score": round(score, 4),
            "symbolSize": symbol_size,
            "itemStyle": {"color": color},
        })

    # 限制边数 (前端力导向图 500 条以下是流畅的)
    if len(edges) > 400:
        edges.sort(key=lambda e: e["weight"], reverse=True)
        edges = edges[:400]

    return {
        "date": latest_date,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
