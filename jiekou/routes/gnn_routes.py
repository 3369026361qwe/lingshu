"""GNN 路由 — 产业链图数据 + 推理端点。"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api", tags=["gnn"])

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PRED_PATH = _MODEL_DIR / "gnn_predictions.json"
_CONFIG_PATH = _MODEL_DIR / "gnn_config.json"

MAX_EDGES = 400


# ── 辅助函数 ──────────────────────────────────────────

def _load_latest_predictions():
    """加载 GNN 预测 JSON → (latest_date, latest_preds) 或错误字典。"""
    if not _PRED_PATH.exists():
        return {"nodes": [], "edges": [], "error": "no predictions yet — run train_gnn_model.py first"}

    all_preds: dict[str, float] = json.loads(_PRED_PATH.read_text())
    if not all_preds:
        return {"nodes": [], "edges": [], "error": "empty predictions"}

    by_date = defaultdict(dict)
    for key, score in all_preds.items():
        date_str, code = key.split("|")
        by_date[date_str][code] = score

    if not by_date:
        return {"nodes": [], "edges": []}

    latest_date = max(by_date.keys())
    return latest_date, by_date[latest_date]


def _load_edges(top_codes: set, latest_preds: dict) -> list[dict]:
    """从 GNN checkpoint 加载边，过滤到涉及 top_codes 的边。"""
    edges = []
    if not _CONFIG_PATH.exists():
        return edges

    try:
        import torch
        pt_path = _MODEL_DIR / "gnn_model.pt"
        if not pt_path.exists():
            return edges

        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        ei = ckpt.get("edge_index", None)
        stock_codes = ckpt.get("stock_codes", [])
        if ei is None or not stock_codes:
            return edges

        ei_np = ei.cpu().numpy() if hasattr(ei, 'cpu') else np.array(ei)
        seen_edges = set()
        n_codes = len(stock_codes)

        for s, d in ei_np.T:
            si, di = int(s), int(d)
            if si >= n_codes or di >= n_codes:
                continue
            sc = stock_codes[si]
            dc = stock_codes[di]
            if not (sc in top_codes or dc in top_codes):
                continue

            edge_key = (sc, dc) if sc < dc else (dc, sc)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            w = abs(latest_preds.get(sc, 0) - latest_preds.get(dc, 0))
            edges.append({
                "source": sc, "target": dc,
                "weight": round(0.5 + w * 5, 2),
            })
    except Exception:
        pass

    return edges


def _build_nodes(all_codes: set, latest_preds: dict) -> list[dict]:
    """归一化得分 → 构建节点列表（颜色、尺寸、分数）。"""
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
        if norm > 0.7:
            color, symbol_size = "#D4AF37", 28
        elif norm > 0.4:
            color, symbol_size = "#2ECC71", 20
        else:
            color, symbol_size = "#4198FF", 14

        nodes.append({
            "id": code,
            "name": code,
            "score": round(score, 4),
            "symbolSize": symbol_size,
            "itemStyle": {"color": color},
        })
    return nodes


def _cap_edges(edges: list[dict], max_edges: int = MAX_EDGES) -> list[dict]:
    """按权重降序排序，截断到 max_edges。"""
    if len(edges) > max_edges:
        edges.sort(key=lambda e: e["weight"], reverse=True)
        return edges[:max_edges]
    return edges


# ── 路由 ──────────────────────────────────────────────

@router.get("/gnn/graph")
async def get_gnn_graph(top_n: int = Query(default=50, ge=10, le=200)):
    """返回 GNN 产业链图数据（节点 + 边），供前端力导向图渲染。"""
    result = _load_latest_predictions()
    if isinstance(result, dict):
        return result  # 错误响应
    latest_date, latest_preds = result

    sorted_codes = sorted(latest_preds.items(), key=lambda x: x[1], reverse=True)
    top_codes = set(c for c, _ in sorted_codes[:top_n])

    edges = _load_edges(top_codes, latest_preds)

    all_nodes_in_edges = set()
    for e in edges:
        all_nodes_in_edges.add(e["source"])
        all_nodes_in_edges.add(e["target"])
    all_codes = top_codes | all_nodes_in_edges

    nodes = _build_nodes(all_codes, latest_preds)
    edges = _cap_edges(edges)

    return {
        "date": latest_date,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
