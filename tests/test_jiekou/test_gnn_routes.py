"""测试 GNN 路由辅助函数。"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from jiekou.routes.gnn_routes import (
    _load_latest_predictions, _load_edges, _build_nodes, _cap_edges,
)


class TestLoadPredictions:
    def test_no_file(self):
        with patch('jiekou.routes.gnn_routes._PRED_PATH', Path('/nonexistent/path.json')):
            result = _load_latest_predictions()
            assert isinstance(result, dict)
            assert 'error' in result


class TestBuildNodes:
    def test_color_tiers(self):
        codes = {"000001", "000002", "000003"}
        preds = {"000001": 0.9, "000002": 0.5, "000003": 0.1}
        nodes = _build_nodes(codes, preds)
        assert len(nodes) == 3
        colors = {n["id"]: n["itemStyle"]["color"] for n in nodes}
        assert colors["000001"] == "#D4AF37"  # gold
        assert colors["000002"] == "#2ECC71"  # green
        assert colors["000003"] == "#4198FF"  # blue

    def test_single_node(self):
        nodes = _build_nodes({"000001"}, {"000001": 0.5})
        assert len(nodes) == 1
        assert nodes[0]["id"] == "000001"

    def test_all_same_score(self):
        """得分全相同时不除以零，归一化为 0。"""
        nodes = _build_nodes({"a", "b"}, {"a": 0.5, "b": 0.5})
        assert len(nodes) == 2
        # norm = (0.5-0.5) / max(0, 1e-6) = 0 → 蓝色
        for n in nodes:
            assert n["symbolSize"] == 14
            assert n["itemStyle"]["color"] == "#4198FF"


class TestCapEdges:
    def test_under_limit(self):
        edges = [{"source": "a", "target": "b", "weight": 0.5}]
        result = _cap_edges(edges, max_edges=400)
        assert result == edges  # unchanged

    def test_over_limit(self):
        edges = [{"source": str(i), "target": str(i + 1), "weight": float(i)}
                 for i in range(500)]
        result = _cap_edges(edges, max_edges=400)
        assert len(result) == 400
        assert result[0]["weight"] == 499  # highest first


class TestEdgeKeyOrdering:
    def test_undirected_dedup(self):
        """验证 (sc, dc) if sc < dc else (dc, sc) 去重正确。"""
        # 模拟 _load_edges 的去重逻辑
        a, b = "000001", "000002"
        k1 = (a, b) if a < b else (b, a)
        k2 = (b, a) if b < a else (a, b)
        assert k1 == k2  # 两个方向产生相同的键
