"""
测试图构建器、图工具、GNN模型、推理、更新器。
"""

from decimal import Decimal

import numpy as np
import pytest

from tushenjing.gnn_model import GATModel, GCNModel, create_model
from tushenjing.graph_builder import GraphBuilder
from tushenjing.graph_inference import GraphInference
from tushenjing.graph_updater import GraphUpdater
from tushenjing.graph_utils import GraphUtils

# ── 测试数据工厂 ─────────────────────────────────────

def _make_stock_list(n=100):
    return [f"{i:06d}" for i in range(1, n + 1)]


def _make_industry_map(stocks):
    industries = ["银行", "电子", "医药生物", "食品饮料", "汽车", "计算机", "有色金属", "钢铁"]
    return {c: industries[i % len(industries)] for i, c in enumerate(stocks)}


def _make_factor_data(stocks, n_factors=5):
    """生成模拟因子数据。"""
    return {
        c: {f"factor_{j}": Decimal(str((i + j) * 0.1)) for j in range(n_factors)}
        for i, c in enumerate(stocks)
    }


# ── GraphUtils ──────────────────────────────────────

class TestGraphUtils:
    def test_normalize_features_zscore(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = GraphUtils.normalize_features(x, method="zscore")
        assert result.shape == (3, 2)
        assert abs(result[:, 0].mean()) < 1e-6  # 均值为 0

    def test_normalize_features_minmax(self):
        x = np.array([[0.0, 5.0], [5.0, 10.0]], dtype=np.float32)
        result = GraphUtils.normalize_features(x, method="minmax")
        assert result.min() >= 0
        assert result.max() <= 1

    def test_build_feature_matrix(self):
        stocks = _make_stock_list(5)
        factor_data = _make_factor_data(stocks, 3)
        features, names = GraphUtils.build_feature_matrix(stocks, factor_data)
        assert features.shape == (5, 3)
        assert len(names) == 3

    def test_build_feature_matrix_missing(self):
        stocks = _make_stock_list(3)
        factor_data = {"000001": {"pe": Decimal("15")}, "000002": {}}
        features, names = GraphUtils.build_feature_matrix(stocks, factor_data, feature_order=["pe"])
        assert features.shape == (3, 1)
        assert features[1, 0] == 0.0  # fill_value

    def test_edges_to_adjacency(self):
        adj = GraphUtils.edges_to_adjacency([(0, 1), (1, 2)], 3)
        assert adj.shape == (3, 3)
        assert adj[0, 1] == 1 and adj[1, 0] == 1  # symmetric

    def test_normalize_adjacency(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
        norm = GraphUtils.normalize_adjacency(adj)
        assert norm.shape == adj.shape

    def test_train_test_split(self):
        train, test = GraphUtils.train_test_split_mask(100)
        assert train.sum() + test.sum() == 100
        assert abs(train.sum() - 80) <= 1


# ── GraphBuilder ────────────────────────────────────

class TestGraphBuilder:
    def test_build_basic_graph(self):
        stocks = _make_stock_list(50)
        industry = _make_industry_map(stocks)
        builder = GraphBuilder()
        result = builder.build(stocks, industry)
        assert result["num_nodes"] == 50
        assert result["num_edges"] > 0
        assert "supply_chain" in result["edges"]
        assert "same_industry" in result["edges"]

    def test_same_industry_edges(self):
        stocks = _make_stock_list(20)
        industry = {c: "银行" for c in stocks}  # 全部同一行业
        builder = GraphBuilder()
        result = builder.build(stocks, industry)
        # 全连接图边数 = n*(n-1)/2
        assert len(result["edges"]["same_industry"]) == 20 * 19 // 2

    def test_fund_holdings_edges(self):
        stocks = _make_stock_list(10)
        fund = {"fund_1": stocks[:5]}
        builder = GraphBuilder()
        result = builder.build(stocks, fund_holdings=fund)
        assert len(result["edges"]["co_holding"]) == 10  # C(5,2) = 10

    def test_concept_edges(self):
        stocks = _make_stock_list(10)
        concept = {"新能源": stocks[:5], "AI": stocks[5:]}
        builder = GraphBuilder()
        result = builder.build(stocks, concept_map=concept)
        assert len(result["edges"]["concept"]) == 20  # 2 * C(5,2)

    def test_to_adjacency_matrix(self):
        builder = GraphBuilder()
        builder.build(_make_stock_list(10), _make_industry_map(_make_stock_list(10)))
        adj = builder.to_adjacency_matrix(["same_industry"])
        assert adj.shape == (10, 10)
        assert (adj == adj.T).all()  # 对称

    def test_to_pyg_data(self):
        stocks = _make_stock_list(10)
        builder = GraphBuilder()
        builder.build(stocks, _make_industry_map(stocks))
        features = np.random.randn(10, 5).astype(np.float32)
        data = builder.to_pyg_data(features)
        assert data["num_nodes"] == 10
        assert data["x"].shape == (10, 5)


# ── GNN Models ──────────────────────────────────────

class TestGNNModels:
    def test_gcn_forward(self):
        model = GCNModel(in_dim=5, hidden_dim=8, out_dim=1)
        x = np.random.randn(20, 5).astype(np.float32)
        adj = GraphUtils.normalize_adjacency(np.ones((20, 20), dtype=np.float32) * 0.05)
        out = model(x, adj)
        assert out.shape == (20, 1)

    def test_gat_forward(self):
        model = GATModel(in_dim=5, hidden_dim=8, out_dim=1)
        x = np.random.randn(20, 5).astype(np.float32)
        adj = np.eye(20, dtype=np.float32)
        np.fill_diagonal(adj, 0)
        # 添加一些边
        for i in range(19):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        out = model(x, adj)
        assert out.shape == (20, 1)

    def test_create_model_factory(self):
        gcn = create_model("gcn", in_dim=5, hidden_dim=8, out_dim=1)
        assert gcn is not None
        gat = create_model("gat", in_dim=5, hidden_dim=8, out_dim=1)
        assert gat is not None
        with pytest.raises(ValueError):
            create_model("unknown", in_dim=5)

    def test_weight_snapshot(self):
        model = GCNModel(in_dim=3, hidden_dim=4, out_dim=1)
        snapshot = model.model.get_weights_snapshot()
        assert isinstance(snapshot, dict) and len(snapshot) > 0, "weight snapshot must be non-empty dict"
        # PyG 模型需设 eval 模式禁用 dropout，保证前向传播确定性
        if hasattr(model.model, 'eval'):
            model.model.eval()
        x = np.random.randn(10, 3).astype(np.float32)
        adj = GraphUtils.normalize_adjacency(np.ones((10, 10)) * 0.1)
        out1 = model(x, adj)
        model.model.load_weights_snapshot(snapshot)
        out2 = model(x, adj)
        # 兼容 torch tensor（detach before numpy）
        try:
            import torch
            if isinstance(out1, torch.Tensor):
                out1 = out1.detach().numpy()
            if isinstance(out2, torch.Tensor):
                out2 = out2.detach().numpy()
        except ImportError:
            pass
        np.testing.assert_array_almost_equal(out1, out2)


# ── GraphTrainer ────────────────────────────────────

class TestGraphTrainer:
    def test_fit_and_improve(self):
        from tushenjing.graph_trainer import GraphTrainer

        model = GCNModel(in_dim=3, hidden_dim=8, out_dim=1)
        trainer = GraphTrainer(model, epochs=50, patience=10)

        x = np.random.randn(30, 3).astype(np.float32)
        adj = GraphUtils.normalize_adjacency(np.eye(30, dtype=np.float32) + 0.1)
        y = np.random.randn(30, 1).astype(np.float32)
        train, val = GraphUtils.train_test_split_mask(30)

        result = trainer.fit(x, adj, y, train, val)
        assert len(result["train_losses"]) > 0
        assert len(result["val_losses"]) > 0
        assert result["best_epoch"] > 0


# ── GraphInference ──────────────────────────────────

class TestGraphInference:
    def test_predict(self):
        stocks = _make_stock_list(10)
        model = GCNModel(in_dim=3, hidden_dim=4, out_dim=1)
        inference = GraphInference(model)

        x = np.random.randn(10, 3).astype(np.float32)
        adj = GraphUtils.normalize_adjacency(np.ones((10, 10)) * 0.1)
        scores = inference.predict(x, adj, stocks)
        assert len(scores) == 10
        assert all(isinstance(v, float) for v in scores.values())

    def test_predict_batch(self):
        stocks = _make_stock_list(10)
        industry = _make_industry_map(stocks)
        builder = GraphBuilder()
        graph_data = builder.build(stocks, industry)

        model = GCNModel(in_dim=2, hidden_dim=4, out_dim=1)
        inference = GraphInference(model)
        factor_data = _make_factor_data(stocks, 2)

        scores = inference.predict_batch(graph_data, factor_data)
        assert len(scores) == 10


# ── GraphUpdater ────────────────────────────────────

class TestGraphUpdater:
    def test_add_remove_nodes(self):
        stocks = _make_stock_list(20)
        builder = GraphBuilder()
        graph_data = builder.build(stocks, _make_industry_map(stocks))
        updater = GraphUpdater(graph_data)

        # 添加节点
        added = updater.add_nodes(["999001", "999002"])
        assert added == 2
        assert updater.graph_data["num_nodes"] == 22

        # 移除节点
        removed = updater.remove_nodes(["000001", "000002"])
        assert removed == 2
        assert updater.graph_data["num_nodes"] == 20

    def test_add_remove_edges(self):
        stocks = _make_stock_list(20)
        builder = GraphBuilder()
        graph_data = builder.build(stocks, _make_industry_map(stocks))
        updater = GraphUpdater(graph_data)
        old_count = graph_data["num_edges"]

        added = updater.add_edges("supply_chain", [("000001", "000003"), ("000002", "000004")])
        assert added > 0
        assert updater.graph_data["num_edges"] > old_count

    def test_update_features(self):
        stocks = _make_stock_list(10)
        builder = GraphBuilder()
        graph_data = builder.build(stocks, _make_industry_map(stocks))
        factor_data = _make_factor_data(stocks, 3)
        features = GraphUpdater.update_features(graph_data, factor_data)
        assert features.shape == (10, 3)
