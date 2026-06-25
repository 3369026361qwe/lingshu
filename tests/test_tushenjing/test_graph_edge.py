"""tushenjing 边缘场景测试：空图、单节点、大数据集、重复边、GAT注意力。"""
import time
import numpy as np
from tushenjing.graph_builder import GraphBuilder
from tushenjing.graph_utils import GraphUtils
from tushenjing.gnn_model import GCNModel
from tushenjing.graph_trainer import GraphTrainer


class TestEmptyGraph:
    def test_empty_stock_list(self):
        result = GraphBuilder().build([])
        assert result["num_nodes"] == 0
        assert result["num_edges"] == 0

    def test_no_industry_map(self):
        result = GraphBuilder().build(["000001", "000002"])
        assert result["num_nodes"] == 2
        assert result["num_edges"] == 0

    def test_single_node(self):
        result = GraphBuilder().build(["000001"], {"000001": "银行"})
        assert result["num_nodes"] == 1
        assert result["num_edges"] == 0


class TestLargeGraph:
    def test_large_industry_performance(self):
        stocks = [f"{i:06d}" for i in range(1, 301)]
        industry = {c: "银行" for c in stocks}
        builder = GraphBuilder()
        t0 = time.perf_counter()
        result = builder.build(stocks, industry)
        elapsed = time.perf_counter() - t0
        assert result["num_nodes"] == 300
        # P1-2: 采样后边数控制
        max_possible = 300 * 20 // 2  # ~3000
        assert len(result["edges"]["same_industry"]) <= max_possible * 2
        assert elapsed < 10.0, f"Build too slow: {elapsed:.2f}s"

    def test_adjacency_empty_edges(self):
        builder = GraphBuilder()
        builder.build(["000001", "000002"])
        adj = builder.to_adjacency_matrix()
        assert adj.sum() == 0


class TestDedup:
    def test_no_duplicate_edges(self):
        stocks = [f"{i:06d}" for i in range(1, 11)]
        industry = {c: "银行" for c in stocks}
        builder = GraphBuilder()
        result = builder.build(stocks, industry)
        for etype, edges in result["edges"].items():
            keys = [(min(a, b), max(a, b)) for a, b in edges]
            assert len(keys) == len(set(keys)), f"Duplicate edges in {etype}"


class TestFeatureEdgeCases:
    def test_empty_factor_data(self):
        features, names = GraphUtils.build_feature_matrix(["000001", "000002"], {})
        assert features.shape == (2, 0)

    def test_all_none_factors(self):
        factor_data = {"000001": {"pe": None}, "000002": {"pe": None}}
        features, _ = GraphUtils.build_feature_matrix(["000001", "000002"], factor_data, feature_order=["pe"])
        assert features[0, 0] == 0.0

    def test_zero_std_features(self):
        x = np.array([[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]], dtype=np.float32)
        result = GraphUtils.normalize_features(x)
        assert not np.isnan(result).any()


class TestGATAttention:
    def test_gat_output_shape(self):
        from tushenjing.gnn_numpy import NumPyGAT
        gat = NumPyGAT(in_dim=3, hidden_dim=6, out_dim=1)
        x = np.random.randn(10, 3).astype(np.float32)
        adj = np.eye(10, dtype=np.float32)
        for i in range(9):
            adj[i, i+1] = 1.0; adj[i+1, i] = 1.0
        out = gat.forward(x, adj)
        assert out.shape == (10, 1)

    def test_gat_weight_snapshot(self):
        from tushenjing.gnn_numpy import NumPyGAT
        gat = NumPyGAT(in_dim=3, hidden_dim=4, out_dim=1)
        snap = gat.get_weights_snapshot()
        assert "a_left" in snap
        x = np.random.randn(10, 3).astype(np.float32)
        adj = np.eye(10, dtype=np.float32) * 0.1
        out1 = gat.forward(x, adj)
        gat.load_weights_snapshot(snap)
        out2 = gat.forward(x, adj)
        np.testing.assert_array_almost_equal(out1, out2)


class TestTrainerEdgeCases:
    def test_early_stopping_triggered(self):
        model = GCNModel(in_dim=2, hidden_dim=4, out_dim=1)
        trainer = GraphTrainer(model, epochs=200, patience=5)
        x = np.random.randn(20, 2).astype(np.float32)
        adj = np.eye(20, dtype=np.float32) * 0.1
        y = np.random.randn(20, 1).astype(np.float32)
        train, val = GraphUtils.train_test_split_mask(20)
        result = trainer.fit(x, adj, y, train, val)
        assert result["early_stopped"] or len(result["train_losses"]) < 200
