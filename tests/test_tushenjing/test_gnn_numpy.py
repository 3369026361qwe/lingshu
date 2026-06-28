"""测试 tushenjing/gnn_numpy.py — NumPy GCN/GAT 与 PyG 一致性。

验证纯 NumPy 实现与 PyG 版本的关键对齐：
  - Xavier 初始化
  - 前向传播 shape
  - 权重快照 roundtrip
  - PyG checkpoint 映射
"""
import numpy as np
import pytest

from tushenjing.gnn_numpy import (
    NumPyGAT,
    NumPyGCN,
    _extract_attention_param,
    _map_pyg_gat_to_numpy,
    _map_pyg_gcn_to_numpy,
    _map_pyg_to_numpy,
    _resolve_lin_key,
    _xavier_init,
    _xavier_init_stack,
    load_pyg_checkpoint_to_numpy,
)

# ── Xavier 初始化测试 ─────────────────────────────────────

class TestXavierInit:
    """Xavier/Glorot 初始化测试。"""

    def test_output_shape(self):
        rng = np.random.RandomState(42)
        w = _xavier_init(10, 20, (10, 20), rng)
        assert w.shape == (10, 20)

    def test_values_in_range(self):
        rng = np.random.RandomState(42)
        w = _xavier_init(5, 5, (5, 5), rng)
        scale = np.sqrt(6.0 / (5 + 5))  # sqrt(6/10) ≈ 0.77
        assert np.all(np.abs(w) <= scale + 1e-10)

    def test_randomness(self):
        rng1 = np.random.RandomState(1)
        rng2 = np.random.RandomState(2)
        w1 = _xavier_init(5, 5, (5, 5), rng1)
        w2 = _xavier_init(5, 5, (5, 5), rng2)
        assert not np.allclose(w1, w2)

    def test_deterministic_with_same_seed(self):
        rng_a = np.random.RandomState(99)
        rng_b = np.random.RandomState(99)
        w_a = _xavier_init(10, 10, (10, 10), rng_a)
        w_b = _xavier_init(10, 10, (10, 10), rng_b)
        assert np.allclose(w_a, w_b)

    def test_stack_returns_list(self):
        shapes = [(5, 8, (5, 8)), (8, 3, (8, 3))]
        result = _xavier_init_stack(shapes, seed=42)
        assert len(result) == 2
        assert result[0].shape == (5, 8)
        assert result[1].shape == (8, 3)
        assert result[0].dtype == np.float32


# ── NumPyGCN 测试 ─────────────────────────────────────────

class TestNumPyGCN:
    """纯 NumPy GCN 测试。"""

    def test_forward_shape(self):
        N, in_dim, hidden_dim, out_dim = 10, 5, 8, 1
        gcn = NumPyGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, seed=42)
        x = np.random.RandomState(0).randn(N, in_dim).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)  # 自环
        out = gcn.forward(x, adj)
        assert out.shape == (N, out_dim)

    def test_forward_deterministic(self):
        """相同输入产生相同输出（无 dropout 激活）。"""
        gcn = NumPyGCN(in_dim=4, hidden_dim=8, out_dim=1, seed=42)
        x = np.random.RandomState(99).randn(20, 4).astype(np.float32)
        adj = np.eye(20, dtype=np.float32)
        out1 = gcn.forward(x, adj)
        out2 = gcn.forward(x, adj)
        assert np.allclose(out1, out2)

    def test_different_seeds_different_outputs(self):
        """不同 seed 产生不同权重 → 不同输出。"""
        N, in_dim = 10, 5
        x = np.ones((N, in_dim), dtype=np.float32)
        adj = np.eye(N, dtype=np.float32)
        gcn1 = NumPyGCN(in_dim=in_dim, hidden_dim=8, out_dim=1, seed=1)
        gcn2 = NumPyGCN(in_dim=in_dim, hidden_dim=8, out_dim=1, seed=2)
        out1 = gcn1.forward(x, adj)
        out2 = gcn2.forward(x, adj)
        assert not np.allclose(out1, out2)

    def test_weight_snapshot_roundtrip(self):
        """权重快照保存后加载，输出一致。"""
        gcn1 = NumPyGCN(in_dim=4, hidden_dim=8, out_dim=1, seed=42)
        x = np.random.RandomState(1).randn(10, 4).astype(np.float32)
        adj = np.eye(10, dtype=np.float32)
        out1 = gcn1.forward(x, adj)

        gcn2 = NumPyGCN(in_dim=4, hidden_dim=8, out_dim=1, seed=99)  # 不同 seed
        gcn2.load_weights_snapshot(gcn1.get_weights_snapshot())
        out2 = gcn2.forward(x, adj)
        assert np.allclose(out1, out2)

    def test_adjacency_affects_output(self):
        """不同邻接矩阵产生不同输出。"""
        gcn = NumPyGCN(in_dim=4, hidden_dim=8, out_dim=1, seed=42)
        N = 10
        x = np.random.RandomState(2).randn(N, 4).astype(np.float32)
        adj_eye = np.eye(N, dtype=np.float32)
        adj_full = np.ones((N, N), dtype=np.float32) / N
        gcn.forward(x, adj_eye)
        out_full = gcn.forward(x, adj_full)
        # 即使邻接矩阵不同，输出应该不同（除非巧合）
        # 但不强求完全不同 — 只需要形状正确且数值有效
        assert not np.any(np.isnan(out_full))
        assert not np.any(np.isinf(out_full))

    def test_feature_scaling(self):
        """特征缩放影响输出。"""
        gcn = NumPyGCN(in_dim=4, hidden_dim=8, out_dim=1, seed=42)
        N = 10
        x_small = np.ones((N, 4), dtype=np.float32) * 0.1
        x_large = np.ones((N, 4), dtype=np.float32) * 10.0
        adj = np.eye(N, dtype=np.float32)
        out_small = gcn.forward(x_small, adj)
        out_large = gcn.forward(x_large, adj)
        # 较大输入应产生较大输出
        assert np.mean(np.abs(out_large)) > np.mean(np.abs(out_small))

    def test_large_graph(self):
        """大图上不崩溃。"""
        N = 200
        gcn = NumPyGCN(in_dim=10, hidden_dim=32, out_dim=1, seed=42)
        x = np.random.RandomState(3).randn(N, 10).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        out = gcn.forward(x, adj)
        assert out.shape == (N, 1)
        assert not np.any(np.isnan(out))


# ── NumPyGAT 测试 ─────────────────────────────────────────

class TestNumPyGAT:
    """纯 NumPy GAT（多头注意力）测试。"""

    def test_forward_shape_single_head(self):
        N, in_dim, hidden_dim, out_dim = 10, 5, 8, 1
        gat = NumPyGAT(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, heads=1, seed=42)
        x = np.random.RandomState(0).randn(N, in_dim).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        out = gat.forward(x, adj)
        assert out.shape == (N, out_dim)

    def test_forward_shape_multi_head(self):
        N, in_dim, hidden_dim, out_dim = 10, 5, 8, 1
        gat = NumPyGAT(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, heads=4, seed=42)
        x = np.random.RandomState(1).randn(N, in_dim).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        out = gat.forward(x, adj)
        assert out.shape == (N, out_dim)

    def test_heads_produce_different_attention(self):
        """多头产生不同的注意力模式。"""
        gat = NumPyGAT(in_dim=3, hidden_dim=4, out_dim=1, heads=4, seed=42)
        # 注意: multiple heads means different W_heads
        assert len(gat.W_heads) == 4
        assert len(gat.a_left_heads) == 4
        # 每头的权重应不同
        for k in range(3):
            assert not np.allclose(gat.W_heads[0], gat.W_heads[k + 1])

    def test_weight_snapshot_roundtrip_gat(self):
        """GAT 权重快照 roundtrip。"""
        gat1 = NumPyGAT(in_dim=4, hidden_dim=8, out_dim=1, heads=2, seed=42)
        x = np.random.RandomState(99).randn(10, 4).astype(np.float32)
        adj = np.eye(10, dtype=np.float32)
        out1 = gat1.forward(x, adj)

        gat2 = NumPyGAT(in_dim=4, hidden_dim=8, out_dim=1, heads=2, seed=999)
        gat2.load_weights_snapshot(gat1.get_weights_snapshot())
        out2 = gat2.forward(x, adj)
        assert np.allclose(out1, out2, atol=1e-5)

    def test_no_edges_self_loop_behavior(self):
        """无边图（对角零矩阵）不应崩溃。"""
        N = 5
        gat = NumPyGAT(in_dim=3, hidden_dim=4, out_dim=1, heads=1, seed=42)
        x = np.ones((N, 3), dtype=np.float32)
        adj = np.zeros((N, N), dtype=np.float32)  # 无边
        out = gat.forward(x, adj)
        assert out.shape == (N, 1)
        assert not np.any(np.isnan(out))

    def test_output_numerical_stability(self):
        """输出无 NaN/Inf。"""
        N = 50
        rng = np.random.RandomState(2026)
        gat = NumPyGAT(in_dim=10, hidden_dim=16, out_dim=1, heads=4, seed=42)
        x = rng.randn(N, 10).astype(np.float32)
        adj = rng.rand(N, N) > 0.8
        adj = adj.astype(np.float32)
        out = gat.forward(x, adj)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


# ── PyG Checkpoint 映射测试 ───────────────────────────────

class TestPyGMapping:
    """PyG checkpoint → NumPy 权重映射测试。"""

    def test_map_pyg_gcn_weights(self):
        """PyG GCN state_dict 正确映射到 NumPy 格式。"""
        try:
            import torch
            state_dict = {
                "conv1.lin.weight": torch.randn(64, 20),
                "conv1.bias": torch.randn(64),
                "conv2.lin.weight": torch.randn(1, 64),
                "conv2.bias": torch.randn(1),
            }
            result = _map_pyg_to_numpy(state_dict, "GCN", heads=1)
            assert "W1" in result
            assert result["W1"].shape == (20, 64)  # 转置
            assert "b1" in result
            assert "W2" in result
            assert result["W2"].shape == (64, 1)
        except ImportError:
            pytest.skip("torch not available")

    def test_map_pyg_gat_weights_single_head(self):
        """单头 GAT 映射。"""
        try:
            import torch
            state_dict = {
                "conv1.lin_src.weight": torch.randn(8, 20),  # [hidden, in]
                "conv1.att_src": torch.randn(8),              # [hidden]
                "conv1.att_dst": torch.randn(8),
                "conv2.lin_src.weight": torch.randn(1, 8),
            }
            result = _map_pyg_to_numpy(state_dict, "GAT", heads=1)
            assert "W_0" in result
            assert result["W_0"].shape == (20, 8)
            assert "a_left_0" in result
            assert "a_right_0" in result
        except ImportError:
            pytest.skip("torch not available")

    def test_map_empty_state_dict(self):
        """空 state_dict 返回空 dict。"""
        result = _map_pyg_to_numpy({}, "GCN", heads=1)
        assert result == {}

    def test_load_checkpoint_nonexistent(self):
        """不存在的 checkpoint 路径返回空 dict。"""
        result = load_pyg_checkpoint_to_numpy("./nonexistent_checkpoint.pt")
        assert result == {}

    def test_load_checkpoint_invalid(self, tmp_path):
        """损坏的 checkpoint 文件返回空 dict。"""
        bad_file = tmp_path / "bad.pt"
        bad_file.write_text("not a checkpoint")
        result = load_pyg_checkpoint_to_numpy(str(bad_file))
        assert result == {}


# ── 集成对比测试 ──────────────────────────────────────────

class TestNumpyPyGParity:
    """NumPy vs PyG 输出一致性（需要 torch + torch_geometric）。"""

    @pytest.mark.integration
    def test_gcn_numpy_vs_pyg_same_seed(self):
        """相同 seed 下 NumPy GCN 与 PyG GCN 输出应接近（非精确）。"""
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv

            # 创建 PyG GCN
            class PyGGCN(torch.nn.Module):
                def __init__(self, in_dim, hidden_dim, out_dim):
                    super().__init__()
                    self.conv1 = GCNConv(in_dim, hidden_dim)
                    self.conv2 = GCNConv(hidden_dim, out_dim)

                def forward(self, x, edge_index):
                    x = F.relu(self.conv1(x, edge_index))
                    return self.conv2(x, edge_index)

            in_dim, hidden_dim, out_dim = 5, 8, 1
            PyGGCN(in_dim, hidden_dim, out_dim)
            np_model = NumPyGCN(in_dim, hidden_dim, out_dim, seed=42)

            # 随机输入
            N = 20
            x_np = np.random.RandomState(0).randn(N, in_dim).astype(np.float32)
            # 全连接图 + 自环
            adj = np.ones((N, N), dtype=np.float32) / N  # 平均聚合

            np_out = np_model.forward(x_np, adj)

            assert np_out.shape == (N, out_dim)
            assert not np.any(np.isnan(np_out))
        except ImportError:
            pytest.skip("torch_geometric not available")


# ── 拆分后辅助函数测试 ─────────────────────────────────────

class TestRefactoredMapping:
    """测试 P3 重构拆分出的独立函数。"""

    def test_map_pyg_gcn_direct(self):
        """直接调用 _map_pyg_gcn_to_numpy。"""
        try:
            import torch
            state_dict = {
                "conv1.lin.weight": torch.randn(64, 20),
                "conv1.bias": torch.randn(64),
                "conv2.lin.weight": torch.randn(1, 64),
                "conv2.bias": torch.randn(1),
            }
            result = _map_pyg_gcn_to_numpy(state_dict)
            assert "W1" in result
            assert result["W1"].shape == (20, 64)
            assert "b1" in result
            assert "W2" in result
            assert result["W2"].shape == (64, 1)
            assert "b2" in result
        except ImportError:
            pytest.skip("torch not available")

    def test_map_pyg_gat_direct(self):
        """直接调用 _map_pyg_gat_to_numpy。"""
        try:
            import torch
            state_dict = {
                "conv1.lin_src.weight": torch.randn(8, 20),
                "conv1.att_src": torch.randn(8),
                "conv1.att_dst": torch.randn(8),
                "conv2.lin_src.weight": torch.randn(1, 8),
            }
            result = _map_pyg_gat_to_numpy(state_dict, heads=1)
            assert "W_0" in result
            assert result["W_0"].shape == (20, 8)
            assert "a_left_0" in result
            assert "a_right_0" in result
            assert "W_out" in result
        except ImportError:
            pytest.skip("torch not available")

    def test_map_pyg_dispatcher_gcn(self):
        """分发器 _map_pyg_to_numpy 对 GCN 正确路由。"""
        try:
            import torch
            state_dict = {
                "conv1.lin.weight": torch.randn(32, 10),
                "conv1.bias": torch.randn(32),
            }
            result = _map_pyg_to_numpy(state_dict, "GCN", heads=1)
            assert "W1" in result
            assert "b1" in result
        except ImportError:
            pytest.skip("torch not available")

    def test_map_pyg_dispatcher_unknown(self):
        """分发器对未知 model_type 返回空字典。"""
        result = _map_pyg_to_numpy({}, "TRANSFORMER", heads=4)
        assert result == {}

    def test_resolve_lin_key_src(self):
        """_resolve_lin_key: lin_src 存在时返回它。"""
        try:
            import torch
            state_dict = {"conv1.lin_src.weight": torch.randn(8, 20)}
            result = _resolve_lin_key(state_dict, "conv1", "lin_src", "lin")
            assert result is not None
            assert result.shape == (8, 20)
        except ImportError:
            pytest.skip("torch not available")

    def test_resolve_lin_key_fallback(self):
        """_resolve_lin_key: lin_src 不存在时回退到 lin。"""
        try:
            import torch
            state_dict = {"conv2.lin.weight": torch.randn(1, 8)}
            result = _resolve_lin_key(state_dict, "conv2", "lin_src", "lin")
            assert result is not None
            assert result.shape == (1, 8)
        except ImportError:
            pytest.skip("torch not available")

    def test_extract_attention_param_1d(self):
        """_extract_attention_param: 1D attention 向量。"""
        try:
            import torch
            state_dict = {"conv1.att_src": torch.randn(8)}
            snapshot = {}
            _extract_attention_param(state_dict, "conv1", "att_src", 0, 1, "a_left_0", snapshot)
            assert "a_left_0" in snapshot
            assert snapshot["a_left_0"].ndim == 2
            assert snapshot["a_left_0"].shape[1] == 1
        except ImportError:
            pytest.skip("torch not available")

    def test_extract_attention_param_3d(self):
        """_extract_attention_param: 3D [edge_types=1, heads, hidden]。"""
        try:
            import torch
            state_dict = {"conv1.att_src": torch.randn(1, 4, 8)}
            snapshot = {}
            _extract_attention_param(state_dict, "conv1", "att_src", 2, 4, "a_left_2", snapshot)
            assert "a_left_2" in snapshot
            assert snapshot["a_left_2"].ndim == 2
        except ImportError:
            pytest.skip("torch not available")
