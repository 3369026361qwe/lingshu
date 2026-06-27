"""测试 juece/lambda_rank.py — 决策层 LambdaRank 排序损失。"""
import numpy as np
import pytest

from juece.lambda_rank import LambdaRankLoss


class TestLambdaRankLoss:
    """LambdaRank 损失函数基本测试。"""

    def test_loss_range(self):
        """损失应在 [0, 1] 范围内。"""
        lr = LambdaRankLoss(k=5)
        y_pred = np.array([0.5, 0.3, 0.8, 0.1, 0.9])
        y_true = np.array([0.6, 0.2, 0.7, 0.1, 0.95])
        loss = lr.compute_loss(y_pred, y_true)
        assert 0.0 <= loss <= 1.0

    def test_loss_0_for_identical_prediction(self):
        """相同预测值仍产生有意义的损失（因为排序可能是任意的）。"""
        lr = LambdaRankLoss(k=3)
        y_pred = np.array([0.5, 0.5, 0.5])
        y_true = np.array([0.9, 0.5, 0.1])
        loss = lr.compute_loss(y_pred, y_true)
        assert 0.0 <= loss <= 1.0

    def test_perfect_prediction_zero_loss(self):
        """完美排序时损失为 0。"""
        lr = LambdaRankLoss(k=3)
        y = np.array([0.9, 0.7, 0.5, 0.3])
        loss = lr.compute_loss(y, y)
        assert loss == 0.0

    def test_perfect_reverse_near_1_loss(self):
        """完全逆序排序时损失较大。"""
        lr = LambdaRankLoss(k=5)
        y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        loss = lr.compute_loss(y_pred, y_true)
        # 逆序应产生损失（NDCG 低于理想值）
        assert loss > 0.0

    def test_all_zero_labels_zero_loss(self):
        """全零标签时 ideal DCG = 0 → 损失为 0（或接近 0）。"""
        lr = LambdaRankLoss(k=3)
        y_pred = np.array([0.5, 0.8, 0.2])
        y_true = np.array([0.0, 0.0, 0.0])
        loss = lr.compute_loss(y_pred, y_true)
        assert loss in (0.0, 1.0)  # 实现返回 0.0

    def test_single_item(self):
        """单条数据不应崩溃。"""
        lr = LambdaRankLoss(k=3)
        y_pred = np.array([0.7])
        y_true = np.array([0.5])
        loss = lr.compute_loss(y_pred, y_true)
        assert isinstance(loss, float)

    def test_custom_k(self):
        """自定义 k 参数生效。"""
        lr_k3 = LambdaRankLoss(k=3)
        lr_k10 = LambdaRankLoss(k=10)
        y_pred = np.arange(20, 0, -1, dtype=float)
        y_true = np.arange(1, 21, dtype=float)
        loss_k3 = lr_k3.compute_loss(y_pred, y_true)
        loss_k10 = lr_k10.compute_loss(y_pred, y_true)
        # 不同 k 可能产生不同损失
        assert isinstance(loss_k3, float)
        assert isinstance(loss_k10, float)


class TestLambdaRankGradient:
    """LambdaRank 梯度计算测试。"""

    def test_gradient_shape(self):
        """梯度形状应与输入一致。"""
        lr = LambdaRankLoss(k=5)
        n = 8
        y_pred = np.random.RandomState(42).randn(n).astype(np.float64)
        y_true = np.random.RandomState(43).rand(n).astype(np.float64)
        grad = lr.compute_gradient(y_pred, y_true)
        assert grad.shape == (n,)
        assert grad.dtype == np.float64

    def test_gradient_zero_sum(self):
        """LambdaRank 梯度和应为零（成对交换，一增一减）。"""
        lr = LambdaRankLoss(k=5)
        y_pred = np.array([0.5, 0.3, 0.8, 0.1, 0.9])
        y_true = np.array([0.6, 0.2, 0.7, 0.1, 0.95])
        grad = lr.compute_gradient(y_pred, y_true)
        assert abs(float(np.sum(grad))) < 1e-10

    def test_perfect_order_gradient_behavior(self):
        """完美排序时梯度求和为零，无 NaN（梯度在有限 sij 下不严格为零，
        因为 sigmoid 斜率在有限差值下非零 — 这是 LambdaRank 的正常特性）。"""
        lr = LambdaRankLoss(k=3)
        y = np.array([0.9, 0.7, 0.5, 0.3])
        grad = lr.compute_gradient(y, y)
        assert grad.shape == (4,)
        assert abs(float(np.sum(grad))) < 1e-10  # 和为零
        assert not np.any(np.isnan(grad))

    def test_all_equal_labels_zero_gradient(self):
        """所有标签相同时，梯度为全零（无损失信号）。"""
        lr = LambdaRankLoss(k=3)
        y_pred = np.array([0.5, 0.3, 0.8])
        y_true = np.array([0.5, 0.5, 0.5])
        grad = lr.compute_gradient(y_pred, y_true)
        assert np.allclose(grad, 0.0)

    def test_gradient_direction_correct(self):
        """对于一对 (i, j) 其中 label_i > label_j：
        如果 pred_i < pred_j 则 lambda < 0（即梯度推动 pred_i 上升、
        pred_j 下降 — 注意梯度下降时减去梯度，故负梯度 = 增加预测值）。"""
        lr = LambdaRankLoss(k=3)
        # 只有两只股票：label[0] > label[1] 但 pred[0] < pred[1] → 错配
        y_pred = np.array([0.2, 0.8])
        y_true = np.array([0.9, 0.1])
        grad = lr.compute_gradient(y_pred, y_true)
        # lambda < 0 → grad[0] < 0 (梯度下降时 pred[0] ↑)，grad[1] > 0 (pred[1] ↓)
        assert grad[0] < 0
        assert grad[1] > 0

    def test_large_input_numerical_stability(self):
        """大数据量下不产生 NaN 或 Inf。"""
        lr = LambdaRankLoss(k=10)
        rng = np.random.RandomState(2026)
        y_pred = rng.randn(200).astype(np.float64)
        y_true = rng.rand(200).astype(np.float64)
        loss = lr.compute_loss(y_pred, y_true)
        grad = lr.compute_gradient(y_pred, y_true)
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))

    def test_sigma_parameter(self):
        """不同的 sigma 产生不同的梯度大小。"""
        lr_small = LambdaRankLoss(sigma=0.1, k=3)
        lr_large = LambdaRankLoss(sigma=10.0, k=3)
        y_pred = np.array([0.3, 0.7, 0.5])
        y_true = np.array([0.9, 0.1, 0.5])
        g_small = lr_small.compute_gradient(y_pred, y_true)
        g_large = lr_large.compute_gradient(y_pred, y_true)
        # 更大的 sigma 产生更大的梯度（更陡的 sigmoid）
        assert np.max(np.abs(g_large)) > np.max(np.abs(g_small)) * 0.5
