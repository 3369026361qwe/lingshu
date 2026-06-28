"""
测试卡尔曼滤波 + 因子验证器。
"""

from decimal import Decimal

import pytest

from yinzi.factor_validator import FactorValidator
from yinzi.kalman_weight import KalmanWeightEstimator


class TestKalmanFilter:
    def test_initial_equal_weights(self):
        kf = KalmanWeightEstimator(5)
        weights = kf.weights
        assert len(weights) == 5
        assert all(abs(float(w) - 0.2) < 0.01 for w in weights)

    def test_single_update(self):
        kf = KalmanWeightEstimator(3)
        initial = kf.weights
        kf.update([Decimal("0.1"), Decimal("0.2"), Decimal("-0.05")], Decimal("0.015"))
        updated = kf.weights
        # 权重应已变化
        assert any(abs(float(updated[i]) - float(initial[i])) > 0 for i in range(3))

    def test_batch_update_converges(self):
        kf = KalmanWeightEstimator(3, process_noise=1e-6, measurement_noise=1e-4)
        for _ in range(100):
            kf.update(
                [Decimal("0.1"), Decimal("0.05"), Decimal("-0.02")],
                Decimal("0.012"),
            )
        weights = kf.weights
        # 权重应趋于稳定
        assert all(abs(float(weights[i])) < 10 for i in range(3))
        assert kf.update_count == 100

    def test_variances_decrease(self):
        kf = KalmanWeightEstimator(3)
        initial_var = max(float(v) for v in kf.variances)
        for _ in range(50):
            kf.update(
                [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")],
                Decimal("0.01"),
            )
        final_var = max(float(v) for v in kf.variances)
        # 方差应减小
        assert final_var < initial_var

    def test_dimension_mismatch(self):
        kf = KalmanWeightEstimator(3)
        with pytest.raises(ValueError):
            kf.update([Decimal("0.1"), Decimal("0.2")], Decimal("0.01"))

    def test_batch_update(self):
        kf = KalmanWeightEstimator(2)
        F = [[Decimal("0.1"), Decimal("0.2")] for _ in range(50)]
        r = [Decimal("0.01") for _ in range(50)]
        kf.batch_update(F, r)
        assert kf.update_count == 50

    def test_zero_innovation_skip(self):
        """S=0 时不应崩溃。"""
        kf = KalmanWeightEstimator(2, measurement_noise=0)
        # 使用零暴露 → FPF = 0, S = R = 0 → 跳过
        weights = kf.update([Decimal("0"), Decimal("0")], Decimal("0"))
        assert len(weights) == 2


class TestFactorValidator:
    def test_rank_ic_perfect_positive(self):
        n = 100
        fv = {f"{i:06d}": Decimal(str(i)) for i in range(n)}
        fr = {f"{i:06d}": Decimal(str(i * 0.01)) for i in range(n)}
        ic = FactorValidator.compute_rank_ic(fv, fr)
        assert ic is not None
        assert ic > 0.9  # 近乎完美的正相关

    def test_rank_ic_perfect_negative(self):
        n = 100
        fv = {f"{i:06d}": Decimal(str(i)) for i in range(n)}
        fr = {f"{i:06d}": Decimal(str((n - i) * 0.01)) for i in range(n)}
        ic = FactorValidator.compute_rank_ic(fv, fr)
        assert ic is not None
        assert ic < -0.9

    def test_rank_ic_small_sample(self):
        fv = {f"{i:06d}": Decimal("1") for i in range(10)}
        fr = {f"{i:06d}": Decimal("1") for i in range(10)}
        ic = FactorValidator.compute_rank_ic(fv, fr)
        assert ic is None  # 样本不足

    def test_ir_positive(self):
        # 使用有变化的 IC 序列（否则 std=0 → IR 未定义）
        ic_series = [Decimal(str(0.02 + i * 0.001)) for i in range(20)]
        ir = FactorValidator.compute_ir(ic_series)
        assert ir is not None

    def test_layered_backtest(self):
        n = 100
        fv = {f"{i:06d}": Decimal(str(i)) for i in range(n)}
        fr = {f"{i:06d}": Decimal(str(i * 0.001)) for i in range(n)}
        layers = FactorValidator.layered_backtest(fv, fr, n_groups=5)
        assert len(layers) == 5
        # 顶层收益应大于底层
        assert layers[-1]["avg_return"] > layers[0]["avg_return"]

    def test_validate_comprehensive(self):
        n = 100
        fv = {f"{i:06d}": Decimal(str(i)) for i in range(n)}
        fr = {f"{i:06d}": Decimal(str(i * 0.001)) for i in range(n)}
        report = FactorValidator.validate("test_factor", fv, fr)
        assert report["factor_name"] == "test_factor"
        assert report["ic"] is not None
        assert len(report["layers"]) > 0
