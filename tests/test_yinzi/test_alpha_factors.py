"""测试Alpha因子: ROC/STD/CORR/MAX/MIN/VMA/CNTP。"""
from decimal import Decimal

from yinzi.alpha_factors import (
    CNTPFactor,
    CORRFactor,
    MAXFactor,
    MINFactor,
    ROCFactor,
    STDFactor,
    VMAFactor,
    create_alpha_factors,
)


def _make_daily(n=120):
    dates = [f"2026{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(n)]
    data = {}
    for i, d in enumerate(dates):
        base = 10.0 + i * 0.05
        data[d] = {"open": base - 0.1, "high": base + 0.3, "low": base - 0.3, "close": base + 0.1, "volume": 10000000 + i * 10000, "amount": 100000000 + i * 100000, "turnover_rate": 1.0 + i * 0.01}
    return data


class TestAlphaFactors:
    def test_roc(self):
        f = ROCFactor(window=5)
        v = f.compute("000001", _make_daily(120))
        assert v is not None and v > 0

    def test_std(self):
        f = STDFactor(window=20)
        v = f.compute("000001", _make_daily(120))
        assert v is not None and v > 0

    def test_corr(self):
        v = CORRFactor(window=20).compute("000001", _make_daily(120))
        assert v is not None and -1 <= float(v) <= 1

    def test_max_factor(self):
        v = MAXFactor(window=20).compute("000001", _make_daily(120))
        assert v is not None

    def test_min_factor(self):
        v = MINFactor(window=20).compute("000001", _make_daily(120))
        assert v is not None

    def test_vma(self):
        v = VMAFactor(window=5).compute("000001", _make_daily(120))
        assert v is not None

    def test_cntp(self):
        v = CNTPFactor(window=20).compute("000001", _make_daily(120))
        assert v is not None and 0 <= float(v) <= 1

    def test_stage1_factors(self):
        """阶段1: RANK/SKEW/TURN/AMP。"""
        from yinzi.alpha_factors import AMPFactor, RANKFactor, SKEWFactor, TURNFactor
        d = _make_daily(120)
        assert RANKFactor(20).compute("000001", d) is not None
        assert TURNFactor(20).compute("000001", d) is not None
        assert AMPFactor(20).compute("000001", d) is not None
        # SKEW需要非恒定收益率 → 使用随机波动数据
        import numpy as np
        rng = np.random.RandomState(42)
        dates = [f"2026{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(200)]
        d2 = {}
        price = 10.0
        for dt in dates:
            price += rng.randn() * 0.1
            d2[dt] = {"open": price - 0.05, "high": price + 0.2, "low": price - 0.2, "close": max(price, 0.1), "volume": 1e7, "amount": 1e8, "turnover_rate": 1.0}
        assert SKEWFactor(30).compute("000001", d2) is not None

    def test_cross_factors(self):
        """阶段3: 交叉因子。"""
        from yinzi.alpha_factors import CrossFactor, ROCFactor, STDFactor
        cf = CrossFactor("test_cross", ROCFactor(20), STDFactor(20), "ratio")
        v = cf.compute("000001", _make_daily(120))
        assert v is not None

    def test_intraday_factors(self):
        """阶段4: 日内因子 VWAP/HLSpread/OC。"""
        from yinzi.alpha_factors import HLSpreadFactor, OCFactor, VWAPFactor
        d = _make_daily(120)
        assert VWAPFactor(20).compute("000001", d) is not None
        assert HLSpreadFactor(20).compute("000001", d) is not None
        assert OCFactor(20).compute("000001", d) is not None

    def test_insufficient_data(self):
        f = ROCFactor(window=200)
        assert f.compute("000001", _make_daily(30)) is None

    def test_factory(self):
        factors = create_alpha_factors()
        assert len(factors) >= 140  # 阶段0~4

    def test_factor_metadata(self):
        for f in create_alpha_factors()[:20]:  # 采样检查20个
            assert f.name and f.category


class TestNDCG:
    def test_perfect_ranking(self):
        from yinzi.factor_validator import FactorValidator
        n = 50
        fv = {f"{i:06d}": Decimal(str(i)) for i in range(n)}
        fr = {f"{i:06d}": Decimal(str(i * 0.01)) for i in range(n)}
        ndcg = FactorValidator.compute_ndcg(fv, fr, k=20)
        assert ndcg is not None and float(ndcg) > 0.9

    def test_ranking_metric(self):
        from yinzi.factor_validator import FactorValidator
        n = 40
        fv = {f"{i:06d}": Decimal(str(i)) for i in range(n)}
        fr = {f"{i:06d}": Decimal(str(i * 0.001)) for i in range(n)}
        rm = FactorValidator.compute_ranking_metric(fv, fr, k=5)
        assert rm is not None


class TestLambdaRank:
    def test_loss_computation(self):
        from juece.lambda_rank import LambdaRankLoss
        lr = LambdaRankLoss(k=5)
        y_pred = __import__('numpy').array([0.5, 0.3, 0.8, 0.1, 0.9])
        y_true = __import__('numpy').array([0.6, 0.2, 0.7, 0.1, 0.95])
        loss = lr.compute_loss(y_pred, y_true)
        assert 0 <= loss <= 1

    def test_gradient_shape(self):
        from juece.lambda_rank import LambdaRankLoss
        lr = LambdaRankLoss(k=5)
        y_pred = __import__('numpy').array([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4])
        y_true = __import__('numpy').array([0.6, 0.2, 0.7, 0.1, 0.95, 0.3, 0.8, 0.5])
        grad = lr.compute_gradient(y_pred, y_true)
        assert grad.shape == (8,)

    def test_perfect_order_zero_loss(self):
        import numpy as np

        from juece.lambda_rank import LambdaRankLoss
        lr = LambdaRankLoss(k=3)
        y = np.array([0.9, 0.7, 0.5, 0.3])
        loss = lr.compute_loss(y, y)
        assert loss < 0.01
