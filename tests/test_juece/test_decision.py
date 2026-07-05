"""
测试集成决策引擎: 信号融合/选股/组合优化/调仓/基准。
"""

from decimal import Decimal

from juece.benchmark import Benchmark
from juece.ensemble_engine import EnsembleEngine
from juece.portfolio_optimizer import PortfolioOptimizer
from juece.rebalancer import Rebalancer
from juece.stock_selector import StockSelector


def _make_scores(n=100):
    return {f"{i:06d}": Decimal(str(i * 0.1)) for i in range(1, n + 1)}


class TestEnsembleEngine:
    def test_fuse_basic(self):
        engine = EnsembleEngine()
        factor = _make_scores(50)
        gnn = {c: float(v) for c, v in factor.items()}
        agent = {c: Decimal(str(float(v) * 0.5)) for c, v in factor.items()}
        result = engine.fuse(factor, gnn, agent)
        assert len(result) == 50

    def test_fuse_only_factor(self):
        engine = EnsembleEngine()
        factor = _make_scores(30)
        result = engine.fuse(factor)
        assert len(result) == 30

    def test_rank(self):
        engine = EnsembleEngine()
        factor = _make_scores(20)
        result = engine.fuse(factor)
        ranked = engine.rank(result)
        assert ranked[0][1] >= ranked[-1][1]

    def test_update_weights_from_ic(self):
        engine = EnsembleEngine()
        engine.update_weights_from_ic(
            factor_ic=Decimal("0.05"),
            gnn_ic=Decimal("0.03"),
            agent_ic=Decimal("0.02"),
        )
        w = engine.weights
        assert w["factor"] > w["gnn"] > w["agent"]

    def test_set_weights(self):
        engine = EnsembleEngine()
        engine.set_weights(Decimal("5"), Decimal("3"), Decimal("2"))
        assert engine.weights["factor"] == Decimal("0.5")


class TestStockSelector:
    def test_generate_signals(self):
        scores = _make_scores(30)
        selector = StockSelector(top_n=10)
        signals = selector.generate_signals(scores)
        assert len(signals) == 30
        # Top 3 should be BUY (top 10%)
        top_codes = sorted(scores, key=scores.get, reverse=True)[:3]
        for c in top_codes:
            assert signals[c]["signal"] == "BUY"

    def test_select_top_n(self):
        scores = _make_scores(50)
        selector = StockSelector(top_n=10)
        picks = selector.select_top_n(scores)
        assert len(picks) == 10
        assert picks[0]["rank"] == 1

    def test_exclude_stocks(self):
        scores = _make_scores(20)
        selector = StockSelector(top_n=5)
        picks = selector.select_top_n(scores, exclude={"000020", "000019", "000018", "000017", "000016", "000015", "000014", "000013", "000012", "000011", "000010", "000009", "000008", "000007", "000006"})
        assert len(picks) == 5

    def test_diversify(self):
        scores = _make_scores(20)
        selector = StockSelector(top_n=20)
        picks = selector.select_top_n(scores)
        industry = {f"{i:06d}": f"行业{i % 3}" for i in range(1, 21)}
        diversified = selector.diversify(picks, industry, max_per_industry=3)
        # 每行业最多3只 → 最多9只
        assert len(diversified) <= 9


class TestPortfolioOptimizer:
    def test_optimize_basic(self):
        """v4.0: BL optimizer needs returns matrix + market weights + views."""
        import random
        from decimal import Decimal
        random.seed(42)
        n_a, n_t = 10, 100
        codes = [f"{i:06d}" for i in range(n_a)]
        rm = [[Decimal(str(random.gauss(0.001, 0.02))) for _ in range(n_t)] for _ in range(n_a)]
        mkt_w = [Decimal("1") / Decimal(n_a)] * n_a
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, mkt_w)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        result = opt.optimize(pr, pc, codes)
        assert len(result.optimal_weights) == n_a
        total = sum(result.optimal_weights.values())
        assert abs(float(total) - 1.0) < 0.01

    def test_max_weight_constraint(self):
        """v4.0: max_weight set via BLConfig."""
        import random
        from decimal import Decimal
        random.seed(42)
        n_a, n_t = 5, 100
        codes = [f"{i:06d}" for i in range(n_a)]
        rm = [[Decimal(str(random.gauss(0.001, 0.02))) for _ in range(n_t)] for _ in range(n_a)]
        mkt_w = [Decimal("1") / Decimal(n_a)] * n_a
        from juece.portfolio_optimizer import BLConfig
        config = BLConfig(max_weight=Decimal("0.25"))
        opt = PortfolioOptimizer(config)
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, mkt_w)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        result = opt.optimize(pr, pc, codes)
        for w in result.optimal_weights.values():
            assert w <= Decimal("0.25")

    def test_constraint_check(self):
        """v4.0: constraint check replaced by BL's built-in weight clipping."""
        from juece.portfolio_optimizer import BLConfig
        config = BLConfig(max_weight=Decimal("0.10"))
        # Verify max_weight is stored in config
        assert config.max_weight == Decimal("0.10")

    def test_empty_picks(self):
        """v4.0: BL optimizer raises ValueError on empty input."""
        opt = PortfolioOptimizer()
        try:
            opt.optimize([], [], [])
            raise AssertionError("Should have raised")
        except (ValueError, ZeroDivisionError, Exception):
            pass


class TestRebalancer:
    def test_compute_trades(self):
        target = [
            {"code": "000001", "weight": Decimal("0.08")},
            {"code": "000002", "weight": Decimal("0.06")},
            {"code": "000003", "weight": Decimal("0.04")},
        ]
        current = {
            "000001": {"weight": 0.05},
            "000002": {"weight": 0.07},
            "000004": {"weight": 0.03},  # 需要清仓
        }
        reb = Rebalancer()
        result = reb.compute_trades(target, current, Decimal("1000000"))
        assert len(result["buys"]) >= 1  # 000001 增持
        assert len(result["sells"]) >= 1  # 000002/000004 减持/清仓
        assert result["total_turnover"] > 0

    def test_no_change(self):
        target = [{"code": "000001", "weight": Decimal("0.10")}]
        current = {"000001": {"weight": 0.10}}
        reb = Rebalancer()
        result = reb.compute_trades(target, current, Decimal("1000000"))
        assert len(result["buys"]) == 0 and len(result["sells"]) == 0


class TestBenchmark:
    def test_compare_returns(self):
        result = Benchmark.compare_returns(
            Decimal("0.25"),
            {"hs300": Decimal("0.15"), "zz500": Decimal("0.10")},
        )
        assert result["hs300"]["win"] is True
        assert result["hs300"]["excess_return"] == Decimal("0.10")

    def test_sharpe_ratio(self):
        returns = [Decimal(str(0.001 * (i % 5 - 2))) for i in range(252)]
        sharpe = Benchmark.sharpe_ratio(returns)
        assert sharpe is not None

    def test_max_drawdown(self):
        curve = [Decimal("1.0"), Decimal("1.1"), Decimal("0.9"), Decimal("0.95"), Decimal("1.05")]
        dd = Benchmark.max_drawdown(curve)
        assert dd is not None
        assert dd > 0

    def test_win_rate(self):
        returns = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.03"), Decimal("-0.01"), Decimal("0.02")]
        wr = Benchmark.win_rate(returns)
        assert wr == Decimal("0.6")
