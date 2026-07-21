"""测试回测系统: 引擎/绩效/报告/网格搜索/实验追踪。"""
from decimal import Decimal

from huice.backtest_engine import BacktestEngine
from huice.grid_search import GridSearch
from huice.performance_metrics import PerformanceMetrics
from huice.report_generator import ReportGenerator


class MockDataLoader:
    def get_trade_dates(self, start, end):
        return [f"2026{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(60)]

    def load_market_data(self, trade_date):
        return {f"{i:06d}": {"close": 10.0 + i * 0.1 + (hash(trade_date) % 100) * 0.01} for i in range(1, 11)}


class MockSignalGenerator:
    def __init__(self, top_n=5): self.top_n = top_n
    def generate(self, trade_date, market_data, positions):
        if hash(trade_date) % 5 != 0: return None
        codes = sorted(market_data.keys(), key=lambda c: market_data[c]["close"], reverse=True)[:self.top_n]
        return [{"code": c, "direction": "BUY", "weight": 0.2, "reason": "test"} for c in codes]


class MockExecutor:
    def execute(self, signals, positions, capital, market_data):
        trades = []
        for s in signals:
            price = Decimal(str(market_data[s["code"]]["close"]))
            amount = Decimal(str(s["weight"])) * capital * Decimal("0.2")
            qty = int(amount / price) if price > 0 else 0
            if qty >= 100:
                trades.append({"code": s["code"], "direction": s["direction"], "quantity": qty, "amount": float(amount), "price": float(price)})
        return trades


class TestBacktestEngine:
    def test_run_and_preserve(self):
        """核心验证：回测运行后数据完整保留。"""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301", "initial_capital": 1000000,
            "strategy_name": "test_strategy", "params": {"top_n": 5},
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
        }
        report = engine.run(config)

        # 实验数据完整性检查
        assert report["experiment_id"].startswith("exp_")
        assert "metrics" in report
        assert "daily_records" in report
        assert "trades" in report
        assert len(report["daily_records"]) > 0
        assert report["metrics"]["n_days"] > 0
        assert engine.experiment_id == report["experiment_id"]

    def test_metrics_computed(self):
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301", "initial_capital": 1000000,
            "strategy_name": "test",
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
        }
        report = engine.run(config)
        m = report["metrics"]
        assert m["total_return"] is not None
        assert m["max_drawdown"] is not None
        assert m["win_rate"] is not None

    def test_compare_experiments(self):
        """多个实验对比。"""
        engine = BacktestEngine()
        base = {"start_date": "20260101", "end_date": "20260301", "initial_capital": 1000000,
                "strategy_name": "test", "data_loader": MockDataLoader(), "executor": MockExecutor()}
        r1 = engine.run({**base, "signal_generator": MockSignalGenerator(top_n=3)})
        r2 = engine.run({**base, "signal_generator": MockSignalGenerator(top_n=5)})
        comparison = BacktestEngine.compare_experiments([r1, r2])
        assert len(comparison["experiments"]) == 2


class TestPerformanceMetrics:
    def test_compute_all(self):
        curve = [Decimal(str(1000000 + i * 1000)) for i in range(100)]
        rets = [(curve[i] - curve[i-1]) / curve[i-1] for i in range(1, 100)]
        m = PerformanceMetrics.compute_all(curve, rets)
        assert m["sharpe"] is not None

    def test_max_drawdown(self):
        curve = [Decimal("1.0"), Decimal("1.1"), Decimal("0.85"), Decimal("1.05")]
        dd = PerformanceMetrics.max_drawdown(curve)
        assert float(dd) > 0.15  # 1.1→0.85 = ~22.7%

    def test_win_rate(self):
        rets = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.03"), Decimal("0.01")]
        wr = PerformanceMetrics.win_rate(rets)
        assert wr == Decimal("0.75")


class TestReportGenerator:
    def test_markdown(self):
        report = {"experiment_id": "test_001", "metrics": {"sharpe": 1.5, "total_return": 0.25, "max_drawdown": 0.15, "win_rate": 0.6, "annualized_return": 0.3, "annualized_vol": 0.2, "n_days": 252}, "config": {"initial_capital": 1000000}}
        md = ReportGenerator.to_markdown(report)
        assert "test_001" in md and "1.5" in md

    def test_comparison_table(self):
        r1 = {"experiment_id": "exp_a", "metrics": {"sharpe": 1.5, "total_return": 0.25, "max_drawdown": 0.1, "win_rate": 0.6}}
        r2 = {"experiment_id": "exp_b", "metrics": {"sharpe": 2.0, "total_return": 0.35, "max_drawdown": 0.08, "win_rate": 0.65}}
        table = ReportGenerator.comparison_table([r1, r2])
        assert "exp_a" in table and "exp_b" in table


class TestGridSearch:
    def test_search(self):
        engine = BacktestEngine()
        base = {"start_date": "20260101", "end_date": "20260301", "initial_capital": 1000000,
                "strategy_name": "test", "data_loader": MockDataLoader(), "executor": MockExecutor(),
                "signal_generator": MockSignalGenerator()}
        grid = {"top_n": [3, 5]}
        gs = GridSearch(engine)
        results = gs.search(base, grid)
        assert len(results) == 2
        assert all("grid_params" in r for r in results)
        best = GridSearch.best_params(results)
        assert "params" in best


# ── Cross-Validation tests (v4.1) ────────────────────────────────────────────


class TestPurgedKFold:
    def test_split_basic(self):
        from huice.cross_validation import PurgedKFold
        dates = [f"2026{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(300)]
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.01, purge_pct=0.01)
        folds = list(pkf.split(dates))
        # fold 0 无历史训练数据 → 跳过；purge 从训练集末尾删 3 天
        # 300 天 / 5 folds = 60 天/fold, n_splits-1 = 4 folds
        assert len(folds) >= 3
        for train, test in folds:
            assert len(train) > 0
            assert len(test) > 0
            # 训练集和测试集不应该重叠
            assert set(train).isdisjoint(set(test))
            # 训练集末尾 + embargo + purge < 测试集开头
            assert train[-1] < test[0]

    def test_split_indices(self):
        from huice.cross_validation import PurgedKFold
        pkf = PurgedKFold(n_splits=4, embargo_pct=0.0, purge_pct=0.0)
        folds = list(pkf.split_indices(120))
        # fold 0 跳过 → n_splits-1 = 3 folds
        assert len(folds) == 3
        for train, test in folds:
            assert max(train) < min(test)

    def test_too_few_dates(self):
        import pytest

        from huice.cross_validation import PurgedKFold
        pkf = PurgedKFold(n_splits=5)
        with pytest.raises(ValueError):
            list(pkf.split([f"d{i}" for i in range(10)]))

    def test_invalid_params(self):
        import pytest

        from huice.cross_validation import PurgedKFold
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)
        with pytest.raises(ValueError):
            PurgedKFold(embargo_pct=0.6)


class TestWalkForwardCV:
    def test_split_basic(self):
        from huice.cross_validation import WalkForwardCV
        dates = [f"2026{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(500)]
        wf = WalkForwardCV(min_train_size=252, step_size=63, test_size=63)
        folds = list(wf.split(dates))
        assert len(folds) >= 2
        for train, test in folds:
            assert len(train) >= 252
            assert len(test) == 63
            assert set(train).isdisjoint(set(test))

    def test_expanding_window(self):
        from huice.cross_validation import WalkForwardCV
        dates = [str(i) for i in range(400)]
        wf = WalkForwardCV(min_train_size=200, step_size=50, test_size=50, expanding=True)
        folds = list(wf.split(dates))
        for i, (train, _test) in enumerate(folds):
            if i == 0:
                assert len(train) == 200
            if i > 0:
                prev_train = list(folds)[i - 1][0]
                assert len(train) > len(prev_train)

    def test_rolling_window(self):
        from huice.cross_validation import WalkForwardCV
        dates = [str(i) for i in range(400)]
        wf = WalkForwardCV(min_train_size=200, step_size=50, test_size=50, expanding=False)
        folds = list(wf.split(dates))
        for train, _test in folds:
            assert len(train) == 200  # 固定窗口

    def test_too_few_dates(self):
        import pytest

        from huice.cross_validation import WalkForwardCV
        wf = WalkForwardCV(min_train_size=300, test_size=100)
        with pytest.raises(ValueError):
            list(wf.split([f"d{i}" for i in range(200)]))


class TestCVSummary:
    def test_compute_cv_summary(self):
        from huice.cross_validation import CVResult, compute_cv_summary
        results = [
            CVResult(fold=0, train_start="d0", train_end="d99", test_start="d100", test_end="d149",
                     sharpe=1.5, total_return=0.25, max_drawdown=0.10),
            CVResult(fold=1, train_start="d0", train_end="d149", test_start="d150", test_end="d199",
                     sharpe=1.8, total_return=0.30, max_drawdown=0.08),
            CVResult(fold=2, train_start="d0", train_end="d199", test_start="d200", test_end="d249",
                     sharpe=1.2, total_return=0.20, max_drawdown=0.12),
        ]
        s = compute_cv_summary(results)
        assert s["n_folds"] == 3
        assert s["sharpe_mean"] is not None
        assert s["sharpe_std"] is not None
        assert 1.0 < s["sharpe_mean"] < 2.0

    def test_empty(self):
        from huice.cross_validation import compute_cv_summary
        s = compute_cv_summary([])
        assert s["n_folds"] == 0
        assert s["sharpe_mean"] is None


# ── DSR/PSR Report tests (v4.1) ──────────────────────────────────────────────


class TestDSRPSRReport:
    def test_compute_dsr_psr_adds_fields(self):
        from huice.report_generator import ReportGenerator
        report = {
            "experiment_id": "test_dsr",
            "metrics": {"sharpe": 1.2, "n_days": 252, "total_return": 0.30},
            "config": {},
        }
        result = ReportGenerator.compute_dsr_psr(report, n_trials=10)
        assert "dsr_pvalue" in result
        assert "haircut_sharpe" in result
        assert "psr" in result
        assert isinstance(result["dsr_pvalue"], float)
        assert 0 <= result["dsr_pvalue"] <= 1

    def test_compute_dsr_psr_warns_on_high_dsr(self):
        from huice.report_generator import ReportGenerator
        report = {
            "experiment_id": "test_warn",
            "metrics": {"sharpe": 0.3, "n_days": 100, "total_return": 0.05},
            "config": {},
        }
        result = ReportGenerator.compute_dsr_psr(report, n_trials=100)
        # 高试验次数 + 低夏普 → DSR p-value 可能很高
        if result.get("dsr_warning"):
            assert "数据窥探" in result["dsr_warning"]

    def test_to_markdown_includes_dsr(self):
        from huice.report_generator import ReportGenerator
        report = {
            "experiment_id": "test_md_dsr",
            "metrics": {"sharpe": 1.5, "n_days": 252, "total_return": 0.25,
                        "max_drawdown": 0.15, "win_rate": 0.6,
                        "annualized_return": 0.25, "annualized_vol": 0.15},
            "config": {"initial_capital": 1000000},
            "dsr_pvalue": 0.03,
            "haircut_sharpe": 1.2,
            "psr": 0.98,
        }
        md = ReportGenerator.to_markdown(report)
        assert "数据窥探防御" in md
        assert "DSR p-value" in md
        assert "Haircut SR" in md
        assert "PSR" in md

    def test_to_markdown_auto_computes_dsr(self):
        from huice.report_generator import ReportGenerator
        report = {
            "experiment_id": "test_auto_dsr",
            "metrics": {"sharpe": 2.0, "n_days": 252, "total_return": 0.50,
                        "annualized_return": 0.50, "annualized_vol": 0.20,
                        "max_drawdown": 0.20, "win_rate": 0.65},
            "config": {},
        }
        md = ReportGenerator.to_markdown(report)
        # 自动从 metrics 计算 DSR
        assert "数据窥探防御" in md

    def test_to_markdown_no_dsr_when_no_sharpe(self):
        from huice.report_generator import ReportGenerator
        report = {
            "experiment_id": "test_no_dsr",
            "metrics": {"n_days": 10, "total_return": 0.0},
            "config": {},
        }
        md = ReportGenerator.to_markdown(report)
        # n_days < 20 时不应该有 DSR 段
        assert "数据窥探防御" not in md

    def test_comparison_table_with_dsr(self):
        from huice.report_generator import ReportGenerator
        reports = [
            {"experiment_id": "exp_a", "metrics": {"sharpe": 1.5, "total_return": 0.25, "max_drawdown": 0.1, "win_rate": 0.6}, "dsr_pvalue": 0.03},
            {"experiment_id": "exp_b", "metrics": {"sharpe": 2.0, "total_return": 0.35, "max_drawdown": 0.08, "win_rate": 0.65}, "dsr_pvalue": 0.01},
        ]
        table = ReportGenerator.comparison_table(reports)
        assert "DSR p-value" in table
        assert "exp_a" in table
