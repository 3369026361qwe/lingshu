"""测试回测系统: 引擎/绩效/报告/网格搜索/实验追踪。"""
from decimal import Decimal
import pytest
from huice.backtest_engine import BacktestEngine
from huice.performance_metrics import PerformanceMetrics
from huice.report_generator import ReportGenerator
from huice.grid_search import GridSearch


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
