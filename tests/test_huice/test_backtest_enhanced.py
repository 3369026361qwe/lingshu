"""测试 v4.0 增强回测引擎 — VaR回测/DSR-PSR/券商模拟/归因。"""
from decimal import Decimal

from huice.backtest_engine import (
    BacktestEngine,
    DBBacktestRunner,
    PerformanceReport,
    _report_to_dict,
    _var_result_to_dict,
)

# ── Mock 组件 ──────────────────────────────────────────

class MockDataLoader:
    def get_trade_dates(self, start, end):
        return [f"2026{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(60)]

    def load_market_data(self, trade_date):
        return {f"{i:06d}": {"close": 10.0 + i * 0.1 + (hash(trade_date) % 100) * 0.01} for i in range(1, 11)}


class MockSignalGenerator:
    def __init__(self, top_n=5): self.top_n = top_n

    def generate(self, trade_date, market_data, positions):
        if hash(trade_date) % 5 != 0:
            return None
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
                trades.append({
                    "code": s["code"], "direction": s["direction"],
                    "quantity": qty, "amount": float(amount), "price": float(price),
                })
        return trades


# ── BacktestEngine 测试 ──────────────────────────────

class TestBacktestEngine:
    def test_run_basic(self):
        """基本回测运行 — 核心验证."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000,
            "strategy_name": "test_strategy",
            "params": {"top_n": 5},
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
        }
        report = engine.run(config)
        assert report["experiment_id"].startswith("exp_")
        assert "metrics" in report
        assert "daily_records" in report
        assert "trades" in report
        assert len(report["daily_records"]) > 0
        assert report["metrics"]["n_days"] > 0

    def test_run_with_broker_sim(self):
        """使用 MockBroker + OrderManager 模拟执行."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000,
            "strategy_name": "test_broker",
            "params": {"top_n": 5},
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
            "use_broker_sim": True,
            "commission_rate": Decimal("0.0003"),
            "slippage": Decimal("0.0001"),
        }
        report = engine.run(config)
        assert len(report["daily_records"]) > 0
        # 无executor也能工作 (通过券商)
        assert "metrics" in report

    def test_var_backtest_disabled_by_default(self):
        """默认不运行 VaR 回测."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000,
            "strategy_name": "test",
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
        }
        report = engine.run(config)
        assert report.get("var_backtest") is None

    def test_var_backtest_enabled(self):
        """启用 VaR 回测检验."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000,
            "strategy_name": "test",
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
            "enable_var_backtest": True,
        }
        report = engine.run(config)
        # 即使启用, 数据可能不足以运行检验 (需>20obs)
        # 但不应抛异常
        assert "metrics" in report

    def test_snooping_defense_enabled(self):
        """启用 DSR/PSR 防御."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000,
            "strategy_name": "test",
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
            "enable_snooping_defense": True,
            "n_trials": 10,
        }
        report = engine.run(config)
        assert "dsr_pvalue" in report
        assert "psr" in report

    def test_attribution_enabled(self):
        """启用归因分析."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000,
            "strategy_name": "test",
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
            "enable_attribution": True,
        }
        report = engine.run(config)
        # 应不抛异常
        assert "metrics" in report

    def test_performance_report_in_output(self):
        """PerformanceReport 出现在输出中."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000,
            "strategy_name": "test",
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
        }
        report = engine.run(config)
        perf = report.get("performance_report")
        assert perf is not None
        assert "total_return" in perf
        assert "sharpe_ratio" in perf

    def test_compare_experiments(self):
        """实验对比含 v4.0 新字段."""
        engine = BacktestEngine()
        base = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000, "strategy_name": "test",
            "data_loader": MockDataLoader(), "executor": MockExecutor(),
        }
        r1 = engine.run({**base, "signal_generator": MockSignalGenerator(top_n=3)})
        r2 = engine.run({**base, "signal_generator": MockSignalGenerator(top_n=5)})
        comparison = BacktestEngine.compare_experiments([r1, r2])
        assert len(comparison["experiments"]) == 2
        # 检查新字段存在
        assert "dsr_pvalue" in comparison["experiments"][0]

    def test_metrics_computed(self):
        """绩效指标全部计算."""
        engine = BacktestEngine()
        config = {
            "start_date": "20260101", "end_date": "20260301",
            "initial_capital": 1000000, "strategy_name": "test",
            "data_loader": MockDataLoader(),
            "signal_generator": MockSignalGenerator(),
            "executor": MockExecutor(),
        }
        report = engine.run(config)
        m = report["metrics"]
        assert m["total_return"] is not None
        assert m["max_drawdown"] is not None
        assert m["win_rate"] is not None
        assert m["sharpe"] is not None


# ── PerformanceReport 测试 ──────────────────────────

class TestPerformanceReport:
    def test_default_values(self):
        """PerformanceReport 默认值."""
        report = PerformanceReport()
        assert report.total_return == Decimal("0")
        assert report.sharpe_ratio == Decimal("0")
        assert report.deflated_sharpe_ratio == 0.0
        assert report.probabilistic_sharpe_ratio == 0.0
        assert report.var_backtest is None
        assert report.attribution is None

    def test_to_dict(self):
        """_report_to_dict 转换."""
        report = PerformanceReport(
            total_return=Decimal("0.25"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.15"),
            deflated_sharpe_ratio=0.95,
        )
        d = _report_to_dict(report)
        assert d["total_return"] == 0.25
        assert d["sharpe_ratio"] == 1.5
        assert d["max_drawdown"] == 0.15
        assert d["deflated_sharpe_ratio"] == 0.95

    def test_with_var_backtest(self):
        """含 VaR 回测结果的报告."""
        report = PerformanceReport(
            var_95=Decimal("0.05"),
            var_99=Decimal("0.08"),
            var_backtest={
                "n_violations": 3,
                "kupiec_pass": True,
                "basel_zone": "green",
            },
        )
        d = _report_to_dict(report)
        assert d["var_95"] == 0.05
        assert d["var_99"] == 0.08


# ── DBBacktestRunner 测试 ───────────────────────────

class TestDBBacktestRunner:
    def test_runner_creation(self):
        """创建 runner."""
        runner = DBBacktestRunner()
        assert runner is not None

    def test_print_summary(self, capsys):
        """打印摘要."""
        report = {
            "start_date": "20260101", "end_date": "20261231",
            "trading_days": 250, "initial_capital": 1000000,
            "final_value": 1200000, "total_return_pct": 20.0,
            "sharpe": 1.5, "max_drawdown_pct": 10.0,
            "risk_events": 5, "elapsed_seconds": 2.5,
        }
        DBBacktestRunner.print_summary(report)
        captured = capsys.readouterr().out
        assert "回测完成" in captured
        assert "1,000,000" in captured

    def test_print_summary_with_var(self, capsys):
        """含 VaR 回测信息的摘要."""
        report = {
            "start_date": "20260101", "end_date": "20261231",
            "trading_days": 250, "initial_capital": 1000000,
            "final_value": 1200000, "total_return_pct": 20.0,
            "sharpe": 1.5, "max_drawdown_pct": 10.0,
            "risk_events": 5, "elapsed_seconds": 2.5,
            "dsr_pvalue": 0.03, "psr": 0.85,
            "var_backtest": {
                "kupiec_pvalue": 0.35, "basel_zone": "green",
            },
        }
        DBBacktestRunner.print_summary(report)
        captured = capsys.readouterr().out
        assert "DSR" in captured
        assert "PSR" in captured
        assert "VaR检验" in captured


# ── 辅助函数测试 ────────────────────────────────────

class TestHelperFunctions:
    def test_var_result_to_dict_none(self):
        """None → None."""
        assert _var_result_to_dict(None) is None

    def test_report_to_dict_complete(self):
        """完整转换."""
        report = PerformanceReport(
            total_return=Decimal("0.30"),
            annualized_return=Decimal("0.25"),
            annualized_volatility=Decimal("0.15"),
            sharpe_ratio=Decimal("1.67"),
            sortino_ratio=Decimal("2.1"),
            calmar_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.10"),
            win_rate=Decimal("0.60"),
            var_95=Decimal("0.05"),
            var_99=Decimal("0.08"),
            cvar_95=Decimal("0.07"),
            cvar_99=Decimal("0.12"),
            deflated_sharpe_ratio=0.92,
            probabilistic_sharpe_ratio=0.88,
        )
        d = _report_to_dict(report)
        assert all(k in d for k in [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "win_rate",
            "var_95", "var_99", "cvar_95", "cvar_99",
            "deflated_sharpe_ratio", "probabilistic_sharpe_ratio",
        ])
