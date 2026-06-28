"""测试 huice/backtest_engine.py — DBBacktestRunner。

使用项目标准测试数据库（conftest.py 配置的 test_lingshu.db）。
测试 fixture 负责建表、预填 minimal 行情/信号数据，测试后清理。
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from huice.backtest_engine import BacktestEngine, DBBacktestRunner
from shujuku.models import Base
from shujuku.session import SessionContext, _engine

# ── Fixtures ──────────────────────────────────────────────

def _now() -> str:
    """返回当前 ISO 时间戳。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _ensure_tables():
    """确保所有表已创建（幂等，仅创建不存在的表）。"""
    Base.metadata.create_all(_engine)


@pytest.fixture
def seeded_db():
    """预填 10 只股票 × 120 天的 daily_bar 和 fusion_score 测试数据。"""
    _ensure_tables()
    codes = [f"{i:06d}" for i in range(1, 11)]  # 10 只股票
    now = _now()

    with SessionContext() as s:
        # 预填 daily_bar
        for code in codes:
            for day_offset in range(120):
                td = f"2026{(day_offset // 30) + 1:02d}{(day_offset % 30) + 1:02d}"
                base_price = 10.0 + (int(code) - 1) * 5.0 + day_offset * 0.02
                noise = (hash(code + td) % 100) / 1000.0
                close = round(base_price + noise, 4)
                s.execute(text(
                    "INSERT OR IGNORE INTO daily_bar "
                    "(code, trade_date, open, high, low, close, volume, amount, is_st, updated_at) "
                    "VALUES (:c, :d, :o, :h, :l, :c2, 10000, 100000, 0, :now)"
                ), {
                    "c": code, "d": td,
                    "o": close - 0.1, "h": close + 0.2, "l": close - 0.2, "c2": close,
                    "now": now,
                })
        s.commit()

    with SessionContext() as s:
        # 预填 fusion_score（top-3 股票 signal=1，其余 signal=0）
        for day_offset in range(120):
            td = f"2026{(day_offset // 30) + 1:02d}{(day_offset % 30) + 1:02d}"
            for rank, code in enumerate(codes, 1):
                score = round(1.0 - (rank - 1) * 0.1, 2)
                signal = 1.0 if rank <= 3 else 0.0
                s.execute(text(
                    "INSERT OR IGNORE INTO fusion_score "
                    "(trade_date, code, composite_score, rank, signal, created_at) "
                    "VALUES (:d, :c, :sc, :r, :sig, :now)"
                ), {"d": td, "c": code, "sc": score, "r": rank, "sig": signal, "now": now})
        s.commit()

    yield

    # 清理测试数据
    with SessionContext() as s:
        s.execute(text("DELETE FROM daily_bar WHERE code LIKE '00000%' OR code LIKE '00001%'"))
        s.execute(text("DELETE FROM fusion_score WHERE code LIKE '00000%' OR code LIKE '00001%'"))
        s.execute(text("DELETE FROM portfolio_snapshot"))
        s.commit()


# ── Tests ─────────────────────────────────────────────────

class TestDBBacktestRunner:
    """DBBacktestRunner 核心功能测试。"""

    def test_run_with_default_params(self, seeded_db):
        """默认参数下回测运行成功，返回完整报告。"""
        runner = DBBacktestRunner()
        report = runner.run(
            start_date="20260101",
            end_date="20260430",
            top_n=5,
            rebalance_freq=20,
            initial_capital=1_000_000,
            persist=False,
        )
        assert report is not None
        assert "start_date" in report
        assert "final_value" in report
        assert "total_return_pct" in report
        assert "sharpe" in report
        assert "snapshots" in report
        assert report["trading_days"] > 0

    def test_run_without_dates(self, seeded_db):
        """无 start_date/end_date 限制时，使用全部数据。"""
        runner = DBBacktestRunner()
        report = runner.run(top_n=3, rebalance_freq=40, persist=False)
        assert report["trading_days"] > 0
        assert isinstance(report["final_value"], float)

    def test_top_n_respected(self, seeded_db):
        """top_n 参数控制最大持仓数。"""
        runner = DBBacktestRunner()
        report = runner.run(top_n=3, rebalance_freq=10, persist=False)
        max_pc = max((s.get("pc", 0) for s in report["snapshots"]), default=0)
        assert max_pc <= 3

    def test_initial_capital_scales(self, seeded_db):
        """不同初始资金产生成比例的终值。"""
        runner = DBBacktestRunner()
        r_1m = runner.run(initial_capital=1_000_000, top_n=3, rebalance_freq=999, persist=False)
        r_2m = runner.run(initial_capital=2_000_000, top_n=3, rebalance_freq=999, persist=False)
        if r_1m["final_value"] > 0:
            ratio = r_2m["final_value"] / r_1m["final_value"]
            assert 1.8 < ratio < 2.2

    def test_persist_writes_to_db(self, seeded_db):
        """persist=True 时写入 portfolio_snapshot。"""
        with SessionContext() as s:
            before = s.execute(text("SELECT COUNT(*) FROM portfolio_snapshot")).scalar()

        runner = DBBacktestRunner()
        runner.run(top_n=3, rebalance_freq=20, persist=True)

        with SessionContext() as s:
            after = s.execute(text("SELECT COUNT(*) FROM portfolio_snapshot")).scalar()
        assert after > before

    def test_signal_source_factor_value(self, seeded_db):
        """signal_source='factor_value' 从 factor_value 表读取。"""
        runner = DBBacktestRunner()
        report = runner.run(
            start_date="20260101", end_date="20260131",
            top_n=3, rebalance_freq=10,
            signal_source="factor_value",
            persist=False,
        )
        assert report is not None
        assert "snapshots" in report

    def test_empty_data_graceful(self):
        """空数据表不崩溃，返回安全默认值。"""
        _ensure_tables()
        runner = DBBacktestRunner()
        report = runner.run(persist=False)
        # 空表时 trading_days=0，final_value=初始资金
        assert report["final_value"] == 1_000_000
        assert isinstance(report["trading_days"], int)

    def test_print_summary(self, seeded_db, capsys):
        """print_summary 不抛异常且输出中文摘要。"""
        runner = DBBacktestRunner()
        report = runner.run(top_n=3, rebalance_freq=20, persist=False)
        runner.print_summary(report)
        captured = capsys.readouterr()
        assert "回测完成" in captured.out
        assert "夏普" in captured.out

    def test_risk_events_recorded(self, seeded_db):
        """风控事件记录在报告中。"""
        runner = DBBacktestRunner()
        report = runner.run(top_n=5, rebalance_freq=10, persist=False)
        assert "risk_events_list" in report
        assert isinstance(report["risk_events_list"], list)

    def test_var_computed(self, seeded_db):
        """足够交易日（>20天）时 VaR 被计算。"""
        runner = DBBacktestRunner()
        report = runner.run(top_n=3, rebalance_freq=5, persist=False)
        assert "var_records" in report
        if report["trading_days"] > 20:
            assert len(report["var_records"]) > 0


class TestBacktestEngineBasic:
    """BacktestEngine 基础功能。"""

    def test_engine_instantiation(self):
        engine = BacktestEngine()
        assert engine is not None
        # experiment_id 在 run() 调用后才赋值，初始为空字符串
        assert isinstance(engine.experiment_id, str)
