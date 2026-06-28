"""
测试 Repository CRUD 仓库。
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from shujuku.models.market_models import DailyBar
from shujuku.repository import Repository
from shujuku.session import SessionContext, init_db


@pytest.fixture(autouse=True)
def setup_db():
    init_db(drop_all=True)


class TestRepositoryStock:
    def test_upsert_stock_new(self):
        with SessionContext() as s:
            repo = Repository(s)
            stock = repo.upsert_stock("000001", "平安银行", "SZ")
            s.commit()
            assert stock.code == "000001"

    def test_upsert_stock_update(self):
        with SessionContext() as s:
            repo = Repository(s)
            repo.upsert_stock("000001", "平安银行", "SZ")
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            stock = repo.upsert_stock("000001", "平安银行V2", "SZ")
            s.commit()
            assert stock.name == "平安银行V2"

    def test_get_active_stocks(self):
        with SessionContext() as s:
            repo = Repository(s)
            for code, name in [("000001", "A"), ("000002", "B"), ("000003", "C")]:
                repo.upsert_stock(code, name)
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            active = repo.get_active_stocks()
            assert len(active) == 3

    def test_get_stock_by_code(self):
        with SessionContext() as s:
            repo = Repository(s)
            repo.upsert_stock("000001", "平安银行")
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            stock = repo.get_stock_by_code("000001")
            assert stock is not None
            assert stock.name == "平安银行"


class TestRepositoryDailyBar:
    def test_get_daily_bars_range(self):
        with SessionContext() as s:
            repo = Repository(s)
            for i in range(5):
                s.add(DailyBar(
                    code="000001", trade_date=date(2026, 6, i + 1),
                    open=Decimal("10"), high=Decimal("11"), low=Decimal("9.5"),
                    close=Decimal(f"10.{i}"), volume=Decimal("1000000"), amount=Decimal("10000000"),
                ))
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            bars = repo.get_daily_bars("000001", date(2026, 6, 1), date(2026, 6, 5))
            assert len(bars) == 5

    def test_get_bars_for_date(self):
        with SessionContext() as s:
            for code in ["000001", "000002", "000003"]:
                s.add(DailyBar(
                    code=code, trade_date=date(2026, 6, 1),
                    open=Decimal("10"), high=Decimal("11"), low=Decimal("9.5"),
                    close=Decimal("10.5"), volume=Decimal("1000000"), amount=Decimal("10000000"),
                ))
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            bars = repo.get_bars_for_date(date(2026, 6, 1))
            assert len(bars) == 3


class TestRepositoryFactor:
    def test_save_and_query_factor_value(self):
        with SessionContext() as s:
            repo = Repository(s)
            repo.save_factor_value(
                code="000001", trade_date=date(2026, 6, 1),
                category="value", factor_name="pe",
                raw_value=Decimal("15.5"), z_score=Decimal("0.3"), percentile=Decimal("0.7"),
            )
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            values = repo.get_factor_values("000001", "pe", date(2026, 1, 1), date(2026, 12, 31))
            assert len(values) == 1
            assert values[0].raw_value == Decimal("15.5")
            assert values[0].z_score == Decimal("0.3")

    def test_upsert_factor_value(self):
        with SessionContext() as s:
            repo = Repository(s)
            repo.save_factor_value("000001", date(2026, 6, 1), "value", "pe", Decimal("15.5"))
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            repo.save_factor_value("000001", date(2026, 6, 1), "value", "pe", Decimal("16.0"))
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            values = repo.get_factor_values("000001", "pe", date(2026, 1, 1), date(2026, 12, 31))
            assert len(values) == 1
            assert values[0].raw_value == Decimal("16.0")


class TestRepositoryAgent:
    def test_save_and_query(self):
        with SessionContext() as s:
            repo = Repository(s)
            repo.save_agent_report(
                agent_id="macro",
                analysis_date=datetime(2026, 6, 1, 9, 15),
                target_stocks='["000001","000002"]',
                signal=Decimal("0.3"),
                confidence=Decimal("0.78"),
                reasoning="宏观环境偏积极",
                risk_flags='["美联储议息"]',
                model_used="deepseek-v4",
                tokens_used=1500,
                latency_ms=3200,
                is_cached=False,
            )
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            reports = repo.get_latest_agent_reports(agent_id="macro", limit=5)
            assert len(reports) == 1
            assert reports[0].agent_id == "macro"
            assert reports[0].confidence == Decimal("0.78")

    def test_get_latest_limited(self):
        with SessionContext() as s:
            repo = Repository(s)
            for i in range(10):
                repo.save_agent_report(
                    agent_id="sector", analysis_date=datetime(2026, 6, i + 1),
                    target_stocks="[]", signal=Decimal("0"), confidence=Decimal("0.5"),
                    reasoning=f"report {i}",
                )
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            reports = repo.get_latest_agent_reports(agent_id="sector", limit=3)
            assert len(reports) == 3


class TestRepositoryPosition:
    def test_upsert_new(self):
        with SessionContext() as s:
            repo = Repository(s)
            pos = repo.upsert_position("000001", 1000, Decimal("10.00"))
            s.commit()
            assert pos.code == "000001"
            assert pos.quantity == 1000

    def test_get_all_positions(self):
        with SessionContext() as s:
            repo = Repository(s)
            repo.upsert_position("000001", 1000, Decimal("10.00"))
            repo.upsert_position("000002", 500, Decimal("20.00"))
            # 清仓
            repo.upsert_position("000003", 0, Decimal("15.00"))
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            positions = repo.get_all_positions()
            assert len(positions) == 2  # quantity=0 的被过滤


class TestRepositoryRisk:
    def test_log_risk(self):
        from shujuku.models.fengkong_models import RiskLog
        with SessionContext() as s:
            repo = Repository(s)
            repo.log_risk("WARNING", "position", "单票仓位超限")
            s.commit()
        with SessionContext() as s:
            logs = s.query(RiskLog).all()
            assert len(logs) == 1
            assert logs[0].level == "WARNING"
            assert logs[0].category == "position"

    def test_log_circuit_breaker(self):
        with SessionContext() as s:
            repo = Repository(s)
            repo.log_circuit_breaker("CLOSED", "OPEN", "连续亏损3次")
            s.commit()


class TestGracefulDegradation:
    """验证降级模式不抛异常。"""

    def test_get_active_stocks_empty(self):
        with SessionContext() as s:
            repo = Repository(s)
            stocks = repo.get_active_stocks()
            assert stocks == []
            assert not repo.is_degraded

    def test_get_nonexistent(self):
        with SessionContext() as s:
            repo = Repository(s)
            result = repo.get_stock_by_code("999999")
            assert result is None
