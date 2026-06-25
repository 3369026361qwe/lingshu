"""
测试 ORM 模型：建表、插入、查询、约束。
"""

import json
from datetime import date, datetime
from decimal import Decimal

import pytest
import sqlalchemy.exc
from sqlalchemy import text

from shujuku.models import utcnow

from shujuku.session import init_db, SessionContext
from shujuku.models.market_models import StockInfo, DailyBar, FinancialReport, IndustryClassification
from shujuku.models.yinzi_models import FactorValue, FactorWeight, FactorICRecord
from shujuku.models.zhinengti_models import AgentReport, AgentEvidence
from shujuku.models.jiaoyi_models import Order, Trade, Position, PortfolioSnapshot
from shujuku.models.fengkong_models import CircuitBreakerEvent, RiskLog, VaRRecord
from shujuku.models.yinzi_models import FactorICRecord


@pytest.fixture(autouse=True)
def setup_db():
    """每个测试前重建数据库。"""
    init_db(drop_all=True)


class TestStockInfo:
    def test_insert_and_query(self):
        with SessionContext() as s:
            s.add(StockInfo(code="000001", name="平安银行", exchange="SZ"))
            s.commit()
        with SessionContext() as s:
            stock = s.get(StockInfo, "000001")
            assert stock is not None
            assert stock.name == "平安银行"
            assert stock.is_active is True

    def test_unique_constraint(self):
        from shujuku.session import get_session
        # 第一次插入成功
        s1 = get_session()
        s1.add(StockInfo(code="000001", name="平安银行"))
        s1.commit()
        s1.close()
        # 第二次插入同 code 应失败
        s2 = get_session()
        s2.add(StockInfo(code="000001", name="dup"))
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            s2.flush()
        s2.rollback()
        s2.close()


class TestDailyBar:
    def test_insert_and_query(self):
        with SessionContext() as s:
            bar = DailyBar(
                code="000001", trade_date=date(2026, 6, 1),
                open=Decimal("10.50"), high=Decimal("11.00"),
                low=Decimal("10.30"), close=Decimal("10.80"),
                volume=Decimal("50000000"), amount=Decimal("530000000"),
            )
            s.add(bar)
            s.commit()
        with SessionContext() as s:
            bars = s.query(DailyBar).where(DailyBar.code == "000001").all()
            assert len(bars) == 1
            assert bars[0].close == Decimal("10.80")

    def test_unique_code_date(self):
        from shujuku.session import get_session
        d = date(2026, 6, 1)
        s1 = get_session()
        s1.add(DailyBar(code="000001", trade_date=d, open=10, high=11, low=10, close=10.5, volume=100, amount=1000))
        s1.commit()
        s1.close()
        s2 = get_session()
        s2.add(DailyBar(code="000001", trade_date=d, open=9, high=10, low=9, close=9.5, volume=200, amount=2000))
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            s2.flush()
        s2.rollback()
        s2.close()


class TestFinancialReport:
    def test_insert(self):
        with SessionContext() as s:
            s.add(FinancialReport(
                code="000001", report_date=date(2026, 3, 31), report_type="Q1",
                pe=Decimal("8.5"), roe=Decimal("12.3"),
            ))
            s.commit()
        with SessionContext() as s:
            reports = s.query(FinancialReport).all()
            assert len(reports) == 1
            assert reports[0].roe == Decimal("12.3")


class TestFactorValue:
    def test_insert_batch(self):
        with SessionContext() as s:
            for i in range(10):
                s.add(FactorValue(
                    code=f"00000{i}", trade_date=date(2026, 6, 1),
                    category="value", factor_name="pe",
                    raw_value=Decimal(f"{10 + i}.5"),
                ))
            s.commit()
        with SessionContext() as s:
            count = s.query(FactorValue).count()
            assert count == 10

    def test_zscore_nullable(self):
        with SessionContext() as s:
            fv = FactorValue(
                code="000001", trade_date=date(2026, 6, 1),
                category="value", factor_name="pe", raw_value=Decimal("15.0"),
            )
            s.add(fv)
            s.commit()
        with SessionContext() as s:
            fv2 = s.query(FactorValue).first()
            assert fv2.z_score is None
            assert fv2.percentile is None


class TestFactorWeight:
    def test_insert_and_query(self):
        with SessionContext() as s:
            s.add(FactorWeight(
                trade_date=date(2026, 6, 1), factor_name="pe",
                weight=Decimal("0.15"), variance=Decimal("0.002"),
            ))
            s.commit()
        with SessionContext() as s:
            fw = s.query(FactorWeight).first()
            assert fw.weight == Decimal("0.15")
            assert fw.variance == Decimal("0.002")


class TestAgentReport:
    def test_insert_with_evidence(self):
        with SessionContext() as s:
            report = AgentReport(
                agent_id="macro",
                analysis_date=datetime(2026, 6, 1, 9, 15),
                target_stocks=json.dumps(["000001", "000002"]),
                signal=Decimal("0.3"),
                confidence=Decimal("0.78"),
                reasoning="PMI连续3月扩张，建议超配制造业",
                risk_flags=json.dumps(["美联储6月议息会议"]),
            )
            ev1 = AgentEvidence(source="国家统计局", metric="PMI", value="51.2")
            ev2 = AgentEvidence(source="中国债券网", metric="10Y国债", value="2.85%")
            report.evidence.append(ev1)
            report.evidence.append(ev2)
            s.add(report)
            s.commit()

        with SessionContext() as s:
            reports = s.query(AgentReport).all()
            assert len(reports) == 1
            r = reports[0]
            assert r.agent_id == "macro"
            assert len(r.evidence) == 2
            assert r.evidence[0].metric == "PMI"

    def test_cascade_delete_evidence(self):
        with SessionContext() as s:
            report = AgentReport(
                agent_id="sector", analysis_date=datetime(2026, 6, 1),
                target_stocks="[]", signal=Decimal("0"), confidence=Decimal("0.5"),
                reasoning="test",
            )
            report.evidence.append(AgentEvidence(source="test", metric="m", value="v"))
            s.add(report)
            s.commit()
            rid = report.id
        with SessionContext() as s:
            r = s.get(AgentReport, rid)
            s.delete(r)
            s.commit()
        with SessionContext() as s:
            ev = s.query(AgentEvidence).all()
            assert len(ev) == 0


class TestOrder:
    def test_lifecycle(self):
        with SessionContext() as s:
            o = Order(
                order_id="ord-001", code="000001", direction="BUY",
                quantity=1000, price=Decimal("10.50"),
            )
            s.add(o)
            s.commit()
        with SessionContext() as s:
            o2 = s.query(Order).where(Order.order_id == "ord-001").first()
            assert o2.status == "PENDING"
            o2.status = "FILLED"
            o2.filled_qty = 1000
            o2.filled_avg_price = Decimal("10.48")
            s.commit()
        with SessionContext() as s:
            o3 = s.query(Order).where(Order.order_id == "ord-001").first()
            assert o3.status == "FILLED"
            assert o3.filled_avg_price == Decimal("10.48")


class TestPosition:
    def test_upsert_via_repo(self):
        from shujuku.repository import Repository
        with SessionContext() as s:
            repo = Repository(s)
            repo.upsert_position("000001", 5000, Decimal("10.00"), Decimal("10.50"))
            s.commit()
        with SessionContext() as s:
            repo = Repository(s)
            pos = repo.get_position("000001")
            assert pos is not None
            assert pos.quantity == 5000
            assert pos.market_value == Decimal("52500")

    def test_unique_code(self):
        from shujuku.session import get_session
        s1 = get_session()
        s1.add(Position(code="000001", quantity=100, avg_cost=Decimal("10")))
        s1.commit()
        s1.close()
        s2 = get_session()
        s2.add(Position(code="000001", quantity=200, avg_cost=Decimal("11")))
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            s2.flush()
        s2.rollback()
        s2.close()


class TestCircuitBreaker:
    def test_log_event(self):
        with SessionContext() as s:
            s.add(CircuitBreakerEvent(
                timestamp=datetime(2026, 6, 1, 10, 0),
                from_state="CLOSED", to_state="OPEN",
                trigger_reason="连续亏损3次",
            ))
            s.commit()
        with SessionContext() as s:
            events = s.query(CircuitBreakerEvent).all()
            assert len(events) == 1
            assert events[0].from_state == "CLOSED"
            assert events[0].to_state == "OPEN"


class TestRiskLog:
    def test_crud(self):
        with SessionContext() as s:
            log = RiskLog(
                timestamp=utcnow(), level="WARNING",
                category="position", message="单票仓位超限: 000001 12%",
            )
            s.add(log)
            s.commit()
        with SessionContext() as s:
            logs = s.query(RiskLog).all()
            assert len(logs) == 1
            assert logs[0].level == "WARNING"


class TestIndustryClassification:
    def test_insert(self):
        with SessionContext() as s:
            s.add(IndustryClassification(code="000001", sw_level1="银行", sw_level2="股份制银行", effective_date=date(2026, 1, 1)))
            s.commit()
        with SessionContext() as s:
            rows = s.query(IndustryClassification).all()
            assert len(rows) == 1
            assert rows[0].sw_level1 == "银行"


class TestVaRRecord:
    def test_insert(self):
        with SessionContext() as s:
            s.add(VaRRecord(calc_date=datetime.utcnow(), confidence_level=Decimal("0.95"), var=Decimal("50000"), cvar=Decimal("75000")))
            s.commit()
        with SessionContext() as s:
            rows = s.query(VaRRecord).all()
            assert len(rows) == 1


class TestPortfolioSnapshot:
    def test_insert(self):
        with SessionContext() as s:
            s.add(PortfolioSnapshot(trade_date=date(2026, 6, 1), total_value=Decimal("1000000"), cash=Decimal("200000"), market_value=Decimal("800000")))
            s.commit()
        with SessionContext() as s:
            rows = s.query(PortfolioSnapshot).all()
            assert len(rows) == 1


class TestTrade:
    def test_insert(self):
        with SessionContext() as s:
            s.add(Trade(trade_id="t-001", order_id="ord-001", code="000001", direction="BUY", quantity=1000, price=Decimal("10.50"), amount=Decimal("10500"), trade_time=datetime.utcnow()))
            s.commit()
        with SessionContext() as s:
            rows = s.query(Trade).all()
            assert len(rows) == 1


class TestFactorICRecord:
    def test_insert(self):
        with SessionContext() as s:
            s.add(FactorICRecord(trade_date=date(2026, 6, 1), factor_name="pe", ic=Decimal("0.045"), ir=Decimal("0.8")))
            s.commit()
        with SessionContext() as s:
            rows = s.query(FactorICRecord).all()
            assert len(rows) == 1


class TestAllTablesExist:
    """验证所有表都已创建。"""

    def test_table_count(self):
        with SessionContext() as s:
            result = s.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_alembic%'"
            ))
            tables = set(row[0] for row in result)
            expected = {
                "stock_info", "daily_bar", "financial_report", "industry_classification",
                "factor_value", "factor_weight", "factor_ic_record",
                "agent_report", "agent_evidence",
                "orders", "trades", "positions", "portfolio_snapshot",
                "circuit_breaker_events", "risk_logs", "var_records",
            }
            missing = expected - tables
            assert not missing, f"Missing tables: {missing}"
