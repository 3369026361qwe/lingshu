"""
交叉验证: shuju ↔ shujuku 集成测试。

验证数据层产出的数据可以正确写入数据库层。
"""

from datetime import date
from decimal import Decimal

import pytest

from shujuku.session import init_db, SessionContext
from shujuku.repository import Repository


@pytest.fixture(autouse=True)
def setup_db():
    init_db(drop_all=True)


class TestShujuShujukuIntegration:
    """验证数据从获取到持久化的完整链路。"""

    def test_write_daily_bar_to_db(self):
        """模拟的数据经过 Repository 写入并读取。"""
        from shujuku.models.market_models import DailyBar

        bar_data = {
            "code": "000001",
            "trade_date": "20260601",
            "open": Decimal("10.50"),
            "high": Decimal("11.00"),
            "low": Decimal("10.30"),
            "close": Decimal("10.80"),
            "volume": Decimal("50000000"),
            "amount": Decimal("530000000"),
            "turnover_rate": Decimal("1.5"),
        }

        with SessionContext() as s:
            bar = DailyBar(
                code=bar_data["code"],
                trade_date=date(2026, 6, 1),
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
                amount=bar_data["amount"],
                turnover_rate=bar_data["turnover_rate"],
            )
            s.add(bar)
            s.commit()

        with SessionContext() as s:
            repo = Repository(s)
            bars = repo.get_daily_bars("000001", date(2026, 6, 1), date(2026, 6, 2))
            assert len(bars) == 1
            assert bars[0].close == Decimal("10.80")

    def test_write_stock_info_chain(self):
        """AKShare 获取的股票列表写入 shujuku。"""
        from shujuku.models.market_models import StockInfo

        stocks = [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
            {"code": "600519", "name": "贵州茅台"},
        ]

        with SessionContext() as s:
            for st in stocks:
                existing = s.get(StockInfo, st["code"])
                if not existing:
                    s.add(StockInfo(code=st["code"], name=st["name"]))
            s.commit()

        with SessionContext() as s:
            repo = Repository(s)
            active = repo.get_active_stocks()
            assert len(active) == 3

    def test_factor_value_roundtrip(self):
        """预处理后的因子值通过 Repository 写入并回读。"""
        preprocessed_factors = [
            ("000001", "value", "pe", Decimal("0.35"), Decimal("0.60")),
            ("000001", "quality", "roe", Decimal("0.82"), Decimal("0.91")),
            ("000002", "value", "pe", Decimal("-0.50"), Decimal("0.25")),
        ]

        with SessionContext() as s:
            repo = Repository(s)
            for code, cat, name, raw, zscore in preprocessed_factors:
                repo.save_factor_value(
                    code=code,
                    trade_date=date(2026, 6, 1),
                    category=cat,
                    factor_name=name,
                    raw_value=raw,
                    z_score=zscore,
                )
            s.commit()

        with SessionContext() as s:
            repo = Repository(s)
            values = repo.get_factor_values("000001", "pe", date(2026, 1, 1), date(2026, 12, 31))
            assert len(values) == 1
            assert values[0].raw_value == Decimal("0.35")
            assert values[0].z_score == Decimal("0.60")

    def test_cache_compatibility(self):
        """shuju.DataCacheManager 使用 shujuku.CacheManager 底层。"""
        from shuju.cache_manager import DataCacheManager

        cache = DataCacheManager()
        bar = {"code": "000001", "trade_date": "20260601", "close": "10.80"}
        cache.cache_daily_bar("000001", "20260601", bar)

        cached = cache.get_daily_bar("000001", "20260601")
        assert cached is not None
        assert cached["close"] == "10.80"

        cache.clear_all()
        assert cache.get_daily_bar("000001", "20260601") is None
