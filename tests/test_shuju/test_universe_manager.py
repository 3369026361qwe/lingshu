"""
测试 UniverseManager — Point-in-Time 股票宇宙 + 幸存者偏差治理.
"""

import os
import tempfile
from pathlib import Path

import pytest

from shuju.universe_manager import (
    UniverseManager,
    _is_st_name,
    _is_limit_board,
    _normalize_date,
    _to_float,
    _fuzzy_eq,
)


class TestNormalizeDate:
    def test_yyyymmdd(self):
        assert _normalize_date("20240115") == "2024-01-15"

    def test_already_normalized(self):
        assert _normalize_date("2024-01-15") == "2024-01-15"

    def test_slash_format(self):
        assert _normalize_date("2024/01/15") == "2024-01-15"

    def test_empty(self):
        assert _normalize_date("") == ""

    def test_whitespace(self):
        assert _normalize_date("  20240115  ") == "2024-01-15"


class TestIsSTName:
    def test_normal_stock(self):
        assert not _is_st_name("平安银行")

    def test_st_prefix(self):
        assert _is_st_name("ST瑞德")

    def test_star_st(self):
        assert _is_st_name("*ST华英")

    def test_st_lowercase(self):
        assert _is_st_name("st天润")

    def test_st_in_middle(self):
        # ST 不在开头不算
        assert not _is_st_name("沪深ST基金")  # unrealistic but tests logic
        assert not _is_st_name("平安银行ST")  # ST suffix — not prefix

    def test_empty(self):
        assert not _is_st_name("")

    def test_none(self):
        assert not _is_st_name(None)  # type: ignore


class TestFuzzyEq:
    def test_equal(self):
        assert _fuzzy_eq(1.0, 1.0)

    def test_close(self):
        assert _fuzzy_eq(1.0, 1.0 + 1e-7)

    def test_not_equal(self):
        assert not _fuzzy_eq(1.0, 1.01)


class TestToFloat:
    def test_int(self):
        assert _to_float(42) == 42.0

    def test_float(self):
        assert _to_float(3.14) == 3.14

    def test_str(self):
        assert _to_float("3.14") == 3.14

    def test_none(self):
        assert _to_float(None) == 0.0

    def test_invalid(self):
        assert _to_float("not-a-number") == 0.0


class TestIsLimitBoard:
    def test_normal_bar(self):
        bar = {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "pre_close": 10.0}
        assert not _is_limit_board(bar)

    def test_up_limit(self):
        # 涨停一字板: O=H=L=C, ~10% up
        bar = {"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "pre_close": 10.0}
        assert _is_limit_board(bar)

    def test_down_limit(self):
        # 跌停一字板
        bar = {"open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "pre_close": 10.0}
        assert _is_limit_board(bar)

    def test_volume_zero_not_limit(self):
        # O=H=L=C 但涨跌不足 → 不是一字板 (可能是停牌)
        pre = 10.0
        c = 10.02  # 0.2% change
        bar = {"open": c, "high": c, "low": c, "close": c, "pre_close": pre}
        assert not _is_limit_board(bar)

    def test_huge_gap_not_one_line(self):
        # 高开但价格不完全相等 → 不是一字板
        bar = {"open": 11.0, "high": 11.5, "low": 11.0, "close": 11.4, "pre_close": 10.0}
        assert not _is_limit_board(bar)

    def test_zero_pre_close(self):
        bar = {"open": 0, "high": 0, "low": 0, "close": 0, "pre_close": 0}
        assert not _is_limit_board(bar)

    def test_5_percent_limit(self):
        # ST 板块 5% 涨跌停
        bar = {"open": 10.5, "high": 10.5, "low": 10.5, "close": 10.5, "pre_close": 10.0}
        assert _is_limit_board(bar)

    def test_20_percent_kcb(self):
        # 科创板 20% 涨跌停
        bar = {"open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0, "pre_close": 10.0}
        assert _is_limit_board(bar)


class TestUniverseManagerInit:
    def test_empty_init(self):
        manager = UniverseManager()
        assert manager.delisted_count >= 0
        assert manager.active_count >= 0

    def test_custom_csv_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_delist.csv"
            csv_path.write_text("code,list_date,delist_date\n000888,2020-01-01,2024-12-31\n", encoding="utf-8")
            manager = UniverseManager(delisted_csv_path=csv_path)
            assert "000888" in manager._delist_db
            assert manager._delist_db["000888"] == ("2020-01-01", "2024-12-31")


class TestRegisterDelisted:
    def test_register_single(self):
        manager = UniverseManager()
        manager.register_delisted("000888", "2020-01-01", "2024-12-31")
        assert "000888" in manager._delist_db
        assert manager._delist_db["000888"] == ("2020-01-01", "2024-12-31")

    def test_register_without_delist_date(self):
        manager = UniverseManager()
        manager.register_delisted("999999", "2020-01-01", None)
        assert manager._delist_db["999999"] == ("2020-01-01", None)

    def test_delisted_count(self):
        manager = UniverseManager()
        manager.register_delisted("001", "2020-01-01", "2023-06-30")
        manager.register_delisted("002", "2019-05-15", "2024-01-01")
        assert manager.delisted_count == 2


class TestRegisterActiveStocks:
    def test_register_names(self):
        manager = UniverseManager()
        names = {"000001": "平安银行", "000002": "万科A", "600519": "贵州茅台"}
        manager.register_active_stocks(names)
        assert manager.active_count == 3
        assert manager._active_names["000001"] == "平安银行"

    def test_register_list_dates(self):
        manager = UniverseManager()
        dates = {"000001": "1991-04-03", "600519": "2001-08-27"}
        manager.register_active_list_dates(dates)
        assert manager._active_list_dates["600519"] == "2001-08-27"


class TestSurvivorshipFreeUniverse:
    def test_active_only(self):
        """已上市活跃股票应包含在内."""
        manager = UniverseManager()
        manager.register_active_stocks({"000001": "平安银行", "000002": "万科A", "600519": "贵州茅台"})
        universe = manager.survivorship_free_universe("2024-01-15", include_delisted=False)
        assert "000001" in universe
        assert "000002" in universe
        assert "600519" in universe

    def test_with_delisted_that_was_active(self):
        """query_date 时仍在交易的退市股票应包含在内."""
        manager = UniverseManager()
        manager.register_active_stocks({"000001": "平安银行"})
        # 000888 在 query_date 时还在交易, 之后退市
        manager.register_delisted("000888", "2020-01-01", "2025-12-31")
        universe = manager.survivorship_free_universe("2024-01-15", include_delisted=True)
        assert "000001" in universe
        assert "000888" in universe  # 当时还在交易

    def test_delisted_before_query(self):
        """query_date 之前已退市的股票不应包含."""
        manager = UniverseManager()
        manager.register_active_stocks({"000001": "平安银行"})
        # 000888 在 query_date 前已经退市
        manager.register_delisted("000888", "2020-01-01", "2023-12-31")
        universe = manager.survivorship_free_universe("2024-01-15", include_delisted=True)
        assert "000888" not in universe  # 已退市

    def test_not_listed_yet(self):
        """上市日期晚于 query_date 的股票不应包含."""
        manager = UniverseManager()
        manager.register_active_stocks({"000001": "平安银行"})
        # 000888 2025年才上市
        manager.register_delisted("000888", "2025-06-01", None)
        universe = manager.survivorship_free_universe("2024-01-15", include_delisted=True)
        assert "000888" not in universe

    def test_exclude_delisted(self):
        """include_delisted=False 时不包含退市股票."""
        manager = UniverseManager()
        manager.register_active_stocks({"000001": "平安银行"})
        manager.register_delisted("000888", "2020-01-01", "2025-12-31")
        universe = manager.survivorship_free_universe("2024-01-15", include_delisted=False)
        assert "000001" in universe
        assert "000888" not in universe

    def test_sorted_output(self):
        manager = UniverseManager()
        manager.register_active_stocks({"600519": "茅台", "000001": "平安"})
        universe = manager.survivorship_free_universe("2024-01-15")
        assert universe == sorted(universe)

    def test_no_duplicates(self):
        """活跃股票和退市股票的交集不应重复."""
        manager = UniverseManager()
        manager.register_active_stocks({"000001": "平安银行"})
        manager.register_delisted("000001", "1991-04-03", None)  # same code
        universe = manager.survivorship_free_universe("2024-01-15", include_delisted=True)
        assert universe.count("000001") == 1


class TestFilterSTStar:
    def test_filter_st_by_name(self):
        manager = UniverseManager()
        universe = ["000001", "000002", "600666", "600777"]
        names = {"000001": "平安银行", "000002": "ST万科", "600666": "*ST瑞德", "600777": "万科企业"}
        result = manager.filter_st_star(universe, names)
        assert result == ["000001", "600777"]

    def test_filter_st_by_cached_names(self):
        manager = UniverseManager()
        manager.register_active_stocks({"000001": "平安银行", "000002": "ST万科"})
        result = manager.filter_st_star(["000001", "000002"])
        assert result == ["000001"]

    def test_fallback_to_code(self):
        """无名称映射时从代码检测 ST."""
        # Create a clean manager with no names
        manager = UniverseManager()
        result = manager.filter_st_star(["ST001", "000002", "*ST003"])
        # Fallback: code-based detection — checks "ST" in code.upper()
        assert "000002" in result  # only non-ST code


class TestFilterSuspended:
    def test_normal_trading(self):
        manager = UniverseManager()
        daily = {
            "000001": {"volume": 1e6, "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "pre_close": 10.0},
            "000002": {"volume": 2e6, "open": 5.0, "high": 5.2, "low": 4.9, "close": 5.1, "pre_close": 5.0},
        }
        result = manager.filter_suspended(["000001", "000002"], daily)
        assert result == ["000001", "000002"]

    def test_suspended_filtered(self):
        """停牌 (量=0, 非一字板) 应被过滤."""
        daily = {
            "000001": {"volume": 1e6, "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "pre_close": 10.0},
            "000002": {"volume": 0, "open": 0.0, "high": 0.0, "low": 0.0, "close": 5.1, "pre_close": 5.0},
        }
        result = UniverseManager.filter_suspended(["000001", "000002"], daily)
        assert result == ["000001"]  # 000002 被过滤

    def test_limit_board_kept(self):
        """一字板 (量=0 但是一字涨跌停) 保留."""
        daily = {
            "000001": {"volume": 1e6, "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "pre_close": 10.0},
            "000002": {"volume": 0, "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "pre_close": 10.0},
        }
        result = UniverseManager.filter_suspended(["000001", "000002"], daily)
        assert "000002" in result  # 涨停一字板保留

    def test_missing_data_kept(self):
        """无数据的股票保守保留."""
        daily = {"000001": {"volume": 1e6, "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "pre_close": 10.0}}
        result = UniverseManager.filter_suspended(["000001", "000002"], daily)
        assert "000002" in result  # 无数据 → 保守保留

    def test_no_daily_data(self):
        """无日线数据时不作过滤."""
        result = UniverseManager.filter_suspended(["000001", "000002"], None)
        assert result == ["000001", "000002"]


class TestSaveLoad:
    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "delisted.csv"
            manager = UniverseManager(delisted_csv_path=csv_path)
            manager.register_delisted("000888", "2020-01-01", "2024-12-31")
            manager.register_delisted("000999", "2019-06-01", None)
            manager.save()

            # Reload
            manager2 = UniverseManager(delisted_csv_path=csv_path)
            assert manager2.delisted_count == 2
            assert manager2._delist_db["000888"] == ("2020-01-01", "2024-12-31")
            assert manager2._delist_db["000999"] == ("2019-06-01", None)

    def test_load_nonexistent_file(self):
        manager = UniverseManager(delisted_csv_path="nonexistent/file.csv")
        assert manager.delisted_count == 0


class TestDelistedCodes:
    def test_sorted(self):
        manager = UniverseManager()
        manager.register_delisted("000002", "2020-01-01", "2024-12-31")
        manager.register_delisted("000001", "2019-01-01", "2023-06-30")
        assert manager.delisted_codes == ["000001", "000002"]
