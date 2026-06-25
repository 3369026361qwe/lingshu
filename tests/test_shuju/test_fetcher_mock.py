"""
Mock 网络层的 Fetcher 单元测试。

不依赖真实 AKShare/Tushare API，使用 monkeypatch 模拟网络响应。
"""

from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest

from shuju.akshare_fetcher import AKShareFetcher


def _make_bar_df(dates=None):
    """构造模拟的 AKShare 返回 DataFrame，所有数组长度与 dates 一致。"""
    if dates is None:
        dates = ["2026-06-01", "2026-06-02"]
    n = len(dates)
    data = {
        "日期": dates,
        "开盘": [10.50 + i * 0.3 for i in range(n)],
        "最高": [11.00 + i * 0.2 for i in range(n)],
        "最低": [10.30 + i * 0.3 for i in range(n)],
        "收盘": [10.80 + i * 0.2 for i in range(n)],
        "成交量": [50000000 - i * 5000000 for i in range(n)],
        "成交额": [530000000 - i * 50000000 for i in range(n)],
        "换手率": [1.5 - i * 0.2 for i in range(n)],
    }
    return pd.DataFrame(data)


class TestAKShareFetcherMocked:
    """使用 mock 数据测试 AKShareFetcher 逻辑。"""

    def test_get_daily_bars_parses_correctly(self):
        with patch("akshare.stock_zh_a_hist", return_value=_make_bar_df()):
            fetcher = AKShareFetcher()
            bars = fetcher.get_daily_bars("000001", use_cache=False)
            assert len(bars) == 2
            assert bars[0]["close"] == Decimal("10.80")
            assert bars[0]["code"] == "000001"
            assert bars[0]["trade_date"] == "20260601"  # 日期格式标准化

    def test_get_daily_bars_empty_response(self):
        with patch("akshare.stock_zh_a_hist", return_value=pd.DataFrame()):
            fetcher = AKShareFetcher()
            bars = fetcher.get_daily_bars("000001", use_cache=False)
            assert bars == []

    def test_get_daily_bars_network_error(self):
        with patch("akshare.stock_zh_a_hist", side_effect=ConnectionError("timeout")):
            fetcher = AKShareFetcher()
            bars = fetcher.get_daily_bars("000001", use_cache=False)
            assert bars == []  # 优雅降级，不抛异常

    def test_get_stock_list_parses_correctly(self):
        mock_df = pd.DataFrame({
            "code": ["000001", "000002"],
            "name": ["平安银行", "万科A"],
        })
        with patch("akshare.stock_info_a_code_name", return_value=mock_df):
            fetcher = AKShareFetcher()
            stocks = fetcher.get_stock_list()
            assert len(stocks) == 2
            assert stocks[0]["code"] == "000001"

    def test_retry_on_failure_then_success(self):
        """前两次失败，第三次成功。"""
        call_count = [0]

        def flaky_akshare(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("timeout")
            return _make_bar_df(["2026-06-01"])

        with patch("akshare.stock_zh_a_hist", side_effect=flaky_akshare):
            fetcher = AKShareFetcher()
            bars = fetcher.get_daily_bars("000001", use_cache=False)
            assert len(bars) == 1
            assert call_count[0] == 3

    def test_cache_hit_skips_api_call(self):
        """缓存命中时不应调用 AKShare API。"""
        fetcher = AKShareFetcher()

        with patch("akshare.stock_zh_a_hist", return_value=_make_bar_df(["2026-06-01"])) as mock_api:
            # 第一次调用 — 触发 API
            bars1 = fetcher.get_daily_bars("000001", use_cache=True)
            assert len(bars1) == 1
            assert mock_api.call_count == 1

            # 第二次调用 — 应走缓存（同一股票+日期范围）
            bars2 = fetcher.get_daily_bars("000001", use_cache=True)
            assert len(bars2) == 1
            # API 不应被再次调用（缓存命中）
            assert mock_api.call_count == 1

    def test_all_retries_exhausted_returns_none(self):
        """全部重试失败返回空列表。"""
        with patch("akshare.stock_zh_a_hist", side_effect=ConnectionError("always fail")):
            fetcher = AKShareFetcher()
            bars = fetcher.get_daily_bars("000001", use_cache=False)
            assert bars == []

    def test_to_decimal_handles_invalid_strings(self):
        """safe_decimal 对非数值字符串返回 None。"""
        from shuju.utils import safe_decimal
        assert safe_decimal("-") is None
        assert safe_decimal("—") is None
        assert safe_decimal(None) is None
        assert safe_decimal("10.5") == Decimal("10.5")
