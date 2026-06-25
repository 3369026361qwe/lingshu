"""
AKShare 行情数据获取器。

功能:
    - 全市场 A 股列表 (含行业分类)
    - 单只/批量日线行情 (OHLCV)
    - 申万行业分类

特性:
    - 自动重试 (3次, 指数退避)
    - 请求频率限制 (避免被封 IP)
    - 优雅降级 (网络异常返回空数据不崩溃)

Usage:
    fetcher = AKShareFetcher()
    stocks = fetcher.get_stock_list()
    bars = fetcher.get_daily_bars("000001", start="20260101", end="20260601")
"""

import logging
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

import pandas as pd

from shuju.cache_manager import DataCacheManager
from shuju.utils import safe_decimal, make_retry

_logger = logging.getLogger(__name__)

# 请求间隔 (秒)
_REQUEST_INTERVAL = 0.5

_retry = make_retry("akshare", max_retries=3, logger=_logger)


class AKShareFetcher:
    """AKShare 行情数据获取器。"""

    def __init__(self, cache: Optional[DataCacheManager] = None) -> None:
        self._cache = cache or DataCacheManager()
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """请求频率限制。"""
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_INTERVAL:
            time.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request = time.time()

    # ── 股票列表 ────────────────────────────────────────

    @_retry
    def _fetch_stock_list_raw(self) -> pd.DataFrame:
        """从 AKShare 拉取全 A 股列表原始数据。"""
        import akshare as ak
        self._rate_limit()
        df = ak.stock_info_a_code_name()
        return df

    def get_stock_list(self) -> list[dict]:
        """获取全 A 股列表。

        Returns:
            [{"code": "000001", "name": "平安银行"}, ...]
        """
        df = self._fetch_stock_list_raw()
        if df is None or df.empty:
            return []

        result = []
        for _, row in df.iterrows():
            try:
                code = str(row.get("code", "")).zfill(6)
                name = str(row.get("name", ""))
                if code and name:
                    result.append({"code": code, "name": name})
            except Exception:
                continue
        return result

    # ── 日线行情 ────────────────────────────────────────

    @_retry
    def _fetch_daily_raw(self, code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        """从 AKShare 拉取单只股票日线原始数据。"""
        import akshare as ak
        self._rate_limit()
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
        return df

    def get_daily_bars(
        self,
        code: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """获取单只股票日线数据。

        Args:
            code: 6 位股票代码
            start: 起始日期 "YYYYMMDD"，默认 1 年前
            end: 结束日期 "YYYYMMDD"，默认今天
            use_cache: 是否使用缓存

        Returns:
            [{"trade_date": "2026-06-01", "open": 10.50, "high": 11.00,
              "low": 10.30, "close": 10.80, "volume": 50000000, ...}, ...]
        """
        if end is None:
            end = date.today().strftime("%Y%m%d")
        if start is None:
            start = date.today().replace(year=date.today().year - 1).strftime("%Y%m%d")

        # 尝试缓存
        if use_cache:
            cached = self._cache.get_daily_bars_batch(code)
            if cached:
                # 按日期范围过滤缓存数据
                return [b for b in cached if start <= b.get("trade_date", "") <= end]

        df = self._fetch_daily_raw(code, start, end)
        if df is None or df.empty:
            return []

        result = []
        for _, row in df.iterrows():
            try:
                # 标准化日期格式为 YYYYMMDD（AKShare 返回 YYYY-MM-DD）
                raw_date = str(row.get("日期", "")).replace("-", "")
                bar = {
                    "code": code,
                    "trade_date": raw_date,
                    "open": safe_decimal(row.get("开盘")),
                    "high": safe_decimal(row.get("最高")),
                    "low": safe_decimal(row.get("最低")),
                    "close": safe_decimal(row.get("收盘")),
                    "volume": safe_decimal(row.get("成交量")),
                    "amount": safe_decimal(row.get("成交额")),
                    "turnover_rate": safe_decimal(row.get("换手率")),
                }
                result.append(bar)
            except Exception as exc:
                _logger.debug("Skip row for %s: %s", code, exc)
                continue

        # 缓存
        if use_cache and result:
            self._cache.cache_daily_bars_batch(code, result)

        return result

    def get_market_snapshot(self, trade_date: Optional[str] = None) -> list[dict]:
        """获取全市场某日行情快照。

        优先使用 AKShare 全市场接口（东方财富实时行情），失败时 fallback 到分批并发拉取。

        Args:
            trade_date: 日期字符串，默认最近交易日

        Returns:
            全市场日线数据列表
        """
        if trade_date is None:
            trade_date = date.today().strftime("%Y%m%d")

        # 尝试缓存
        cached = self._cache.get_preprocessed(f"market_snapshot:{trade_date}")
        if cached:
            return cached

        # 方案 A：全市场接口
        try:
            results = self._fetch_spot_market(trade_date)
            if results:
                self._cache.cache_preprocessed(f"market_snapshot:{trade_date}", results)
                return results
        except Exception as exc:
            _logger.warning("Full market API failed, falling back to batch: %s", exc)

        # 方案 B：分批并发拉取
        return self._fallback_batch_snapshot(trade_date)

    @_retry
    def _fetch_spot_market(self, trade_date: str) -> list[dict]:
        """方案 A: AKShare 全市场实时行情接口。"""
        import akshare as ak
        self._rate_limit()
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            try:
                code = str(row.get("代码", "")).zfill(6)
                if not code or code.startswith("900"):  # 跳过 B 股
                    continue
                results.append({
                    "code": code,
                    "trade_date": trade_date,
                    "open": safe_decimal(row.get("今开")),
                    "high": safe_decimal(row.get("最高")),
                    "low": safe_decimal(row.get("最低")),
                    "close": safe_decimal(row.get("最新价")),
                    "volume": safe_decimal(row.get("成交量")),
                    "amount": safe_decimal(row.get("成交额")),
                    "turnover_rate": safe_decimal(row.get("换手率")),
                })
            except Exception:
                continue
        return results

    def _fallback_batch_snapshot(self, trade_date: str, max_workers: int = 10) -> list[dict]:
        """方案 B: 分批并发拉取全市场快照。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        stocks = self.get_stock_list()
        if not stocks:
            return []

        results = []
        codes = [s["code"] for s in stocks]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_daily_bars, code, trade_date, trade_date): code
                for code in codes
            }
            for future in as_completed(futures):
                try:
                    bars = future.result()
                    if bars:
                        results.extend(bars)
                except Exception:
                    continue

        return results

    # ── 行业分类 ────────────────────────────────────────

    @_retry
    def _fetch_industry_raw(self) -> pd.DataFrame:
        """从 AKShare 拉取申万行业分类原始数据。"""
        import akshare as ak
        self._rate_limit()
        df = ak.stock_info_a_industry()
        return df

    def get_industry_map(self) -> dict[str, dict]:
        """获取股票代码 → 行业分类映射。

        Returns:
            {"000001": {"sw_level1": "银行", "sw_level2": "股份制银行"}, ...}
        """
        cached = self._cache.get_industry("__all__")
        if cached:
            return cached

        df = self._fetch_industry_raw()
        if df is None or df.empty:
            return {}

        result = {}
        for _, row in df.iterrows():
            try:
                code = str(row.get("code", "")).zfill(6)
                if code:
                    result[code] = {
                        "sw_level1": str(row.get("industry", "")),
                        "sw_level2": str(row.get("industry_detail", "")),
                    }
            except Exception:
                continue

        if result:
            self._cache.cache_industry("__all__", result)

        return result

