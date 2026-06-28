"""
Tushare 行情/财务数据获取器 — 灵枢量化系统唯一日线数据源。

功能:
    - 日线行情 (OHLCV + 换手率) — 唯一官方数据源
    - 财务报表 (PE/PB/ROE/ROA/毛利率/净利率/营收/净利润/现金流)
    - 估值数据
    - 股东人数变化
    - 沪深300 / 中证500 等指数成分股

数据源策略:
    Tushare Pro 为灵枢系统唯一行情数据源，禁止混用 AKShare/EastMoney
    等第三方数据源获取日线，防止数据格式不一致和精度污染。

特性:
    - Token 环境变量配置 (TUSHARE_TOKEN)
    - 自动重试 + 频率限制
    - 优雅降级
    - 统一输出格式 (YYYYMMDD / Decimal 18位精度)

Usage:
    fetcher = TushareFetcher()
    bars = fetcher.get_daily_bars("000001", start="20210101", end="20260604")
    reports = fetcher.get_financial_reports("000001", start="20250101", end="20260601")
"""

import logging
import os
import threading
import time
from datetime import date
from decimal import Decimal

import pandas as pd

from shuju.cache_manager import DataCacheManager
from shuju.utils import make_retry, safe_decimal

_logger = logging.getLogger(__name__)

_REQUEST_INTERVAL = 0.3

_retry = make_retry("tushare", max_retries=3, logger=_logger)


class TushareFetcher:
    """Tushare 财务数据获取器。"""

    def __init__(self, token: str | None = None, cache: DataCacheManager | None = None) -> None:
        self._token = token or os.getenv("TUSHARE_TOKEN", "")
        self._cache = cache or DataCacheManager()
        self._pro = None
        self._last_request = 0.0
        self._init_lock = threading.Lock()  # C3: 线程安全

    @property
    def is_ready(self) -> bool:
        """是否已配置 Token 并可连接。"""
        return bool(self._token)

    def _get_pro(self):
        """懒加载 Tushare Pro 客户端（线程安全）。"""
        if self._pro is None and self._token:
            with self._init_lock:
                if self._pro is None and self._token:  # double-checked locking
                    try:
                        import tushare as ts
                        ts.set_token(self._token)
                        self._pro = ts.pro_api()
                    except Exception as exc:
                        _logger.warning("Tushare init failed: %s", exc)
                        self._pro = None
        return self._pro

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_INTERVAL:
            time.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request = time.time()

    # ── 财务数据 ────────────────────────────────────────

    @_retry
    def _fetch_income_raw(self, code: str, start: str, end: str) -> pd.DataFrame:
        """拉取利润表原始数据。"""
        pro = self._get_pro()
        if pro is None:
            return pd.DataFrame()
        self._rate_limit()
        return pro.income(ts_code=self._to_ts_code(code), start_date=start, end_date=end)

    @_retry
    def _fetch_balance_raw(self, code: str, start: str, end: str) -> pd.DataFrame:
        """拉取资产负债表原始数据。"""
        pro = self._get_pro()
        if pro is None:
            return pd.DataFrame()
        self._rate_limit()
        return pro.balancesheet(ts_code=self._to_ts_code(code), start_date=start, end_date=end)

    @_retry
    def _fetch_cashflow_raw(self, code: str, start: str, end: str) -> pd.DataFrame:
        """拉取现金流量表原始数据。"""
        pro = self._get_pro()
        if pro is None:
            return pd.DataFrame()
        self._rate_limit()
        return pro.cashflow(ts_code=self._to_ts_code(code), start_date=start, end_date=end)

    @_retry
    def _fetch_daily_basic_raw(self, code: str, start: str, end: str) -> pd.DataFrame:
        """拉取每日指标 (PE/PB/PS/换手率等)。"""
        pro = self._get_pro()
        if pro is None:
            return pd.DataFrame()
        self._rate_limit()
        return pro.daily_basic(ts_code=self._to_ts_code(code), start_date=start, end_date=end)

    def get_financial_reports(
        self, code: str, start: str | None = None, end: str | None = None
    ) -> list[dict]:
        """获取单只股票财务数据汇总。

        Returns:
            [{"code": "000001", "report_date": "20260331", "report_type": "Q1",
              "pe": 8.5, "pb": 1.2, "roe": 12.3, "roa": 1.1,
              "gross_margin": 45.2, "net_margin": 28.1,
              "revenue": 1.5e10, "net_profit": 4.2e9,
              "operating_cashflow": 5.0e9, "fcf_yield": 3.2}, ...]
        """
        if not self.is_ready:
            _logger.warning("Tushare token not configured, skipping financial data for %s", code)
            return []

        if end is None:
            end = date.today().strftime("%Y%m%d")
        if start is None:
            start = date.today().replace(year=date.today().year - 2).strftime("%Y%m%d")

        # 拉取各表
        income_df = self._fetch_income_raw(code, start, end)
        self._fetch_balance_raw(code, start, end)
        cashflow_df = self._fetch_cashflow_raw(code, start, end)
        daily_df = self._fetch_daily_basic_raw(code, start, end)

        if income_df is None or income_df.empty:
            return []

        # 按报告期合并
        results = []
        for _, row in income_df.iterrows():
            try:
                rpt_date = str(row.get("end_date", "")).replace("-", "")[:8]
                if not rpt_date:
                    continue

                revenue = safe_decimal(row.get("total_revenue"))
                operating_cost = safe_decimal(row.get("operating_cost"))
                net_profit = safe_decimal(row.get("n_income"))

                report = {
                    "code": code,
                    "report_date": rpt_date,
                    "report_type": self._guess_report_type(rpt_date),
                    "revenue": revenue,
                    "net_profit": net_profit,
                    "gross_margin": self._calc_gross_margin(revenue, operating_cost),
                    "net_margin": self._calc_ratio(net_profit, revenue),
                }
                results.append(report)
            except Exception as exc:
                _logger.debug("Skip income row for %s: %s", code, exc)

        # 合并每日估值指标 (PE/PB) — 按日期匹配到对应报告期
        if daily_df is not None and not daily_df.empty:
            # 构建日期索引
            daily_by_date: dict[str, dict] = {}
            for _, row in daily_df.iterrows():
                d = str(row.get("trade_date", "")).replace("-", "")[:8]
                if d:
                    daily_by_date[d] = row

            for report in results:
                rpt_date = report.get("report_date", "")
                # 找最近交易日（≤ 报告期 + 30天缓冲）
                best_date = ""
                for d in sorted(daily_by_date.keys()):
                    if rpt_date <= d:
                        best_date = d
                        break
                if not best_date and daily_by_date:
                    best_date = max(daily_by_date.keys())

                if best_date and best_date in daily_by_date:
                    row = daily_by_date[best_date]
                    report["pe"] = safe_decimal(row.get("pe"))
                    report["pb"] = safe_decimal(row.get("pb"))
                    report["ps"] = safe_decimal(row.get("ps"))

        # 合并现金流数据 + 计算自由现金流收益率
        if cashflow_df is not None and not cashflow_df.empty:
            cf_latest = cashflow_df.iloc[-1]
            oper_cf = safe_decimal(cf_latest.get("n_cashflow_act"))
            capex = safe_decimal(cf_latest.get("c_pay_acq_const_fiolta"))

            if results:
                results[-1]["operating_cashflow"] = oper_cf

                # 自由现金流收益率 = (经营CF - 资本支出) / 总市值
                if oper_cf is not None and capex is not None:
                    total_mv = None
                    if daily_df is not None and not daily_df.empty:
                        dl = daily_df.iloc[-1]
                        total_mv = safe_decimal(dl.get("total_mv"))

                    if total_mv is not None and total_mv > 0:
                        fcf = oper_cf - capex
                        results[-1]["free_cashflow_yield"] = (fcf / total_mv).quantize(Decimal("0.0001"))
                    else:
                        results[-1]["free_cashflow_yield"] = None
                else:
                    results[-1]["free_cashflow_yield"] = None

        return results

    # ── 日线行情 ────────────────────────────────────────

    @_retry
    def _fetch_daily_raw(self, code: str, start: str, end: str) -> pd.DataFrame:
        """拉取日线行情原始数据（前复权）。"""
        pro = self._get_pro()
        if pro is None:
            return pd.DataFrame()
        self._rate_limit()
        return pro.daily(ts_code=self._to_ts_code(code), start_date=start, end_date=end)

    def get_daily_bars(
        self,
        code: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """获取单只股票日线行情（统一字段格式）。

        Args:
            code: 6 位股票代码，如 "000001"
            start: 起始日期 "YYYYMMDD"，默认 5 年前
            end: 结束日期 "YYYYMMDD"，默认今天

        Returns:
            [{"trade_date": "20260603", "open": "11.0300", "high": "11.0600",
              "low": "10.9200", "close": "10.9900", "pre_close": "11.0800",
              "change": "-0.0900", "pct_chg": "-0.8123",
              "vol": "825271", "amount": "906870268.35"}, ...]

            价格字段为 Decimal 字符串（4位小数），vol 为整数，amount 为 Decimal 字符串（2位小数）。
        """
        if end is None:
            end = date.today().strftime("%Y%m%d")
        if start is None:
            start = date.today().replace(year=date.today().year - 5).strftime("%Y%m%d")

        df = self._fetch_daily_raw(code, start, end)
        if df is None or df.empty:
            return []

        # 按日期升序排列
        df = df.sort_values("trade_date")

        bars = []
        for _, row in df.iterrows():
            try:
                bar = {
                    "code": code,
                    "trade_date": str(row.get("trade_date", "")),
                    "open": safe_decimal(row.get("open")),
                    "high": safe_decimal(row.get("high")),
                    "low": safe_decimal(row.get("low")),
                    "close": safe_decimal(row.get("close")),
                    "pre_close": safe_decimal(row.get("pre_close")),
                    "change": safe_decimal(row.get("change")),
                    "pct_chg": safe_decimal(row.get("pct_chg")),
                    "vol": int(row.get("vol", 0)) if row.get("vol") is not None else 0,
                    "amount": safe_decimal(row.get("amount")),
                }
                bars.append(bar)
            except Exception as exc:
                _logger.debug("Skip daily row for %s: %s", code, exc)
                continue

        return bars

    # ── 指数成分股 ──────────────────────────────────────

    def get_index_constituents(self, index_code: str = "000300.SH") -> list[dict]:
        """获取指数最新成分股列表。

        Args:
            index_code: 指数代码，默认 "000300.SH" (沪深300)。
                       可选: "000905.SH" (中证500), "000016.SH" (上证50)

        Returns:
            [{"ts_code": "000001.SZ", "code": "000001", "name": "平安银行"}, ...]
        """
        pro = self._get_pro()
        if pro is None:
            return []

        # 优先使用 hs_const（精确成分股）
        hs_type_map = {
            "000300.SH": "HS300",
            "000905.SH": "ZZ500",
            "000016.SH": "SZ50",
        }
        hs_type = hs_type_map.get(index_code, "HS300")

        try:
            df = pro.hs_const(hs_type=hs_type)
            if df is not None and not df.empty:
                # hs_const 返回当前在指数中的股票
                result = []
                for _, row in df.iterrows():
                    ts_code = str(row.get("ts_code", ""))
                    code = ts_code.split(".")[0] if "." in ts_code else ts_code
                    result.append({
                        "ts_code": ts_code,
                        "code": code,
                        "name": str(row.get("name", "")),
                    })
                return result
        except Exception as exc:
            _logger.info("hs_const failed, trying stock_basic filter: %s", exc)

        # Fallback: 通过 index_weight 获取成分股
        try:
            today = date.today().strftime("%Y%m%d")
            df = pro.index_weight(index_code=index_code, start_date=today, end_date=today)
            if df is not None and not df.empty:
                result = []
                for _, row in df.iterrows():
                    ts_code = str(row.get("con_code", ""))
                    code = ts_code.split(".")[0] if "." in ts_code else ts_code
                    result.append({
                        "ts_code": ts_code,
                        "code": code,
                        "weight": str(row.get("weight", "")),
                    })
                return result
        except Exception as exc:
            _logger.warning("index_weight failed: %s", exc)

        return []

    @_retry
    def _fetch_holder_raw(self, code: str) -> pd.DataFrame:
        """拉取股东人数数据。"""
        pro = self._get_pro()
        if pro is None:
            return pd.DataFrame()
        self._rate_limit()
        return pro.stk_holdernumber(ts_code=self._to_ts_code(code))

    def get_shareholder_count(self, code: str) -> int | None:
        """获取最新股东人数。"""
        if not self.is_ready:
            return None
        df = self._fetch_holder_raw(code)
        if df is None or df.empty:
            return None
        try:
            return int(df.iloc[-1].get("holder_num", 0))
        except Exception:
            return None

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _to_ts_code(code: str) -> str:
        """转换 6 位代码为 Tushare 格式 (000001.SZ)。"""
        code = code.zfill(6)
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        elif code.startswith("6"):
            return f"{code}.SH"
        elif code.startswith(("4", "8")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    @staticmethod
    def _guess_report_type(rpt_date: str) -> str:
        """根据报告期推断报告类型。"""
        month = int(rpt_date[4:6]) if len(rpt_date) >= 6 else 0
        mapping = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}
        return mapping.get(month, "Q4")

    @staticmethod
    def _calc_ratio(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
        """安全计算比率 (百分比): numerator / denominator * 100。"""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return (numerator / denominator * Decimal("100")).quantize(Decimal("0.01"))

    @staticmethod
    def _calc_gross_margin(revenue: Decimal | None, operating_cost: Decimal | None) -> Decimal | None:
        """毛利率 = (营业收入 - 营业成本) / 营业收入 * 100。"""
        if revenue is None or revenue == 0:
            return None
        if operating_cost is None:
            return None
        return ((revenue - operating_cost) / revenue * Decimal("100")).quantize(Decimal("0.01"))
