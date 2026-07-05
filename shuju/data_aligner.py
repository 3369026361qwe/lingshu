"""
多源数据时间对齐器。

将不同频率的数据源对齐到统一的交易日期：
    - 日线行情 (日频) → 直接对齐
    - 财务报表 (季频) → 前值填充到日频
    - 新闻舆情 (不定频) → 聚合到日频
    - 行业分类 (低频) → 前值填充

Usage:
    aligner = DataAligner()
    aligned = aligner.align_to_daily(
        daily_bars=bars,
        financials=fin_data,
        sentiment=sent_data,
        industry=ind_data,
    )
"""

import logging
from collections import defaultdict
from datetime import datetime

from shuju.config import get_config

FINANCIAL_FILL_LIMIT_DAYS = get_config().fill_limit_days

_logger = logging.getLogger(__name__)


class DataAligner:
    """多源数据时间对齐器。"""

    def __init__(self, fill_limit: int = 0):
        """
        Args:
            fill_limit: 前值填充的最大天数 (超过则标记为缺失)
        """
        self.fill_limit = fill_limit if fill_limit > 0 else FINANCIAL_FILL_LIMIT_DAYS

    # ── 日线对齐 ────────────────────────────────────────

    def align_to_daily(
        self,
        trade_dates: list[str],
        daily_bars: dict[str, list[dict]],       # {code: [bar, ...]}
        financials: dict[str, list[dict]] | None = None,
        sentiment: dict[str, list[dict]] | None = None,
        industry: dict[str, str] | None = None,  # {code: sw_level1}
    ) -> dict[str, list[dict]]:
        """将所有数据源对齐到统一交易日历。

        Args:
            trade_dates: 交易日列表 ["20260601", "20260602", ...]
            daily_bars: 日线行情 {code: [bar]}
            financials: 财务数据 {code: [report]}
            sentiment: 舆情数据 {code: [{date, score}]}
            industry: 行业分类 {code: sw_level1}

        Returns:
            {code: [{date, open, high, low, close, volume,
                     pe, roe, sentiment_score, industry, ...}]}
        """
        result: dict[str, list[dict]] = defaultdict(list)

        for code, bars in daily_bars.items():
            # 建立日期索引
            bar_map = {b.get("trade_date", ""): b for b in bars}

            for d in trade_dates:
                bar = bar_map.get(d)
                if bar is None:
                    continue

                aligned = dict(bar)

                # 合并财务数据 (前值填充，跳过与行情冲突的 key)
                if financials and code in financials:
                    fin = self._get_latest_financial(financials[code], d)
                    if fin:
                        _BAR_KEYS = {"code", "trade_date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"}
                        for k, v in fin.items():
                            if k not in _BAR_KEYS:
                                aligned[k] = v

                # 合并舆情
                if sentiment and code in sentiment:
                    sent = self._get_sentiment_for_date(sentiment[code], d)
                    if sent:
                        aligned["sentiment_score"] = sent.get("score", 0)

                # 合并行业
                if industry and code in industry:
                    aligned["industry"] = industry[code]

                result[code].append(aligned)

        return dict(result)

    # ── 财务数据日频化 ──────────────────────────────────

    def financial_to_daily(
        self,
        financials: dict[str, list[dict]],
        trade_dates: list[str],
    ) -> dict[str, list[dict]]:
        """将季频财务数据前值填充到日频。

        Args:
            financials: {code: [{report_date, pe, roe, ...}]}
            trade_dates: 交易日列表

        Returns:
            {code: [{trade_date, pe, roe, ...}]}
        """
        result: dict[str, list[dict]] = {}

        for code, reports in financials.items():
            if not reports:
                continue

            # 按报告期排序
            reports_sorted = sorted(reports, key=lambda r: r.get("report_date", ""))

            daily = []
            pointer = 0  # 指向当前有效的财务报告

            for d in trade_dates:
                # 推进 pointer 到最新报告
                while pointer + 1 < len(reports_sorted):
                    next_date = reports_sorted[pointer + 1].get("report_date", "")
                    if next_date <= d:
                        pointer += 1
                    else:
                        break

                current_report = reports_sorted[pointer]
                report_date = current_report.get("report_date", "")

                # 检查是否在有效期内
                if report_date and self._days_between(report_date, d) <= self.fill_limit:
                    row = {"trade_date": d}
                    # 复制财务字段
                    for key in ("pe", "pb", "ps", "roe", "roa", "gross_margin",
                                "net_margin", "revenue", "net_profit",
                                "operating_cashflow", "free_cashflow_yield"):
                        if key in current_report:
                            row[key] = current_report[key]
                    daily.append(row)

            if daily:
                result[code] = daily

        return result

    # ── 舆情日频化 ──────────────────────────────────────

    def sentiment_to_daily(
        self,
        sentiment_data: dict[str, dict],  # {code: {score, ...}}
        trade_dates: list[str],
    ) -> dict[str, list[dict]]:
        """将舆情快照扩展到日频 (同值填充)。

        舆情数据通常是最新快照，同一天内所有交易日使用相同值。
        """
        result: dict[str, list[dict]] = {}
        for code, sent in sentiment_data.items():
            if not sent or sent.get("total_mentions", 0) == 0:
                continue
            result[code] = [
                {"trade_date": d, "sentiment_score": sent.get("score", 0)}
                for d in trade_dates
            ]
        return result

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _get_latest_financial(reports: list[dict], trade_date: str) -> dict | None:
        """获取交易日当天最新的财务数据 (报告期不晚于交易日)。"""
        best = None
        for r in reports:
            rpt_date = r.get("report_date", "")
            if rpt_date <= trade_date:
                if best is None or rpt_date > best.get("report_date", ""):
                    best = r
        return best

    @staticmethod
    def _get_sentiment_for_date(sentiment_list: list[dict], trade_date: str) -> dict | None:
        """获取交易日对应的舆情数据（不含未来信息）。"""
        for s in sentiment_list:
            if s.get("date", "") == trade_date:
                return s
        # 返回 ≤ trade_date 的最新一条（无 look-ahead bias）
        best = None
        for s in sentiment_list:
            s_date = s.get("date", "")
            if s_date <= trade_date:
                if best is None or s_date > best.get("date", ""):
                    best = s
        return best

    @staticmethod
    def _days_between(date1: str, date2: str) -> int:
        """计算两个 YYYYMMDD 日期之间的天数。"""
        try:
            d1 = datetime.strptime(date1, "%Y%m%d")
            d2 = datetime.strptime(date2, "%Y%m%d")
            return abs((d2 - d1).days)
        except ValueError:
            return 999
