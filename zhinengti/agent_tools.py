"""
Agent 共享工具集。

提供各 Agent 调用的数据查询接口:
    - query_market_data      → 行情数据 (通过 shuju 层)
    - query_financial_data   → 财务数据
    - query_news             → 新闻公告
    - query_factor_values    → 因子值 (通过 yinzi 层)
    - query_industry_map     → 行业分类
    - get_market_sentiment   → 市场情绪

所有工具函数签名统一为 (code_or_query: str, **kwargs) → dict | list[dict]
"""

import logging
from datetime import date
from decimal import Decimal

_logger = logging.getLogger(__name__)


class AgentToolkit:
    """Agent 工具集，封装数据层访问。"""

    def __init__(self, cache_manager=None):
        self._cache = cache_manager

    # ── 行情查询 ────────────────────────────────────────

    def query_market_data(self, code: str, days: int = 60) -> dict:
        """查询股票近期行情数据。

        Returns:
            {trade_date: {open, high, low, close, volume, amount, turnover_rate}}
        """
        try:
            from shuju.akshare_fetcher import AKShareFetcher
            fetcher = AKShareFetcher()
            bars = fetcher.get_daily_bars(code)
            if not bars:
                return {}
            # 取最近 N 天
            return {b["trade_date"]: b for b in bars[-days:]}
        except Exception as exc:
            _logger.warning("query_market_data(%s) failed: %s", code, exc)
            return {}

    # ── 财务查询 ────────────────────────────────────────

    def query_financial_data(self, code: str) -> dict:
        """查询股票最新财务数据。

        Returns:
            {pe, pb, roe, roa, gross_margin, net_margin, revenue, net_profit, ...}
        """
        try:
            from shuju.tushare_fetcher import TushareFetcher
            fetcher = TushareFetcher()
            reports = fetcher.get_financial_reports(code)
            return reports[-1] if reports else {}
        except Exception as exc:
            _logger.warning("query_financial_data(%s) failed: %s", code, exc)
            return {}

    # ── 新闻查询 ────────────────────────────────────────

    def query_news(self, code: str, limit: int = 10) -> list[dict]:
        """查询个股相关新闻。"""
        try:
            from shuju.news_fetcher import NewsFetcher
            fetcher = NewsFetcher()
            return fetcher.get_stock_news(code, limit=limit)
        except Exception as exc:
            _logger.warning("query_news(%s) failed: %s", code, exc)
            return []

    # ── 因子查询 ────────────────────────────────────────

    def query_factor_values(self, code: str, factor_names: list[str] | None = None) -> dict[str, Decimal]:
        """查询股票最新因子值。"""
        try:
            from shujuku.repository import Repository
            from shujuku.session import SessionContext

            if factor_names is None:
                factor_names = ["pe", "momentum_1m", "roe", "historical_vol"]

            result = {}
            with SessionContext() as s:
                repo = Repository(s)
                for fname in factor_names:
                    values = repo.get_factor_values(
                        code, fname,
                        date.today().replace(year=date.today().year - 1),
                        date.today(),
                    )
                    if values:
                        result[fname] = values[-1].raw_value
            return result
        except Exception as exc:
            _logger.warning("query_factor_values(%s) failed: %s", code, exc)
            return {}

    # ── 行业查询 ────────────────────────────────────────

    def query_industry_map(self) -> dict[str, str]:
        """查询股票→行业映射。"""
        try:
            from shuju.akshare_fetcher import AKShareFetcher
            fetcher = AKShareFetcher()
            raw = fetcher.get_industry_map()
            return {code: info.get("sw_level1", "") for code, info in raw.items()}
        except Exception as exc:
            _logger.warning("query_industry_map failed: %s", exc)
            return {}

    # ── 情绪查询 ────────────────────────────────────────

    def get_market_sentiment(self, sample_codes: list[str] | None = None) -> dict:
        """获取全市场情绪指数。"""
        try:
            from shuju.sentiment_fetcher import SentimentFetcher
            fetcher = SentimentFetcher()
            return fetcher.get_market_sentiment_index(sample_codes)
        except Exception as exc:
            _logger.warning("get_market_sentiment failed: %s", exc)
            return {"index": 0.0, "label": "无数据"}

    # ── 获取分析上下文 ──────────────────────────────────

    def build_context(self, stock_list: list[str]) -> dict:
        """为给定股票列表构建完整的分析上下文。

        Returns:
            {stocks: [...], market_data: {...}, news: [...], sentiment: {...}, industry: {...}}
        """
        context = {
            "stocks": stock_list,
            "market_data": {},
            "financial_data": {},
            "news": {},
            "sentiment": self.get_market_sentiment(),
            "industry": self.query_industry_map(),
        }

        for code in stock_list[:20]:  # 限制上下文大小
            context["market_data"][code] = self.query_market_data(code, days=60)
            context["financial_data"][code] = self.query_financial_data(code)
            context["news"][code] = self.query_news(code, limit=5)

        return context
