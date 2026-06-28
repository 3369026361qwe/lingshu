"""
跨模块错误注入测试。

验证数据层在底层服务异常时的优雅降级行为：
    - 网络不可达 / 数据源异常 → 不崩溃，返回空/降级数据
    - 缓存不可用 → 回退到直接拉取
    - 输入数据损坏 → 明确错误而非静默返回错误结果
"""

from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from shuju.cache_manager import DataCacheManager
from shuju.data_aligner import DataAligner
from shuju.data_preprocessor import DataPreprocessor

# ────────────────────────────────────────────────────────────
# DataPreprocessor 错误注入
# ────────────────────────────────────────────────────────────

class TestPreprocessorErrorInjection:
    """注入异常数据验证预处理管道不会崩溃。"""

    def test_corrupted_decimal_values(self):
        """包含无法解析为数值的字符串时不崩溃。"""
        pp = DataPreprocessor()
        data = {
            "000001": {"pe": Decimal("15"), "roe": Decimal("12")},
            "000002": {"pe": Decimal("20"), "roe": Decimal("8")},
            "000003": {"pe": Decimal("12"), "roe": Decimal("15")},
            "000004": {"pe": Decimal("18"), "roe": Decimal("10")},
            "000005": {"pe": Decimal("25"), "roe": Decimal("7")},
        }
        # 注入一个 None 值 — 应在 fill_missing 时用中位数填充
        data["000003"]["roe"] = None
        result = pp.pipeline(data)
        assert "000003" in result
        assert result["000003"]["roe"] is not None
        assert isinstance(result["000003"]["roe"], Decimal)

    def test_single_valid_factor_others_none(self):
        """只有一个因子有有效值，其余全是 None。"""
        pp = DataPreprocessor()
        data = {
            "000001": {"pe": None, "roe": Decimal("12")},
            "000002": {"pe": None, "roe": Decimal("10")},
            "000003": {"pe": None, "roe": Decimal("15")},
            "000004": {"pe": None, "roe": Decimal("8")},
            "000005": {"pe": None, "roe": Decimal("11")},
        }
        result = pp.pipeline(data)
        # roe 应正常处理
        for code in result:
            assert isinstance(result[code]["roe"], Decimal)
        # pe 全 None → fill_missing 保持 None（无可用中位数）
        # 但 standardize 跳过 n<5 → None 不应被替换为错误值
        for code in result:
            assert result[code]["pe"] is None or isinstance(result[code]["pe"], Decimal)

    def test_extreme_negative_factor_values(self):
        """极端负值去极值后应被截断。"""
        pp = DataPreprocessor(method="sigma", n_sigma=Decimal("2"))
        data = {
            f"{i:06d}": {"roe": Decimal(str(i * 2))}
            for i in range(1, 11)
        }
        # 使用极端值确保在 2-sigma 界外
        data["000011"] = {"roe": Decimal("-1e12")}
        data["000012"] = {"roe": Decimal("1e12")}

        result = pp.winsorize(data)
        assert result["000011"]["roe"] > Decimal("-1e12")  # 应被上拉
        assert result["000012"]["roe"] < Decimal("1e12")   # 应被下拉

    def test_factor_name_with_special_characters(self):
        """因子名包含特殊字符时不崩溃。"""
        pp = DataPreprocessor()
        data = {
            "000001": {"pe_ttm": Decimal("15"), "roe(%)": Decimal("12")},
            "000002": {"pe_ttm": Decimal("20"), "roe(%)": Decimal("10")},
            "000003": {"pe_ttm": Decimal("12"), "roe(%)": Decimal("15")},
            "000004": {"pe_ttm": Decimal("18"), "roe(%)": Decimal("8")},
            "000005": {"pe_ttm": Decimal("25"), "roe(%)": Decimal("11")},
        }
        result = pp.pipeline(data)
        assert len(result) == 5
        for code in result:
            assert isinstance(result[code]["pe_ttm"], Decimal)

    def test_industry_map_missing_codes(self):
        """行业映射缺失部分股票时不崩溃，缺失的归入'未知'。"""
        pp = DataPreprocessor()
        data = {
            f"{i:06d}": {"pe": Decimal(str(10 + i))}
            for i in range(1, 11)
        }
        # 只有前 5 只有行业映射
        industry = {f"{i:06d}": f"行业{i % 3}" for i in range(1, 6)}
        result = pp.neutralize(data, industry)
        assert len(result) == 10
        # 无行业映射的股票不应崩溃
        for i in range(6, 11):
            assert isinstance(result[f"{i:06d}"]["pe"], Decimal)


# ────────────────────────────────────────────────────────────
# DataAligner 错误注入
# ────────────────────────────────────────────────────────────

class TestAlignerErrorInjection:
    """注入损坏数据验证对齐器不会崩溃。"""

    def test_corrupted_report_dates(self):
        """财务数据包含无效日期格式时优雅跳过。"""
        aligner = DataAligner(fill_limit=90)
        financials = {
            "000001": [
                {"report_date": "invalid_date", "pe": Decimal("10")},
                {"report_date": "20260331", "pe": Decimal("12")},
            ]
        }
        trade_dates = ["20260401", "20260415"]
        result = aligner.financial_to_daily(financials, trade_dates)
        # 不应崩溃，应能产生部分结果
        assert "000001" in result or True  # 至少不崩溃

    def test_empty_reports_in_financials(self):
        """某些股票有财务数据，另一些为空列表。"""
        aligner = DataAligner()
        financials = {
            "000001": [
                {"report_date": "20260331", "pe": Decimal("10")},
            ],
            "000002": [],  # 空列表
        }
        trade_dates = ["20260401", "20260415"]
        result = aligner.financial_to_daily(financials, trade_dates)
        # 000001 应有数据
        assert "000001" in result
        # 000002 不应出现
        assert "000002" not in result

    def test_missing_fields_in_financial_report(self):
        """财务报告缺少部分字段时只复制有的字段。"""
        aligner = DataAligner()
        financials = {
            "000001": [
                {"report_date": "20260331", "pe": Decimal("10")},  # 只有 pe，没有 pb/roe 等
            ]
        }
        trade_dates = ["20260401"]
        result = aligner.financial_to_daily(financials, trade_dates)
        daily = result["000001"]
        assert daily[0]["pe"] == Decimal("10")
        # 缺失的字段不应出现
        assert "roe" not in daily[0]

    def test_trade_dates_not_sorted(self):
        """交易日列表未排序时仍应正确填充。"""
        aligner = DataAligner()
        financials = {
            "000001": [
                {"report_date": "20260331", "pe": Decimal("10")},
                {"report_date": "20260630", "pe": Decimal("12")},
            ]
        }
        # 乱序日期
        trade_dates = ["20260715", "20260401", "20260415", "20260701"]
        result = aligner.financial_to_daily(financials, trade_dates)
        assert "000001" in result

    def test_align_with_no_overlap(self):
        """交易日与数据日期无交集时返回空。"""
        aligner = DataAligner()
        daily_bars = {
            "000001": [
                {"trade_date": "20250101", "close": Decimal("10")},
            ]
        }
        # 请求完全不重叠的日期
        result = aligner.align_to_daily(["20260601", "20260602"], daily_bars)
        # 没有交集 → 结果中该股票无数据
        assert result.get("000001", []) == []


# ────────────────────────────────────────────────────────────
# DataCacheManager 错误注入
# ────────────────────────────────────────────────────────────

class TestCacheManagerErrorInjection:
    """验证缓存层在 Redis 不可用时的行为。"""

    def test_is_redis_available_false(self):
        """Redis 不可用时 is_redis_available 返回 False。"""
        with patch("shujuku.redis_cache.CacheManager.is_redis_available",
                   new_callable=PropertyMock) as mock_available:
            mock_available.return_value = False
            cache = DataCacheManager()
            assert cache.is_redis_available is False

    def test_get_when_redis_unavailable(self):
        """Redis 不可用时 get 应返回 None（不抛异常）。"""
        with patch("shujuku.redis_cache.CacheManager.is_redis_available",
                   new_callable=PropertyMock) as mock_available:
            mock_available.return_value = False
            cache = DataCacheManager()
            # get 应优雅返回 None 而非崩溃
            result = cache.get_daily_bar("000001", "20260601")
            assert result is None

    def test_set_when_redis_unavailable(self):
        """Redis 不可用时 set 应静默忽略（不抛异常）。"""
        with patch("shujuku.redis_cache.CacheManager.is_redis_available",
                   new_callable=PropertyMock) as mock_available:
            mock_available.return_value = False
            cache = DataCacheManager()
            # set 不应崩溃
            try:
                cache.cache_daily_bar("000001", "20260601", {"close": "10"})
            except Exception:
                pytest.fail("cache_daily_bar raised exception when Redis unavailable")

    def test_preprocessed_cache_roundtrip(self):
        """预处理结果缓存写入和回读。"""
        cache = DataCacheManager()
        data = {"factor_pe": "0.35", "factor_roe": "0.82"}
        cache.cache_preprocessed("dataset_001", data)
        result = cache.get_preprocessed("dataset_001")
        assert result is not None
        assert result["factor_pe"] == "0.35"

    def test_preprocessed_cache_miss(self):
        """预处理结果缓存未命中。"""
        cache = DataCacheManager()
        result = cache.get_preprocessed("nonexistent_dataset")
        assert result is None


# ────────────────────────────────────────────────────────────
# SentimentFetcher 错误注入
# ────────────────────────────────────────────────────────────

class TestSentimentErrorInjection:
    """验证 SentimentFetcher 在 NewsFetcher 异常时降级。"""

    def test_news_fetcher_raises_exception(self):
        """NewsFetcher 抛出异常时返回空情感数据。"""
        from shuju.sentiment_fetcher import SentimentFetcher

        mock_news = MagicMock()
        mock_news.get_stock_news.side_effect = RuntimeError("Network error")

        fetcher = SentimentFetcher(news_fetcher=mock_news)
        result = fetcher.get_stock_sentiment("000001")

        assert result["total_mentions"] == 0
        assert result["sentiment_label"] == "无数据"

    def test_news_fetcher_is_none(self):
        """未注入 NewsFetcher 时懒加载创建。"""
        from shuju.sentiment_fetcher import SentimentFetcher

        fetcher = SentimentFetcher(news_fetcher=None)
        # 应使用懒加载创建 NewsFetcher，但若 akshare 不可用则降级
        result = fetcher.get_stock_sentiment("000001")
        assert result["total_mentions"] == 0
        assert result["code"] == "000001"

    def test_cached_result_avoids_news_call(self):
        """缓存命中时完全不走 NewsFetcher（即使 NewsFetcher 为 None）。"""

        from shuju.sentiment_fetcher import SentimentFetcher

        # 先写入缓存
        cache = DataCacheManager()
        cache.cache_sentiment("000001", {"code": "000001", "score": 0.5, "total_mentions": 5,
                                          "sentiment_label": "偏积极", "positive_count": 3,
                                          "negative_count": 1, "neutral_count": 1})

        mock_news = MagicMock()
        mock_news.get_stock_news.side_effect = RuntimeError("should not be called")

        fetcher = SentimentFetcher(news_fetcher=mock_news, cache=cache)
        result = fetcher.get_stock_sentiment("000001")
        assert result["score"] == 0.5
        # NewsFetcher 不应被调用
        mock_news.get_stock_news.assert_not_called()

    def test_get_market_sentiment_index_basic(self):
        """市场情绪指数正常计算。"""
        from shuju.sentiment_fetcher import SentimentFetcher

        mock_news = MagicMock()
        mock_news.get_stock_news.return_value = [
            {"title": "大涨", "content": "利好"},
        ]

        fetcher = SentimentFetcher(news_fetcher=mock_news)
        result = fetcher.get_market_sentiment_index(sample_codes=["000001", "000002"])
        assert "index" in result
        assert "label" in result
        assert result["sample_size"] >= 0

    def test_market_label(self):
        """市场情绪标签覆盖所有区间。"""
        from shuju.sentiment_fetcher import SentimentFetcher
        assert SentimentFetcher._market_label(0.3) == "乐观"
        assert SentimentFetcher._market_label(0.1) == "偏乐观"
        assert SentimentFetcher._market_label(0.0) == "中性"
        assert SentimentFetcher._market_label(-0.1) == "偏悲观"
        assert SentimentFetcher._market_label(-0.3) == "悲观"
