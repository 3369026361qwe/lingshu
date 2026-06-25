"""
测试 DataCacheManager 数据层缓存管理器。
"""

from shuju.cache_manager import DataCacheManager


class TestDailyBarCache:
    def test_cache_and_retrieve_single_bar(self):
        cache = DataCacheManager()
        bar = {"code": "000001", "trade_date": "20260601", "close": "10.80"}
        cache.cache_daily_bar("000001", "20260601", bar)
        result = cache.get_daily_bar("000001", "20260601")
        assert result is not None
        assert result["close"] == "10.80"

    def test_cache_miss(self):
        cache = DataCacheManager()
        result = cache.get_daily_bar("999999", "20200101")
        assert result is None

    def test_batch_cache(self):
        cache = DataCacheManager()
        bars = [
            {"trade_date": "20260601", "close": "10.0"},
            {"trade_date": "20260602", "close": "10.5"},
        ]
        cache.cache_daily_bars_batch("000001", bars)
        result = cache.get_daily_bars_batch("000001")
        assert result is not None
        assert len(result) == 2


class TestFinancialCache:
    def test_cache_and_retrieve(self):
        cache = DataCacheManager()
        fin = {"pe": "8.5", "roe": "12.3"}
        cache.cache_financial("000001", "20260331", fin)
        result = cache.get_financial("000001", "20260331")
        assert result["pe"] == "8.5"


class TestNewsCache:
    def test_cache_and_retrieve(self):
        cache = DataCacheManager()
        news = {"id": "n1", "title": "test"}
        cache.cache_news("n1", news)
        result = cache.get_news("n1")
        assert result["title"] == "test"


class TestSentimentCache:
    def test_cache_and_retrieve(self):
        cache = DataCacheManager()
        sent = {"code": "000001", "score": 0.35}
        cache.cache_sentiment("000001", sent)
        result = cache.get_sentiment("000001")
        assert result["score"] == 0.35


class TestIndustryCache:
    def test_cache_and_retrieve(self):
        cache = DataCacheManager()
        ind = {"sw_level1": "银行"}
        cache.cache_industry("000001", ind)
        result = cache.get_industry("000001")
        assert result["sw_level1"] == "银行"


class TestClearAll:
    def test_clear_removes_all(self):
        cache = DataCacheManager()
        cache.cache_daily_bar("000001", "20260601", {"close": "10"})
        cache.cache_sentiment("000001", {"score": 0.5})
        cache.clear_all()
        assert cache.get_daily_bar("000001", "20260601") is None
        assert cache.get_sentiment("000001") is None
