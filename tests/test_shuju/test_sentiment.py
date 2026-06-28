"""
测试 SentimentFetcher 情感分析：情感词统计、否定词反转、市场情绪指数。
"""

from unittest.mock import MagicMock

from shuju.sentiment_fetcher import SentimentFetcher


class TestWordCounting:
    def test_count_sentiment_words_pure_positive(self):
        text = "公司业绩大涨，盈利超预期，龙头地位稳固"
        pos, neg = SentimentFetcher._count_sentiment_words(text)
        assert pos >= 3
        assert neg == 0

    def test_count_sentiment_words_pure_negative(self):
        text = "公司业绩下滑，亏损扩大，面临退市风险"
        pos, neg = SentimentFetcher._count_sentiment_words(text)
        assert pos == 0
        assert neg >= 3

    def test_negation_reverses_positive_to_negative(self):
        """'没有大涨' — 否定词 + 正面情感词 → 算负面。"""
        text = "公司业绩没有大涨"
        pos, neg = SentimentFetcher._count_sentiment_words(text)
        # "大涨" 被否定 → neg += 1
        assert neg >= 1

    def test_negation_reverses_negative_to_positive(self):
        """'没有下跌' — 否定词 + 负面情感词 → 算正面。"""
        text = "股价没有下跌"
        pos, neg = SentimentFetcher._count_sentiment_words(text)
        # "下跌" 被否定 → pos += 1
        assert pos >= 1

    def test_double_negation_not_handled(self):
        """双重否定不作特别处理（已知局限）。"""
        text = "不是没有利好"
        pos, neg = SentimentFetcher._count_sentiment_words(text)
        # "利好"前面同时有"没有"和"不是"，但只检查窗口内的否定词
        assert pos + neg >= 0  # 至少不崩溃

    def test_negation_beyond_window(self):
        """否定词距离情感词太远不应反转。"""
        text = "不" + "x" * 10 + "利好"
        pos, neg = SentimentFetcher._count_sentiment_words(text)
        # 否定词距离"利好"超过 5 个字，不应反转
        assert pos == 1
        assert neg == 0

    def test_multiple_matches(self):
        """P2-7: 文本中多次出现同一情感词应全部计数。"""
        text = "大涨！继续大涨！持续大涨！"
        pos, neg = SentimentFetcher._count_sentiment_words(text)
        assert pos >= 3, f"Expected >=3 positive matches, got {pos}"


class TestSentimentLabel:
    def test_strong_bullish(self):
        assert SentimentFetcher._label(0.5) == "强烈看多"

    def test_neutral(self):
        assert SentimentFetcher._label(0) == "中性"

    def test_strong_bearish(self):
        assert SentimentFetcher._label(-0.5) == "强烈看空"


class TestSentimentFetcherMocked:
    def test_get_stock_sentiment_with_mock_news(self):
        """使用 mock NewsFetcher 测试情感分析流程。"""
        mock_news = MagicMock()
        mock_news.get_stock_news.return_value = [
            {"title": "业绩大涨", "content": "公司Q1盈利超预期"},
            {"title": "股价下跌", "content": "股东减持"},
        ]

        fetcher = SentimentFetcher(news_fetcher=mock_news)
        result = fetcher.get_stock_sentiment("000001")

        assert result["code"] == "000001"
        assert result["total_mentions"] == 2
        # 第一条正面 + 第二条负面
        assert -1 <= result["score"] <= 1

    def test_no_news_returns_empty(self):
        mock_news = MagicMock()
        mock_news.get_stock_news.return_value = []

        fetcher = SentimentFetcher(news_fetcher=mock_news)
        result = fetcher.get_stock_sentiment("000001")

        assert result["total_mentions"] == 0
        assert result["sentiment_label"] == "无数据"

    def test_cache_hit(self):
        mock_news = MagicMock()
        mock_news.get_stock_news.return_value = [
            {"title": "利好", "content": "增长"}
        ]

        fetcher = SentimentFetcher(news_fetcher=mock_news)
        result1 = fetcher.get_stock_sentiment("000001")
        # 第二次应从缓存读取
        result2 = fetcher.get_stock_sentiment("000001")
        assert result1["score"] == result2["score"]
        # NewsFetcher 只应被调用一次（第二次走缓存）
        assert mock_news.get_stock_news.call_count == 1
