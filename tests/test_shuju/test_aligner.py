"""
测试 DataAligner 多源数据时间对齐。
"""

from decimal import Decimal

from shuju.data_aligner import DataAligner


class TestFinancialToDaily:
    def test_forward_fill_basic(self):
        aligner = DataAligner(fill_limit=90)
        financials = {
            "000001": [
                {"report_date": "20260331", "pe": Decimal("10"), "roe": Decimal("12")},
                {"report_date": "20260630", "pe": Decimal("11"), "roe": Decimal("13")},
            ]
        }
        trade_dates = ["20260401", "20260415", "20260701", "20260715"]
        result = aligner.financial_to_daily(financials, trade_dates)

        daily = result["000001"]
        assert len(daily) == 4
        # 4月: 使用 Q1 数据
        assert daily[0]["pe"] == Decimal("10")
        assert daily[1]["pe"] == Decimal("10")
        # 7月: 使用 Q2 数据
        assert daily[2]["pe"] == Decimal("11")
        assert daily[3]["pe"] == Decimal("11")

    def test_fill_limit_exceeded(self):
        aligner = DataAligner(fill_limit=30)
        financials = {
            "000001": [
                {"report_date": "20260331", "pe": Decimal("10")},
            ]
        }
        # 报告期距今超过 30 天，不填充
        trade_dates = ["20260701"]
        result = aligner.financial_to_daily(financials, trade_dates)
        assert "000001" not in result or len(result.get("000001", [])) == 0

    def test_empty_financials(self):
        aligner = DataAligner()
        result = aligner.financial_to_daily({}, ["20260601"])
        assert result == {}


class TestAlignToDaily:
    def test_basic_merge(self):
        aligner = DataAligner()
        daily_bars = {
            "000001": [
                {"trade_date": "20260601", "open": Decimal("10"), "close": Decimal("10.5")},
                {"trade_date": "20260602", "open": Decimal("10.5"), "close": Decimal("11")},
            ]
        }
        trade_dates = ["20260601", "20260602"]
        industry = {"000001": "银行"}

        result = aligner.align_to_daily(
            trade_dates, daily_bars, industry=industry
        )
        assert "000001" in result
        assert len(result["000001"]) == 2
        assert result["000001"][0]["industry"] == "银行"

    def test_missing_bar_date(self):
        aligner = DataAligner()
        daily_bars = {
            "000001": [
                {"trade_date": "20260601", "close": Decimal("10")},
            ]
        }
        result = aligner.align_to_daily(["20260601", "20260602"], daily_bars)
        # 20260602 没有 bar，跳过
        assert len(result["000001"]) == 1
        assert result["000001"][0]["trade_date"] == "20260601"


class TestSentimentToDaily:
    def test_expand_to_dates(self):
        aligner = DataAligner()
        sentiment = {"000001": {"score": 0.35, "total_mentions": 10}}
        trade_dates = ["20260601", "20260602", "20260603"]
        result = aligner.sentiment_to_daily(sentiment, trade_dates)
        assert len(result["000001"]) == 3
        assert all(r["sentiment_score"] == 0.35 for r in result["000001"])

    def test_empty_sentiment_skipped(self):
        aligner = DataAligner()
        sentiment = {"000001": {"score": 0, "total_mentions": 0}}
        result = aligner.sentiment_to_daily(sentiment, ["20260601"])
        assert "000001" not in result
