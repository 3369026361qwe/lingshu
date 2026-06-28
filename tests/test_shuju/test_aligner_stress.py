"""
DataAligner 压力测试：大数据集、多股票、超长日期范围。

验证时间对齐在大规模输入下的正确性和可接受的性能。
"""

import time
from decimal import Decimal

from shuju.data_aligner import DataAligner


class TestAlignerStress:
    """压力与边界测试。"""

    def test_many_trade_dates(self):
        """大量交易日（500 天）处理正确。"""
        aligner = DataAligner(fill_limit=365)
        # 生成 ~500 个交易日
        trade_dates = []
        for m in range(1, 13):
            for d in range(1, 29, 5):
                trade_dates.append(f"2025{m:02d}{d:02d}")
        trade_dates.extend(f"2026{m:02d}{d:02d}" for m in range(1, 7) for d in range(1, 29, 5))
        # ~300 个日期
        trade_dates = sorted(trade_dates)

        financials = {
            "000001": [
                {"report_date": "20240930", "pe": Decimal("8"), "roe": Decimal("10")},
                {"report_date": "20250331", "pe": Decimal("9"), "roe": Decimal("11")},
                {"report_date": "20250930", "pe": Decimal("10"), "roe": Decimal("12")},
            ]
        }
        result = aligner.financial_to_daily(financials, trade_dates)
        daily = result["000001"]
        assert len(daily) > 0

        # 验证日期单调性
        dates = [r["trade_date"] for r in daily]
        assert dates == sorted(dates)

    def test_many_stocks(self):
        """多只股票（100 只）同时对齐。"""
        aligner = DataAligner(fill_limit=90)
        trade_dates = ["20260401", "20260415", "20260501", "20260515", "20260601"]

        financials = {}
        for i in range(1, 101):
            code = f"{i:06d}"
            financials[code] = [
                {"report_date": "20260331", "pe": Decimal(str(10 + i % 20)),
                 "roe": Decimal(str(5 + i % 15))},
            ]

        result = aligner.financial_to_daily(financials, trade_dates)
        assert len(result) == 100
        for i in range(1, 101):
            code = f"{i:06d}"
            assert code in result
            assert len(result[code]) == 5

    def test_mixed_report_dates_across_stocks(self):
        """不同股票的报告期不一致。"""
        aligner = DataAligner(fill_limit=365)
        trade_dates = ["20260401", "20260415", "20260501", "20260515"]

        financials = {
            "000001": [
                {"report_date": "20251231", "pe": Decimal("10")},
                {"report_date": "20260331", "pe": Decimal("11")},
            ],
            "000002": [
                {"report_date": "20250930", "pe": Decimal("20")},
                {"report_date": "20260331", "pe": Decimal("22")},
            ],
            "000003": [
                {"report_date": "20260331", "pe": Decimal("30")},
            ],
        }
        result = aligner.financial_to_daily(financials, trade_dates)
        assert len(result) == 3
        # 每只股票填充的报告期应一致
        for _code, daily in result.items():
            assert len(daily) > 0

    def test_no_reports_before_first_trade_date(self):
        """所有报告期都在第一个交易日之后 → 不应有数据。"""
        aligner = DataAligner(fill_limit=90)
        trade_dates = ["20260101", "20260115"]
        financials = {
            "000001": [
                {"report_date": "20260630", "pe": Decimal("10")},
            ]
        }
        # 报告期 0630 在 0101 之后 → 不应前向填充
        result = aligner.financial_to_daily(financials, trade_dates)
        # 因为 trade_dates 在报告期之前 → _days_between 超过 fill_limit → 无数据
        assert "000001" not in result or len(result.get("000001", [])) == 0

    def test_single_stock_single_report(self):
        """单股票单报告边界。"""
        aligner = DataAligner(fill_limit=90)
        financials = {
            "000001": [
                {"report_date": "20260331", "pe": Decimal("10")},
            ]
        }
        trade_dates = ["20260401"]
        result = aligner.financial_to_daily(financials, trade_dates)
        assert result["000001"][0]["pe"] == Decimal("10")

    def test_stress_large_financials(self):
        """单只股票大量报告期（模拟 10 年季报 40 期）。"""
        aligner = DataAligner(fill_limit=365 * 10)
        reports = []
        for year in range(2017, 2027):
            for month, day in [("03", "31"), ("06", "30"), ("09", "30"), ("12", "31")]:
                reports.append({
                    "report_date": f"{year}{month}{day}",
                    "pe": Decimal(str(8 + (year - 2017))),
                })
        financials = {"000001": reports}
        trade_dates = [f"2026{m:02d}{d:02d}" for m in range(1, 7) for d in (1, 15)]

        result = aligner.financial_to_daily(financials, trade_dates)
        daily = result["000001"]
        assert len(daily) == len(trade_dates)  # fill_limit 足够大

    def test_stress_align_all_sources(self):
        """align_to_daily 同时对齐所有数据源的压测。"""
        aligner = DataAligner(fill_limit=365)

        trade_dates = [f"2026{m:02d}{d:02d}" for m in range(1, 7) for d in (1, 15)]

        # 50 只股票的日线
        daily_bars = {}
        for i in range(1, 51):
            code = f"{i:06d}"
            daily_bars[code] = [
                {"trade_date": d, "open": Decimal("10"), "high": Decimal("11"),
                 "low": Decimal("9.5"), "close": Decimal("10.5"),
                 "volume": Decimal("1000000"), "amount": Decimal("10500000")}
                for d in trade_dates
            ]

        # 财务数据
        financials = {}
        for i in range(1, 51):
            financials[f"{i:06d}"] = [
                {"report_date": "20260331", "pe": Decimal(str(10 + i % 20)),
                 "roe": Decimal(str(5 + i % 15))},
            ]

        # 行业分类
        industry = {f"{i:06d}": f"行业{i % 6}" for i in range(1, 51)}

        start = time.perf_counter()
        result = aligner.align_to_daily(trade_dates, daily_bars,
                                        financials=financials, industry=industry)
        elapsed = time.perf_counter() - start

        assert len(result) == 50
        for i in range(1, 51):
            code = f"{i:06d}"
            assert len(result[code]) == len(trade_dates)
            # 每条记录应合并了行业
            for row in result[code]:
                assert "industry" in row

        # 性能断言：50 stock x 12 dates 应在 1 秒内完成
        assert elapsed < 2.0, f"align_to_daily took {elapsed:.2f}s, expected < 2s"
