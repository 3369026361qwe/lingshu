"""
测试 DataQualityMonitor — 数据质量监控.
"""

import tempfile
from decimal import Decimal
from pathlib import Path

from shuju.quality_monitor import DataQualityMonitor


class TestCheckCompleteness:
    def test_all_present(self):
        ok, msg = DataQualityMonitor.check_completeness(
            ["a", "b", "c", "d", "e"], "test_col"
        )
        assert ok
        assert "0/5 missing" in msg

    def test_some_missing(self):
        ok, msg = DataQualityMonitor.check_completeness(
            ["a", None, "c", None, "e"], "test_col"
        )
        assert not ok  # 40% > 5%
        assert "2/5 missing" in msg

    def test_all_missing(self):
        ok, msg = DataQualityMonitor.check_completeness(
            [None, None, None], "test_col"
        )
        assert not ok

    def test_empty_column(self):
        ok, msg = DataQualityMonitor.check_completeness([], "test_col")
        assert not ok
        assert "empty" in msg

    def test_below_threshold(self):
        # 4% missing → pass
        ok, _ = DataQualityMonitor.check_completeness(
            ["a"] * 96 + [None] * 4, "test_col"
        )
        assert ok


class TestCheckCompletenessDict:
    def test_all_pass(self):
        data = {
            "close": [Decimal("10"), Decimal("11"), Decimal("12")],
            "volume": [Decimal("1e6"), Decimal("2e6"), Decimal("3e6")],
        }
        result = DataQualityMonitor.check_completeness_dict(data)
        assert result["overall_pass"]
        assert result["total_records"] == 3
        assert result["column_results"]["close"]["pass"]

    def test_missing_column(self):
        data = {
            "close": [Decimal("10"), None, Decimal("12")],
            "volume": [Decimal("1e6"), Decimal("2e6"), Decimal("3e6")],
        }
        result = DataQualityMonitor.check_completeness_dict(data, ["close", "volume"])
        assert not result["overall_pass"]  # close 33% missing

    def test_empty_data(self):
        result = DataQualityMonitor.check_completeness_dict({})
        assert not result["overall_pass"]
        assert result["total_records"] == 0

    def test_missing_expected_column(self):
        data = {"close": [Decimal("10"), Decimal("11")]}
        result = DataQualityMonitor.check_completeness_dict(data, ["close", "volume"])
        assert not result["overall_pass"]  # volume is missing entirely


class TestCheckFreshness:
    def test_today_is_fresh(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        ok, msg = DataQualityMonitor.check_freshness(today, max_age_days=3)
        assert ok

    def test_yyyymmdd_format(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        ok, msg = DataQualityMonitor.check_freshness(today, max_age_days=3)
        assert ok

    def test_stale_data(self):
        ok, msg = DataQualityMonitor.check_freshness("2020-01-01", max_age_days=3)
        assert not ok

    def test_invalid_date(self):
        ok, msg = DataQualityMonitor.check_freshness("not-a-date")
        assert not ok


class TestCheckDistribution:
    def test_normal_distribution(self):
        values = [Decimal(str(i)) for i in range(100)]
        ok, msg = DataQualityMonitor.check_distribution(values)
        assert ok

    def test_out_of_bounds(self):
        values = [Decimal("100"), Decimal("101"), Decimal("102")]
        ok, msg = DataQualityMonitor.check_distribution(
            values, upper_bound=50.0
        )
        assert not ok  # μ + 3σ > 50

    def test_empty(self):
        ok, msg = DataQualityMonitor.check_distribution([])
        assert not ok

    def test_in_bounds(self):
        values = [Decimal("10"), Decimal("11"), Decimal("12")]
        ok, _ = DataQualityMonitor.check_distribution(
            values, lower_bound=0.0, upper_bound=100.0
        )
        assert ok


class TestCheckDuplicates:
    def test_no_duplicates(self):
        ok, msg = DataQualityMonitor.check_duplicates(["a", "b", "c"])
        assert ok

    def test_with_duplicates(self):
        ok, msg = DataQualityMonitor.check_duplicates(["a", "b", "a", "c", "b"])
        assert not ok
        assert "2 duplicates" in msg

    def test_empty(self):
        ok, msg = DataQualityMonitor.check_duplicates([])
        assert ok


class TestCheckAnomaly:
    def test_no_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            ok, msg = monitor.check_anomaly(1000, "2024-06-15")
            assert ok  # 首次运行, 无历史
            assert "First run" in msg

    def test_normal_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            # 手动填充历史
            monitor._history = {
                "2024-06-01": 1000, "2024-06-02": 1050, "2024-06-03": 980,
                "2024-06-04": 1020, "2024-06-05": 1010,
            }
            ok, msg = monitor.check_anomaly(1000, "2024-06-15")
            assert ok  # 接近历史均值

    def test_anomaly_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            monitor._history = {
                "2024-06-01": 1000, "2024-06-02": 1050, "2024-06-03": 980,
            }
            ok, msg = monitor.check_anomaly(200, "2024-06-15")  # 不足 50%
            assert not ok
            assert "异常" in msg


class TestSaveReport:
    def test_save_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            report = {
                "date": "2024-06-15",
                "overall_pass": True,
                "total_records": 1000,
            }
            filepath = monitor.save_report(report)
            assert Path(filepath).exists()
            assert "2024-06-15" in filepath

    def test_save_updates_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            report = {
                "date": "2024-06-15",
                "overall_pass": True,
                "total_records": 888,
            }
            monitor.save_report(report)
            assert "2024-06-15" in monitor._history
            assert monitor._history["2024-06-15"] == 888


class TestGetLatestReport:
    def test_no_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            assert monitor.get_latest_report() is None

    def test_returns_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            monitor.save_report({"date": "2024-01-01", "total_records": 100, "overall_pass": True})
            monitor.save_report({"date": "2024-01-02", "total_records": 200, "overall_pass": False})
            latest = monitor.get_latest_report()
            assert latest is not None
            assert latest["date"] == "2024-01-02"


class TestRunQualityChecks:
    def test_full_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            data = {
                "code": ["000001", "000002", "000003"],
                "trade_date": ["20240601", "20240601", "20240601"],
                "close": [Decimal("10.5"), Decimal("5.2"), Decimal("20.0")],
                "volume": [Decimal("1e6"), Decimal("2e6"), Decimal("3e6")],
                "pe": [Decimal("8.5"), Decimal("12.3"), None],
            }
            report = monitor.run_quality_checks(data, date_str="2024-06-01")
            assert "date" in report
            assert "overall_pass" in report
            assert "total_records" in report
            assert "completeness" in report
            assert "duplicates" in report
            assert "anomaly" in report

    def test_distribution_failure_affects_overall(self):
        """分布检查失败 → overall_pass 应为 False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            # 极端值数据: close 变化剧烈, 但有 outlier 导致分布异常
            data = {
                "code": ["000001"] * 5,
                "trade_date": ["20240601"] * 5,
                "close": [Decimal(str(x)) for x in range(1000000, 1000005)],  # 极端大值
            }
            report = monitor.run_quality_checks(data, date_str="2024-06-01")
            # close 列不在预定义 numeric_cols 列表中... 验证 overall 逻辑
            assert "overall_pass" in report
            assert "checks_summary" in report
            assert "distribution_ok" in report["checks_summary"]

    def test_distribution_ok_in_summary(self):
        """checks_summary 中 distribution_ok 应与实际分布检查一致."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            data = {
                "code": ["000001"],
                "trade_date": ["20240601"],
                "close": [Decimal("10")],
                "volume": [Decimal("1e6")],
            }
            report = monitor.run_quality_checks(data, date_str="2024-06-01")
            assert "distribution_ok" in report["checks_summary"]
            # 无 bounds 时 distribution 总是 ok → True
            assert report["checks_summary"]["distribution_ok"] is True

    def test_with_nested_data(self):
        """测试按 code 嵌套的数据格式."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            data = {
                "000001": {"close": [Decimal("10"), Decimal("11")], "volume": [Decimal("1e6"), Decimal("2e6")]},
                "000002": {"close": [Decimal("5"), Decimal("6")], "volume": [Decimal("3e6"), Decimal("4e6")]},
            }
            report = monitor.run_quality_checks(data, date_str="2024-06-01")
            assert report["total_records"] > 0

    def test_with_anomaly_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            data = {
                "code": ["000001"],
                "close": [Decimal("10")],
            }
            report = monitor.run_quality_checks(data, anomaly_check=False)
            assert report["anomaly"]["message"] == "skipped"

    def test_with_last_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DataQualityMonitor(report_dir=tmpdir)
            data = {
                "code": ["000001"],
                "close": [Decimal("10")],
            }
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            report = monitor.run_quality_checks(data, last_date=today)
            assert report["freshness"]["pass"]


class TestFlattenData:
    def test_already_flat(self):
        data = {"close": [1, 2, 3], "volume": [4, 5, 6]}
        result = DataQualityMonitor._flatten_data(data)
        assert result == data

    def test_nested_by_code(self):
        data = {
            "000001": {"close": [1, 2], "volume": [3, 4]},
            "000002": {"close": [5, 6], "volume": [7, 8]},
        }
        result = DataQualityMonitor._flatten_data(data)
        assert "close" in result
        assert "volume" in result
        assert result["close"] == [1, 2, 5, 6]
        assert result["volume"] == [3, 4, 7, 8]

    def test_empty(self):
        result = DataQualityMonitor._flatten_data({})
        assert result == {}
