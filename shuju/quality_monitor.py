"""
数据质量监控器 (v4.0).

自动化检查: 完整性、时效性、分布漂移、重复。
每项检查返回 (pass, message, details).

Usage:
    from shuju.quality_monitor import DataQualityMonitor
    ok, msg = DataQualityMonitor.check_completeness(df, expected_cols)
"""

from decimal import Decimal

from shuju.utils import safe_mean


class DataQualityMonitor:
    """数据质量自动化监控. 纯静态方法."""

    @staticmethod
    def check_completeness(
        column_values: list[str | None],
        column_name: str = "unknown",
    ) -> tuple[bool, str]:
        """检查缺失值比例."""
        n = len(column_values)
        if n == 0:
            return False, f"{column_name}: empty column"
        missing = sum(1 for v in column_values if v is None)
        ratio = Decimal(missing) / Decimal(n)
        ok = ratio < Decimal("0.05")
        return ok, f"{column_name}: {missing}/{n} missing ({float(ratio):.1%})"

    @staticmethod
    def check_freshness(
        last_date: str,
        max_age_days: int = 3,
    ) -> tuple[bool, str]:
        """检查数据时效性.

        Stub: 需要进行日期差计算 (需引入 datetime 依赖).
        """
        return True, f"Last date: {last_date} (max age: {max_age_days}d)"

    @staticmethod
    def check_distribution(
        values: list[Decimal],
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> tuple[bool, str]:
        """检查数值分布是否在合理区间."""
        n = len(values)
        if n == 0:
            return False, "No values to check"

        mu = safe_mean(values)
        var = sum((v - mu) ** 2 for v in values) / Decimal(max(1, n - 1))
        sigma = var.sqrt()

        in_range = True
        if lower_bound is not None and float(mu - Decimal("3") * sigma) < lower_bound:
            in_range = False
        if upper_bound is not None and float(mu + Decimal("3") * sigma) > upper_bound:
            in_range = False

        return in_range, f"μ={float(mu):.4f} σ={float(sigma):.4f} n={n}"

    @staticmethod
    def check_duplicates(
        keys: list[str],
    ) -> tuple[bool, str]:
        """检查重复键."""
        n = len(keys)
        unique = len(set(keys))
        dup = n - unique
        ok = dup == 0
        return ok, f"{dup} duplicates in {n} keys"
