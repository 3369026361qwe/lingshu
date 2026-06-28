"""
数据预处理管道。

流水线:
    1. 去极值 (MAD / 标准差法)
    2. 缺失值处理 (行业均值填充 / 前值填充)
    3. 标准化 (Z-Score)
    4. 行业中性化 (截面回归去行业均值)

所有计算使用 Decimal，精度 18 位。

Usage:
    pp = DataPreprocessor()
    cleaned = pp.pipeline(raw_data, industry_map)
"""

import logging
import time
from decimal import Decimal

from shuju.constants import FILL_METHOD, MIN_SAMPLE_SIZE, WINSORIZE_METHOD, WINSORIZE_N_SIGMA
from shuju.metrics import preprocessor_duration, preprocessor_records_total

_logger = logging.getLogger(__name__)

# 默认 MAD 常数 (约等于 1.4826 * σ 对于正态分布)
_MAD_SCALE = Decimal("0.6745")


def _decimal_median(values: list[Decimal]) -> Decimal:
    """Decimal-native median — 避免 float 转换精度丢失。"""
    if not values:
        return Decimal("0")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / Decimal("2")


class DataPreprocessor:
    """数据预处理管道。"""

    def __init__(
        self,
        method: str = "",          # "mad" | "sigma"，空字符串表示使用环境变量
        n_sigma: Decimal | None = None,
        fill_method: str = "",     # "median" | "forward" | "zero"
    ) -> None:
        self.method = method or WINSORIZE_METHOD
        self.n_sigma = n_sigma if n_sigma is not None else Decimal(str(WINSORIZE_N_SIGMA))
        self.fill_method = fill_method or FILL_METHOD

    # ── Pipeline ────────────────────────────────────────

    def pipeline(
        self,
        data: dict[str, dict[str, Decimal]],  # {code: {factor_name: value}}
        industry_map: dict[str, str] | None = None,  # {code: sw_level1}
    ) -> dict[str, dict[str, Decimal]]:
        """完整预处理流水线。

        Args:
            data: {code: {factor_name: raw_value}}
            industry_map: {code: industry_name}

        Returns:
            {code: {factor_name: processed_value}}
        """
        n_records = sum(len(factors) for factors in data.values())

        t0 = time.perf_counter()
        result = self.winsorize(data)
        preprocessor_duration.labels(stage="winsorize").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result = self.fill_missing(result)
        preprocessor_duration.labels(stage="fill_missing").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result = self.standardize(result)
        preprocessor_duration.labels(stage="standardize").observe(time.perf_counter() - t0)

        if industry_map:
            t0 = time.perf_counter()
            result = self.neutralize(result, industry_map)
            preprocessor_duration.labels(stage="neutralize").observe(time.perf_counter() - t0)

        preprocessor_records_total.labels(stage="pipeline").inc(n_records)
        return result

    # ── 1. 去极值 ──────────────────────────────────────

    def winsorize(
        self, data: dict[str, dict[str, Decimal]]
    ) -> dict[str, dict[str, Decimal]]:
        """MAD / 标准差法去极值。

        将超出 [median - n*mad, median + n*mad] 的值截断到边界。
        不修改输入数据。
        """
        if not data:
            return data

        # 深拷贝，避免原地修改
        data = {code: dict(factors) for code, factors in data.items()}

        # 收集所有因子名
        factor_names: set[str] = set()
        for factors in data.values():
            factor_names.update(factors.keys())

        for fname in factor_names:
            values = []
            for _code, factors in data.items():
                v = factors.get(fname)
                if v is not None:
                    values.append(v)

            if len(values) < MIN_SAMPLE_SIZE:
                continue

            if self.method == "mad":
                lower, upper = self._mad_bounds(values)
            else:
                lower, upper = self._sigma_bounds(values)

            if lower is None or upper is None:
                continue

            # 截断
            for code in data:
                v = data[code].get(fname)
                if v is not None:
                    if v < lower:
                        data[code][fname] = lower
                    elif v > upper:
                        data[code][fname] = upper

        return data

    def _mad_bounds(self, values: list[Decimal]) -> tuple[Decimal | None, Decimal | None]:
        """MAD 法计算上下界。"""
        try:
            med = _decimal_median(values)
            abs_devs = [abs(v - med) for v in values]
            mad = _decimal_median(abs_devs)
            if mad == 0:
                return None, None
            bound = self.n_sigma * mad / _MAD_SCALE
            return med - bound, med + bound
        except Exception as exc:
            _logger.warning("MAD bounds failed: %s", exc)
            return None, None

    def _sigma_bounds(self, values: list[Decimal]) -> tuple[Decimal | None, Decimal | None]:
        """标准差法计算上下界。"""
        try:
            n = Decimal(len(values))
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = variance.sqrt()
            if std == 0:
                return None, None
            bound = self.n_sigma * std
            return mean - bound, mean + bound
        except Exception as exc:
            _logger.warning("Sigma bounds failed: %s", exc)
            return None, None

    # ── 2. 缺失值处理 ──────────────────────────────────

    def fill_missing(
        self, data: dict[str, dict[str, Decimal]]
    ) -> dict[str, dict[str, Decimal]]:
        """缺失值填充（不修改输入）。"""
        if not data:
            return data
        data = {code: dict(factors) for code, factors in data.items()}

        factor_names: set[str] = set()
        for factors in data.values():
            factor_names.update(factors.keys())

        for fname in factor_names:
            # 计算中位数
            present_values = []
            for factors in data.values():
                v = factors.get(fname)
                if v is not None:
                    present_values.append(v)

            if not present_values:
                continue

            if self.fill_method == "median":
                fill_value = _decimal_median(present_values)
            elif self.fill_method == "zero":
                fill_value = Decimal("0")
            else:
                fill_value = _decimal_median(present_values)

            for code in data:
                if data[code].get(fname) is None:
                    data[code][fname] = fill_value

        return data

    # ── 3. Z-Score 标准化 ──────────────────────────────

    def standardize(
        self, data: dict[str, dict[str, Decimal]]
    ) -> dict[str, dict[str, Decimal]]:
        """Z-Score 标准化: (x - μ) / σ（不修改输入）。"""
        if not data:
            return data
        data = {code: dict(factors) for code, factors in data.items()}

        factor_names: set[str] = set()
        for factors in data.values():
            factor_names.update(factors.keys())

        for fname in factor_names:
            values = []
            for factors in data.values():
                v = factors.get(fname)
                if v is not None:
                    values.append(v)

            if len(values) < MIN_SAMPLE_SIZE:
                continue

            n = Decimal(len(values))
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = variance.sqrt()

            if std == 0:
                for code in data:
                    if fname in data[code]:
                        data[code][fname] = Decimal("0")
                continue

            for code in data:
                v = data[code].get(fname)
                if v is not None:
                    data[code][fname] = (v - mean) / std

        return data

    # ── 4. 行业中性化 ──────────────────────────────────

    def neutralize(
        self,
        data: dict[str, dict[str, Decimal]],
        industry_map: dict[str, str],
    ) -> dict[str, dict[str, Decimal]]:
        """行业中性化: 减去行业内均值（不修改输入）。"""
        if not data or not industry_map:
            return data
        data = {code: dict(factors) for code, factors in data.items()}

        factor_names: set[str] = set()
        for factors in data.values():
            factor_names.update(factors.keys())

        for fname in factor_names:
            # 按行业分组
            industry_values: dict[str, list[Decimal]] = {}
            for code, factors in data.items():
                industry = industry_map.get(code, "未知")
                v = factors.get(fname)
                if v is not None:
                    industry_values.setdefault(industry, []).append(v)

            # 计算行业均值
            industry_means: dict[str, Decimal] = {}
            for ind, vals in industry_values.items():
                if vals:
                    industry_means[ind] = sum(vals) / Decimal(len(vals))

            # 减去行业均值
            for code in data:
                industry = industry_map.get(code, "未知")
                ind_mean = industry_means.get(industry)
                v = data[code].get(fname)
                if v is not None and ind_mean is not None:
                    data[code][fname] = v - ind_mean

        return data

    # ── 工具 ────────────────────────────────────────────

    def compute_percentiles(
        self, data: dict[str, dict[str, Decimal]]
    ) -> dict[str, dict[str, Decimal]]:
        """计算因子值的全市场分位数 [0, 1]。

        对每个因子，将原始值转换为分位数排名。
        """
        if not data:
            return data

        factor_names: set[str] = set()
        for factors in data.values():
            factor_names.update(factors.keys())

        result = {code: dict(factors) for code, factors in data.items()}

        for fname in factor_names:
            pairs = []
            for code in result:
                v = result[code].get(fname)
                if v is not None:
                    pairs.append((code, v))

            if len(pairs) < 2:
                continue

            # 按值排序
            pairs.sort(key=lambda x: x[1])
            n = Decimal(len(pairs) - 1)

            for rank, (code, _) in enumerate(pairs):
                percentile = Decimal(rank) / n if n > 0 else Decimal("0.5")
                result[code][fname] = percentile

        return result
