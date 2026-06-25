"""
测试 DataPreprocessor 预处理管道: 去极值 → 填充 → 标准化 → 中性化。
"""

from decimal import Decimal

import pytest

from shuju.data_preprocessor import DataPreprocessor


def _make_data(*pairs):
    """构造 {code: {factor: value}} 格式的测试数据。"""
    result = {}
    for code, pe, roe in pairs:
        result[code] = {"pe": Decimal(str(pe)), "roe": Decimal(str(roe))}
    return result


class TestWinsorize:
    def test_no_outliers(self,):
        pp = DataPreprocessor(method="sigma", n_sigma=Decimal("3"))
        data = _make_data(
            ("000001", 15, 12), ("000002", 18, 10), ("000003", 12, 15),
            ("000004", 20, 8), ("000005", 16, 11),
        )
        result = pp.winsorize(data)
        # 无极端值时应保持不变
        for code in data:
            assert result[code]["pe"] == data[code]["pe"]

    def test_extreme_high_clipped(self):
        pp = DataPreprocessor(method="sigma", n_sigma=Decimal("2"))
        data = _make_data(
            ("000001", 15, 12), ("000002", 18, 12), ("000003", 12, 12),
            ("000004", 20, 12), ("000005", 16, 12),
            ("000006", 500, 12),  # 极端高值
        )
        result = pp.winsorize(data)
        assert result["000006"]["pe"] < Decimal("500")

    def test_extreme_low_clipped(self):
        pp = DataPreprocessor(method="sigma", n_sigma=Decimal("2"))
        data = _make_data(
            ("000001", 15, 12), ("000002", 18, 12), ("000003", 12, 12),
            ("000004", 20, 12), ("000005", 16, 12),
            ("000006", -500, 12),  # 极端低值
        )
        result = pp.winsorize(data)
        assert result["000006"]["pe"] > Decimal("-500")

    def test_mad_method(self):
        pp = DataPreprocessor(method="mad", n_sigma=Decimal("3"))
        data = _make_data(*[(f"{i:06d}", i * 10, i) for i in range(1, 21)])
        result = pp.winsorize(data)
        assert len(result) == 20

    def test_small_sample_no_change(self):
        pp = DataPreprocessor()
        data = _make_data(("000001", 15, 12), ("000002", 18, 10))
        result = pp.winsorize(data)
        # 样本太小 (<5)，不做处理
        for code in data:
            assert result[code]["pe"] == data[code]["pe"]


class TestFillMissing:
    def test_fill_median(self):
        pp = DataPreprocessor(fill_method="median")
        data = {
            "000001": {"pe": Decimal("10")},
            "000002": {"pe": Decimal("20")},
            "000003": {"pe": Decimal("30")},
            "000004": {},  # 缺失
        }
        result = pp.fill_missing(data)
        assert result["000004"]["pe"] == Decimal("20")  # median of [10,20,30]

    def test_fill_zero(self):
        pp = DataPreprocessor(fill_method="zero")
        data = {
            "000001": {"pe": Decimal("10")},
            "000002": {},
        }
        result = pp.fill_missing(data)
        assert result["000002"]["pe"] == Decimal("0")

    def test_no_data_unchanged(self):
        pp = DataPreprocessor()
        assert pp.fill_missing({}) == {}


class TestStandardize:
    def test_zscore_mean_zero(self):
        pp = DataPreprocessor()
        data = _make_data(
            ("000001", 10, 5), ("000002", 20, 5), ("000003", 30, 5),
            ("000004", 40, 5), ("000005", 50, 5),
        )
        result = pp.standardize(data)
        # Z-Score 均值为 0
        pe_values = [result[c]["pe"] for c in result]
        mean_pe = sum(pe_values) / len(pe_values)
        assert abs(float(mean_pe)) < 0.01

    def test_zscore_std_one(self):
        pp = DataPreprocessor()
        data = _make_data(
            ("000001", 10, 5), ("000002", 20, 5), ("000003", 30, 5),
            ("000004", 40, 5), ("000005", 50, 5),
        )
        result = pp.standardize(data)
        pe_values = [float(result[c]["pe"]) for c in result]
        mean_pe = sum(pe_values) / len(pe_values)
        variance = sum((v - mean_pe) ** 2 for v in pe_values) / len(pe_values)
        assert abs(variance - 1.0) < 0.01

    def test_constant_factor_zeroed(self):
        pp = DataPreprocessor()
        data = _make_data(
            ("000001", 10, 100), ("000002", 20, 100), ("000003", 30, 100),
            ("000004", 40, 100), ("000005", 50, 100), ("000006", 60, 100),
        )
        result = pp.standardize(data)
        # roe 全部相同 → 标准差为 0 → 全部置 0
        for code in result:
            assert result[code]["roe"] == Decimal("0")


class TestNeutralize:
    def test_industry_neutralize(self):
        pp = DataPreprocessor()
        data = _make_data(
            ("000001", 15, 12), ("000002", 25, 12),
            ("000003", 10, 12), ("000004", 20, 12),
        )
        industry = {"000001": "银行", "000002": "银行", "000003": "科技", "000004": "科技"}
        result = pp.neutralize(data, industry)
        # 行业内均值应接近 0
        bank_pe = [float(result[c]["pe"]) for c in ("000001", "000002")]
        assert abs(sum(bank_pe)) < 0.01

    def test_no_industry_map(self):
        pp = DataPreprocessor()
        data = _make_data(("000001", 15, 12))
        result = pp.neutralize(data, {})
        assert result["000001"]["pe"] == data["000001"]["pe"]


class TestPipeline:
    def test_end_to_end(self):
        pp = DataPreprocessor()
        data = {
            f"{i:06d}": {
                "pe": Decimal(str(10 + i * 2 + (i % 5) * 3)),
                "roe": Decimal(str(5 + i + (i % 3) * 2)),
            }
            for i in range(1, 31)
        }
        industry = {f"{i:06d}": f"行业{i % 5}" for i in range(1, 31)}
        result = pp.pipeline(data, industry)
        assert len(result) == 30
        # 所有值应为 Decimal
        for code, factors in result.items():
            for v in factors.values():
                assert isinstance(v, Decimal)

    def test_percentiles(self):
        pp = DataPreprocessor()
        data = _make_data(*[(f"{i:06d}", i * 10, i) for i in range(1, 101)])
        result = pp.compute_percentiles(data)
        # 最小值应接近 0，最大值应接近 1
        pe_vals = [(c, float(result[c]["pe"])) for c in result]
        pe_vals.sort(key=lambda x: x[1])
        assert pe_vals[0][1] < 0.05
        assert pe_vals[-1][1] > 0.95
