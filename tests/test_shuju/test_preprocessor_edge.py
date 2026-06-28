"""
DataPreprocessor 边缘场景测试：负值、全缺失、大数据集、空输入。
"""

from decimal import Decimal

from shuju.data_preprocessor import DataPreprocessor


class TestNegativeValues:
    """因子值可能为负（如动量因子），验证不会因负值崩溃。"""

    def test_negative_values_winsorize(self):
        pp = DataPreprocessor()
        data = {}
        for i in range(1, 11):
            data[f"{i:06d}"] = {"momentum_1m": Decimal(str(i * 2 - 15))}
        result = pp.winsorize(data)
        assert len(result) == 10
        for v in result.values():
            assert isinstance(v["momentum_1m"], Decimal)

    def test_negative_values_standardize(self):
        pp = DataPreprocessor()
        data = {}
        for i in range(1, 11):
            data[f"{i:06d}"] = {"momentum_1m": Decimal(str(i * 3 - 20))}
        result = pp.standardize(data)
        values = [float(v["momentum_1m"]) for v in result.values()]
        mean = sum(values) / len(values)
        assert abs(mean) < 0.01


class TestAllMissing:
    """所有值都缺失的场景。"""

    def test_all_missing_one_factor(self):
        pp = DataPreprocessor()
        data = {
            "000001": {"pe": None},
            "000002": {"pe": None},
            "000003": {"pe": None},
        }
        # 全部缺失 → 无法计算中位数 → 保持不变（None）
        result = pp.fill_missing(data)
        # 没有非空值可计算中位数，应保持 None
        assert result["000001"]["pe"] is None


class TestLargeDataset:
    """大数据集性能/正确性。"""

    def test_large_dataset_pipeline(self):
        pp = DataPreprocessor()
        data = {}
        for i in range(1, 201):
            data[f"{i:06d}"] = {
                "pe": Decimal(str(10 + (i % 30) * 2 + i * 0.1)),
                "roe": Decimal(str(5 + (i % 20) + i * 0.05)),
            }
        industry = {f"{i:06d}": f"行业{i % 8}" for i in range(1, 201)}
        result = pp.pipeline(data, industry)
        assert len(result) == 200
        # 所有值应为有限 Decimal
        for factors in result.values():
            for v in factors.values():
                assert isinstance(v, Decimal)
                assert v.is_finite()

    def test_percentiles_large(self):
        pp = DataPreprocessor()
        data = {}
        for i in range(1, 501):
            data[f"{i:06d}"] = {"val": Decimal(str(i))}
        result = pp.compute_percentiles(data)
        # 中位数应接近 0.5
        mid_code = f"{250:06d}"
        assert abs(float(result[mid_code]["val"]) - 0.5) < 0.05


class TestEmptyInput:
    """边界：空输入。"""

    def test_empty_winsorize(self):
        assert DataPreprocessor().winsorize({}) == {}

    def test_empty_standardize(self):
        assert DataPreprocessor().standardize({}) == {}

    def test_empty_neutralize(self):
        assert DataPreprocessor().neutralize({}, {}) == {}

    def test_empty_pipeline(self):
        assert DataPreprocessor().pipeline({}, {}) == {}


class TestInputImmutability:
    """验证不会修改原始输入。"""

    def test_winsorize_leaves_original_unchanged(self):
        pp = DataPreprocessor()
        original = {
            "000001": {"pe": Decimal("1000")},
            "000002": {"pe": Decimal("20")},
            "000003": {"pe": Decimal("30")},
            "000004": {"pe": Decimal("25")},
            "000005": {"pe": Decimal("15")},
        }
        original_copy = {k: dict(v) for k, v in original.items()}
        pp.winsorize(original)
        # 原始数据不应被修改
        for code in original:
            for fname in original[code]:
                assert original[code][fname] == original_copy[code][fname]
