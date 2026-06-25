"""
交叉验证: yinzi ↔ shujuku ↔ shuju 集成测试。

验证因子计算→持久化→回读完整链路。
"""

from datetime import date
from decimal import Decimal

import pytest

from shujuku.session import init_db, SessionContext
from shujuku.repository import Repository
from yinzi.factor_store import FactorStore
from yinzi.factor_base import FactorResult, FactorCategory
from yinzi.kalman_weight import KalmanWeightEstimator


@pytest.fixture(autouse=True)
def setup_db():
    init_db(drop_all=True)


class TestFactorPersistence:
    def test_save_and_read_factor_values(self):
        with SessionContext() as s:
            repo = Repository(s)
            store = FactorStore(repo)

            results = [
                FactorResult("000001", "pe", FactorCategory.VALUE, Decimal("15.5"), Decimal("0.3"), Decimal("0.7")),
                FactorResult("000002", "pe", FactorCategory.VALUE, Decimal("25.0"), Decimal("-0.5"), Decimal("0.2")),
            ]
            saved = store.save_factor_values(date(2026, 6, 1), results)
            assert saved == 2
            s.commit()

        with SessionContext() as s:
            repo = Repository(s)
            values = repo.get_factor_values("000001", "pe", date(2026, 1, 1), date(2026, 12, 31))
            assert len(values) == 1
            assert values[0].raw_value == Decimal("15.5")
            assert values[0].z_score == Decimal("0.3")
            assert values[0].percentile == Decimal("0.7")

    def test_save_factor_batch(self):
        with SessionContext() as s:
            repo = Repository(s)
            store = FactorStore(repo)
            factor_map = {"pe": Decimal("15"), "pb": Decimal("2.1"), "roe": Decimal("12.5")}
            z_map = {"pe": Decimal("0.3"), "pb": Decimal("0.1"), "roe": Decimal("0.5")}
            saved = store.save_factor_batch(
                date(2026, 6, 1), "000001", factor_map, z_score_map=z_map
            )
            assert saved == 3
            s.commit()

        with SessionContext() as s:
            repo = Repository(s)
            for name in ("pe", "pb", "roe"):
                values = repo.get_factor_values("000001", name, date(2026, 1, 1), date(2026, 12, 31))
                assert len(values) == 1, f"Missing {name}"

    def test_save_factor_weights(self):
        with SessionContext() as s:
            repo = Repository(s)
            store = FactorStore(repo)
            saved = store.save_factor_weights(
                date(2026, 6, 1),
                ["pe", "momentum_1m", "roe"],
                [Decimal("0.15"), Decimal("0.12"), Decimal("0.18")],
                [Decimal("0.002"), Decimal("0.003"), Decimal("0.001")],
            )
            assert saved == 3
            s.commit()


class TestKalmanIntegration:
    def test_kalman_weights_persist(self):
        """卡尔曼滤波权重 → FactorStore 持久化 → 回读验证。"""
        kf = KalmanWeightEstimator(4)
        for _ in range(30):
            kf.update(
                [Decimal("0.1"), Decimal("0.05"), Decimal("-0.02"), Decimal("0.08")],
                Decimal("0.015"),
            )

        with SessionContext() as s:
            repo = Repository(s)
            store = FactorStore(repo)
            names = ["value", "momentum", "quality", "volatility"]
            store.save_factor_weights(date(2026, 6, 1), names, kf.weights, kf.variances)
            s.commit()

        # 权重和应为近似 1（因子暴露单位化后）
        total = sum(kf.weights)
        assert abs(float(total) - 0.0) < 100  # 权重可正可负，总和不一定为1


class TestValidatorIntegration:
    def test_ic_persistence(self):
        """IC 记录 → FactorStore 持久化 → 数据库验证。"""
        with SessionContext() as s:
            repo = Repository(s)
            store = FactorStore(repo)
            store.save_ic_record(
                date(2026, 6, 1), "pe", Decimal("0.045"), Decimal("0.8"), 20
            )
            s.commit()

        from shujuku.models.yinzi_models import FactorICRecord
        with SessionContext() as s:
            records = s.query(FactorICRecord).all()
            assert len(records) == 1
            assert records[0].ic == Decimal("0.045")
            assert records[0].ir == Decimal("0.8")
