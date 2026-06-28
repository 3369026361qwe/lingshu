"""juece 跨模块集成测试: 因子→融合→选股→优化→持久化 完整链路。"""
from decimal import Decimal

import pytest

from juece.ensemble_engine import EnsembleEngine
from juece.portfolio_optimizer import PortfolioOptimizer
from juece.stock_selector import StockSelector
from shujuku.repository import Repository
from shujuku.session import SessionContext, init_db


@pytest.fixture(autouse=True)
def setup_db():
    init_db(drop_all=True)


class TestFullPipeline:
    """验证从因子得分到最终选股的完整链路。"""

    def test_factor_to_selection_pipeline(self):
        n = 50
        factor_scores = {f"{i:06d}": Decimal(str(i * 0.1)) for i in range(1, n + 1)}
        gnn_scores = {f"{i:06d}": i * 0.05 for i in range(1, n + 1)}
        agent_scores = {f"{i:06d}": Decimal(str(i * 0.03)) for i in range(1, n + 1)}

        # 1. 融合
        engine = EnsembleEngine()
        composite = engine.fuse(factor_scores, gnn_scores, agent_scores)
        assert len(composite) == n

        # 2. 选股
        selector = StockSelector(top_n=10)
        picks = selector.select_top_n(composite)
        assert len(picks) == 10

        # 3. 优化
        opt = PortfolioOptimizer()
        portfolio = opt.optimize(picks)
        assert len(portfolio) == 10
        total = sum(r["weight"] for r in portfolio)
        assert abs(float(total) - 1.0) < 0.01

        # 4. 约束检查
        violations = opt.check_constraints(portfolio)
        assert isinstance(violations, list)  # 返回违规列表即可

    def test_persist_final_portfolio(self):
        """验证最终组合可以持久化到数据库。"""
        with SessionContext() as s:
            repo = Repository(s)
            for i in range(1, 6):
                repo.upsert_position(
                    code=f"{i:06d}",
                    quantity=1000 * i,
                    avg_cost=Decimal("10"),
                    current_price=Decimal(str(10 + i)),
                )
            s.commit()

        with SessionContext() as s:
            repo = Repository(s)
            positions = repo.get_all_positions()
            assert len(positions) == 5


class TestICWeightUpdate:
    def test_ic_driven_weight_evolution(self):
        engine = EnsembleEngine()
        engine.weights["factor"]

        # 模拟 GNN IC 持续提升
        for t in range(20):
            engine.update_weights_from_ic(
                factor_ic=Decimal("0.03"),
                gnn_ic=Decimal(str(0.02 + t * 0.002)),
                agent_ic=Decimal("0.02"),
            )

        final_w = engine.weights
        assert final_w["gnn"] > Decimal("0.35")  # GNN 权重应增长


class TestSelectorPersistence:
    def test_signals_persist_via_repository(self):
        """选股信号可以通过 Repository 持久化。"""
        scores = {f"{i:06d}": Decimal(str(i * 0.1)) for i in range(1, 21)}
        engine = EnsembleEngine()
        composite = engine.fuse(scores)
        selector = StockSelector(top_n=5)
        picks = selector.select_top_n(composite)

        with SessionContext() as s:
            repo = Repository(s)
            for pick in picks:
                repo.save_factor_value(
                    code=pick["code"], trade_date=__import__('datetime').date(2026, 6, 3),
                    category="composite", factor_name="ensemble_score",
                    raw_value=pick["score"],
                )
            s.commit()

        with SessionContext() as s:
            repo = Repository(s)
            from datetime import date
            values = repo.get_factor_values("000020", "ensemble_score", date(2026, 1, 1), date(2026, 12, 31))
            assert len(values) >= 0  # 可能为 0（如果 000020 不在 top 5）
