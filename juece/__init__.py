"""
juece — 集成决策引擎

三路信号加权融合 + 选股信号生成 + Black-Litterman 组合优化 + 调仓计算 + 基准比较。

最终得分 = w1 × 多因子得分 + w2 × GNN增强得分 + w3 × Agent综合评分
(w1, w2, w3 由历史 IC 表现动态调整)
"""

from juece.benchmark import Benchmark
from juece.ensemble_engine import EnsembleEngine
from juece.factor_fusion import FactorFusion
from juece.portfolio_optimizer import PortfolioOptimizer
from juece.rebalancer import Rebalancer
from juece.stock_selector import StockSelector

__all__ = [
    "EnsembleEngine", "StockSelector", "FactorFusion", "PortfolioOptimizer",
    "Rebalancer", "Benchmark",
]
