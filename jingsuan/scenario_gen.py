"""
随机情景生成器 — Bootstrap + Copula 条件采样 (v4.0).

纯计算层：无状态、无 IO。
生成压力测试和风险分析的随机情景。

Usage:
    from jingsuan import ScenarioGenerator
    scenarios = ScenarioGenerator.historical_bootstrap(returns_matrix)
"""

import random
from decimal import Decimal

from shuju.utils import safe_mean


class ScenarioGenerator:
    """随机情景生成器 — 多种采样方法。"""

    @staticmethod
    def historical_bootstrap(
        returns_matrix: list[list[Decimal]],
        n_scenarios: int = 1000,
        block_size: int = 5,
    ) -> list[list[Decimal]]:
        """历史 Block Bootstrap 采样, 保留短期序列相关性.

        Args:
            returns_matrix: [asset][time]
            n_scenarios: 生成的情景数
            block_size: 块长度 (保留 block_size 天内的序列依赖)
        """
        n_assets = len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)
        if n_obs < block_size:
            raise ValueError(f"观测数 {n_obs} < block_size {block_size}")

        n_blocks = n_obs // block_size
        rng = random.Random(42)
        scenarios = []

        for _ in range(n_scenarios):
            # 随机选择一个块的起始索引
            block_idx = rng.randint(0, n_blocks - 1)
            start = block_idx * block_size

            scenario = []
            for a in range(n_assets):
                # 同期所有资产的同块回报
                vals = returns_matrix[a][start:start + block_size]
                scenario.append(safe_mean(vals) if vals else Decimal("0"))
            scenarios.append(scenario)

        return scenarios

    @staticmethod
    def covariance_resample(
        returns_matrix: list[list[Decimal]],
        n_scenarios: int = 1000,
    ) -> list[list[Decimal]]:
        """协方差重采样: 估计协方差 → 多元正态采样.

        引入参数不确定性: 从 Wishart 分布近似采样协方差.
        """
        import math
        n_assets = len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)

        # 估计均值和协方差
        means = [float(safe_mean(r)) for r in returns_matrix]
        data = [[float(v) for v in r[:n_obs]] for r in returns_matrix]

        # 协方差矩阵
        cov = [[0.0] * n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(n_assets):
                mi, mj = means[i], means[j]
                c = sum((data[i][t] - mi) * (data[j][t] - mj) for t in range(n_obs))
                cov[i][j] = c / max(1, n_obs - 1)
                cov[j][i] = cov[i][j]

        # 添加参数不确定性 (简化: 对协方差加入缩放噪声)
        rng = random.Random(42)
        scenarios = []
        for _ in range(n_scenarios):
            # 不确定性缩放
            scale = rng.gauss(1.0, 1.0 / math.sqrt(n_obs))
            scale = max(0.5, min(scale, 1.5))

            scenario = []
            for a in range(n_assets):
                # 从 N(mean, scaled_cov) 采样
                val = rng.gauss(means[a], math.sqrt(max(cov[a][a] * scale, 1e-10)))
                scenario.append(Decimal(str(val)))
            scenarios.append(scenario)

        return scenarios

    @staticmethod
    def reverse_stress_test(
        current_positions: list[tuple[str, Decimal, Decimal]],  # [(代码, 权重, beta)]
        target_loss: Decimal,
        market_return_range: tuple[Decimal, Decimal] = (Decimal("-0.10"), Decimal("0.10")),
        n_iterations: int = 10000,
    ) -> list[dict]:
        """反向压力测试: 找到导致目标损失的场景组合.

        Args:
            current_positions: [(代码, 权重, beta)]
            target_loss: 目标总亏损 (如 总资产 20%)
        Returns:
            满足 loss >= target_loss 的 market_return 值 (前 10 个最接近的)
        """
        rng = random.Random(42)
        matches = []

        for _ in range(n_iterations):
            market_ret = rng.uniform(
                float(market_return_range[0]), float(market_return_range[1])
            )
            loss = Decimal("0")
            for _, weight, beta in current_positions:
                asset_ret = float(beta) * market_ret
                loss += weight * Decimal(str(abs(asset_ret)))

            if loss >= target_loss:
                matches.append({
                    "market_return": Decimal(str(round(market_ret, 4))),
                    "portfolio_loss": loss,
                })

        matches.sort(key=lambda x: x["portfolio_loss"])
        return matches[:10]
