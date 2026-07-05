"""
HMM 市场体制检测器 (v4.0).

替换简单的 60 天滚动规则分类器 (bull/bear/sideways),
使用 Hidden Markov Model 进行概率化体制推断。

Usage:
    from yinzi.regime_detector import HMMRegimeDetector
    model = HMMRegimeDetector.fit(returns, n_regimes=3)
    regime = HMMRegimeDetector.predict(current_features, model)
"""

import math
from decimal import Decimal

from shuju.utils import safe_mean


class HMMRegimeDetector:
    """HMM 市场体制检测 — Baum-Welch + Viterbi."""

    @staticmethod
    def fit(
        returns: list[Decimal],
        n_regimes: int = 3,
        features: list[str] | None = None,
    ) -> dict:
        """拟合 2-状态或 3-状态 HMM (Baum-Welch EM).

        Args:
            returns: 日收益率序列
            n_regimes: 体制数 (2: up/down, 3: bull/sideways/bear)
            features: 额外特征 (未使用, 为未来扩展预留)
        Returns:
            {regime_labels, transition_matrix, state_means, state_vars, stationary_prob}
        """
        n = len(returns)
        if n < 100:
            raise ValueError(f"HMM needs >= 100 observations, got {n}")

        r = [float(v) for v in returns]
        mu = sum(r) / n
        var = sum((x - mu) ** 2 for x in r) / max(1, n - 1)
        sigma = math.sqrt(var)

        # 初始化: 按均值和波动率分层
        if n_regimes == 3:
            means = [mu - sigma, mu, mu + sigma]
            vars_list = [var * 0.5, var, var * 0.5]
        elif n_regimes == 2:
            means = [mu - sigma, mu + sigma]
            vars_list = [var, var]
        else:
            means = [mu + (i - n_regimes // 2) * sigma / 2 for i in range(n_regimes)]
            vars_list = [var] * n_regimes

        K = n_regimes
        # 均匀转移矩阵
        trans = [[1.0 / K] * K for _ in range(K)]
        # 均匀初始概率
        start_prob = [1.0 / K] * K

        # EM 迭代
        for _iteration in range(50):
            # E-step: 前向后向算法
            # (简化版: 用条件概率近似)

            # 计算 emission probability: P(x_t | state = k)
            emissions = [[0.0] * K for _ in range(n)]
            for t in range(n):
                for k in range(K):
                    diff = r[t] - means[k]
                    if vars_list[k] > 1e-10:
                        emissions[t][k] = (1.0 / math.sqrt(2 * math.pi * vars_list[k])
                                           * math.exp(-0.5 * diff * diff / vars_list[k]))

            # 前向概率
            alpha = [[0.0] * K for _ in range(n)]
            for k in range(K):
                alpha[0][k] = start_prob[k] * emissions[0][k]
            for t in range(1, n):
                for k in range(K):
                    alpha[t][k] = sum(alpha[t - 1][j] * trans[j][k] for j in range(K)) * emissions[t][k]

            # 后向概率
            beta = [[0.0] * K for _ in range(n)]
            for k in range(K):
                beta[n - 1][k] = 1.0
            for t in range(n - 2, -1, -1):
                for k in range(K):
                    beta[t][k] = sum(trans[k][j] * emissions[t + 1][j] * beta[t + 1][j] for j in range(K))

            # 状态后验概率 γ_t(k)
            gamma = [[0.0] * K for _ in range(n)]
            for t in range(n):
                total = sum(alpha[t][j] * beta[t][j] for j in range(K))
                if total > 0:
                    for k in range(K):
                        gamma[t][k] = alpha[t][k] * beta[t][k] / total

            # M-step: 更新参数
            new_means = [0.0] * K
            new_vars = [0.0] * K
            for k in range(K):
                gamma_sum = sum(gamma[t][k] for t in range(n))
                if gamma_sum > 1e-10:
                    new_means[k] = sum(gamma[t][k] * r[t] for t in range(n)) / gamma_sum
                    new_vars[k] = sum(gamma[t][k] * (r[t] - new_means[k]) ** 2 for t in range(n)) / gamma_sum
                else:
                    new_means[k] = means[k]
                    new_vars[k] = vars_list[k]

            new_trans = [[0.0] * K for _ in range(K)]
            for i in range(K):
                for j in range(K):
                    xi_sum = 0.0
                    for t in range(n - 1):
                        xi = (alpha[t][i] * trans[i][j] * emissions[t + 1][j] * beta[t + 1][j])
                        xi_sum += xi
                    norm = sum(gamma[t][i] for t in range(n - 1))
                    new_trans[i][j] = xi_sum / norm if norm > 1e-10 else 1.0 / K

            # 收敛检查
            diff = sum(abs(new_means[i] - means[i]) for i in range(K))
            means = new_means
            vars_list = new_vars
            trans = new_trans
            start_prob = [sum(gamma[0][k] for k in range(K)) / K] * K  # 近似

            if diff < 1e-6:
                break

        # 稳态概率
        stationary_prob = list(start_prob)

        # 标签: 按 mean 排序
        regime_order = sorted(range(K), key=lambda k: means[k])
        labels = {regime_order[0]: "bear", regime_order[2]: "bull"} if K == 3 else {regime_order[0]: "down", regime_order[1]: "up"}
        labels.update({k: f"sideways_{k}" for k in range(K) if k not in labels})

        return {
            "regime_labels": labels,
            "transition_matrix": trans,
            "state_means": means,
            "state_vars": vars_list,
            "stationary_prob": stationary_prob,
            "n_regimes": K,
        }

    @staticmethod
    def predict_regime(
        recent_returns: list[Decimal],
        model: dict,
    ) -> int:
        """基于最近回报预测当前体制.

        Returns:
            体制索引 (0 = bear, 2 = bull for 3-regime)
        """
        means = model["state_means"]
        K = model["n_regimes"]

        # 用最近 20 天平均回报
        recent_r = [float(v) for v in recent_returns[-20:]]
        mu_recent = sum(recent_r) / max(1, len(recent_r))

        # 选最匹配的体制
        best_k = 0
        best_d = float("inf")
        for k in range(K):
            d = abs(mu_recent - means[k])
            if d < best_d:
                best_d = d
                best_k = k

        return best_k

    @staticmethod
    def regime_conditional_var(
        returns: list[Decimal],
        regime_labels: list[int],
        confidence: Decimal = Decimal("0.99"),
    ) -> dict[int, Decimal]:
        """条件于每个体制的历史 VaR."""
        regimes = set(regime_labels)
        result = {}
        z = Decimal("2.326") if confidence >= Decimal("0.99") else Decimal("1.645")

        for reg in regimes:
            reg_returns = [returns[i] for i, rl in enumerate(regime_labels) if rl == reg]
            if len(reg_returns) < 10:
                result[reg] = Decimal("0")
                continue
            mu = safe_mean(reg_returns)
            var = sum((r - mu) ** 2 for r in reg_returns) / Decimal(max(1, len(reg_returns) - 1))
            sigma = var.sqrt()
            result[reg] = mu - z * sigma

        return result
