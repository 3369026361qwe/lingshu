"""
卡尔曼滤波动态权重估计。

状态空间模型:
    观测方程: r_t = F_t · β_t + ε_t   (收益 = 因子暴露 × 时变权重 + 噪声)
    状态方程: β_t = β_{t-1} + η_t     (权重遵循随机游走)
    其中 ε_t ~ N(0, R), η_t ~ N(0, Q)

预测-更新两步迭代:
    预测: β_pred = β_prev,  P_pred = P_prev + Q
    更新: K = P_pred @ F^T / (F @ P_pred @ F^T + R)
          β_new = β_pred + K * (r - F @ β_pred)
          P_new = (I - K @ F) @ P_pred

Usage:
    kf = KalmanWeightEstimator(n_factors=5)
    kf.update(factor_exposures, realized_return)
    weights = kf.weights  # 当前因子权重向量
"""

import logging
from decimal import Decimal
from typing import Optional

import logging

from yinzi.metrics import kalman_update_total, kalman_converged, kalman_max_variance

_logger = logging.getLogger(__name__)


class KalmanWeightEstimator:
    """卡尔曼滤波时变因子权重估计器。

    每个交易日输入因子暴露向量和实现收益，输出更新后的因子权重。
    """

    def __init__(
        self,
        n_factors: int,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
    ) -> None:
        """
        Args:
            n_factors: 因子数量
            process_noise: 过程噪声 Q (标量，默认 1e-4)
            measurement_noise: 观测噪声 R (标量，默认 1e-2)
        """
        self.n_factors = n_factors
        self.Q = Decimal(str(process_noise))   # 过程噪声
        self.R = Decimal(str(measurement_noise))  # 观测噪声

        # 状态向量: 因子权重 (初始化为等权)
        self._weights: list[Decimal] = [Decimal("1") / Decimal(str(n_factors)) for _ in range(n_factors)]

        # 状态协方差矩阵 (初始化为单位阵)
        self._P: list[list[Decimal]] = [
            [Decimal("1") if i == j else Decimal("0") for j in range(n_factors)]
            for i in range(n_factors)
        ]

        self._initialized = False
        self._update_count = 0

    @property
    def weights(self) -> list[Decimal]:
        """当前因子权重向量。"""
        return list(self._weights)

    @property
    def variances(self) -> list[Decimal]:
        """当前权重方差 (对角线)。"""
        return [self._P[i][i] for i in range(self.n_factors)]

    @property
    def update_count(self) -> int:
        return self._update_count

    def update(
        self,
        factor_exposures: list[Decimal],
        realized_return: Decimal,
    ) -> list[Decimal]:
        """执行一步卡尔曼滤波更新。

        Args:
            factor_exposures: 因子暴露向量 F_t [n_factors]
            realized_return: 实现收益 r_t

        Returns:
            更新后的权重向量
        """
        if len(factor_exposures) != self.n_factors:
            raise ValueError(
                f"Expected {self.n_factors} factor exposures, got {len(factor_exposures)}"
            )

        F = factor_exposures
        r = realized_return

        # ── 预测步骤 ──────────────────────────────────
        # β_pred = β_prev
        beta_pred = list(self._weights)

        # P_pred = P_prev + Q·I
        P_pred = [
            [self._P[i][j] + (self.Q if i == j else Decimal("0"))
             for j in range(self.n_factors)]
            for i in range(self.n_factors)
        ]

        # ── 更新步骤 ──────────────────────────────────
        # 计算 F @ P_pred @ F^T (标量)
        FPF = self._quadratic_form(P_pred, F)

        # 计算 S = FPF + R (新息协方差)
        S = FPF + self.R
        if S == 0:
            _logger.warning("Innovation covariance is 0, skipping update")
            return list(self._weights)

        # 计算卡尔曼增益 K = P_pred @ F / S
        K = [
            sum(P_pred[i][j] * F[j] for j in range(self.n_factors)) / S
            for i in range(self.n_factors)
        ]

        # 计算新息 ν = r - F @ β_pred
        innovation = r - sum(F[i] * beta_pred[i] for i in range(self.n_factors))

        # 更新权重: β_new = β_pred + K * ν
        self._weights = [beta_pred[i] + K[i] * innovation for i in range(self.n_factors)]

        # 更新协方差: P_new = (I - K @ F^T) @ P_pred
        # 即 P_new[i][j] = P_pred[i][j] - K[i] * sum(F[k] * P_pred[k][j])
        self._P = [
            [
                P_pred[i][j] - K[i] * sum(F[k] * P_pred[k][j] for k in range(self.n_factors))
                for j in range(self.n_factors)
            ]
            for i in range(self.n_factors)
        ]

        self._update_count += 1
        if not self._initialized and self._update_count >= 10:
            self._initialized = True

        # 上报 Prometheus 指标
        kalman_update_total.inc()
        kalman_converged.set(1 if self.is_converged() else 0)
        kalman_max_variance.set(float(max(self.variances)))

        return list(self._weights)

    def batch_update(
        self,
        factor_matrix: list[list[Decimal]],  # [T x n_factors]
        returns: list[Decimal],               # [T]
    ) -> list[Decimal]:
        """批量更新（按时间序列顺序）。"""
        for t in range(len(returns)):
            self.update(factor_matrix[t], returns[t])
        return list(self._weights)

    def is_converged(self, tolerance: Optional[Decimal] = None) -> bool:
        """判断权重是否收敛 (方差下降到过程噪声的 10 倍以内)。"""
        if self._update_count < 20:
            return False
        if tolerance is None:
            tolerance = self.Q * Decimal("10")
        max_var = max(self.variances)
        return max_var < tolerance

    @staticmethod
    def _quadratic_form(P: list[list[Decimal]], F: list[Decimal]) -> Decimal:
        """计算 F^T @ P @ F (二次型)。"""
        n = len(F)
        result = Decimal("0")
        for i in range(n):
            row_sum = sum(P[i][j] * F[j] for j in range(n))
            result += F[i] * row_sum
        return result
