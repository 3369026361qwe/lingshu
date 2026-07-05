"""
数据窥探防御 (v4.0).

防止过拟合: Deflated Sharpe Ratio, Haircut Sharpe, Probabilistic Sharpe Ratio.
Harvey & Liu (2014) + Bailey & López de Prado (2014, 2017).

Usage:
    from huice.data_snooping import DataSnoopingDefender
    dsr = DataSnoopingDefender.deflated_sharpe(sharpe, n_trials, n_obs)
"""

import math


class DataSnoopingDefender:
    """数据窥探防御 — DSR / Haircut / PSR。"""

    @staticmethod
    def deflated_sharpe_ratio(
        sharpe_observed: float,
        n_trials: int,
        n_observations: int,
        sharpe_expected: float = 0.0,
    ) -> float:
        """Deflated Sharpe Ratio (Harvey & Liu 2014).

        DSR = Prob[max SR > SR_obs | N trials].
        返回 p-value: 值越小, SR 越不可能来自数据窥探.

        Args:
            sharpe_observed: 策略的样本内夏普比率
            n_trials: 尝试的策略/参数组合数
            n_observations: 观测数 (交易日)
            sharpe_expected: 零假设下的期望夏普 (默认 0)
        """
        if n_observations < 2:
            return 0.5

        # SR 标准误: SE(SR) ≈ sqrt((1 + SR²/2) / T)
        se = math.sqrt((1 + 0.5 * sharpe_observed ** 2) / n_observations)

        # E[max SR | N trials] ≈ se · sqrt(2 · log(N))
        if n_trials > 1:
            expected_max = se * math.sqrt(2 * math.log(n_trials))
        else:
            expected_max = 0.0

        # DSR: 对多次试验进行调整后的 SR
        dsr_value = (sharpe_observed - sharpe_expected - expected_max) / se

        # 转为 p-value (单侧)
        return 1.0 - DataSnoopingDefender._normal_cdf(dsr_value)

    @staticmethod
    def haircut_sharpe(
        sharpe_observed: float,
        n_trials: int,
        n_observations: int,
    ) -> float:
        """Haircut Sharpe Ratio.

        SR_haircut = SR_obs - E[max SR | N trials].
        扣减多次试验带来的过拟合偏差.
        """
        if n_observations < 2:
            return 0.0

        se = math.sqrt((1 + 0.5 * sharpe_observed ** 2) / n_observations)
        haircut = se * math.sqrt(2 * math.log(max(n_trials, 1)))
        return max(sharpe_observed - haircut, 0.0)

    @staticmethod
    def probabilistic_sharpe_ratio(
        sharpe_observed: float,
        sharpe_benchmark: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Probabilistic Sharpe Ratio (Bailey & López de Prado).

        PSR = Prob[SR > SR_benchmark].
        考虑非正态性 (偏度, 峰度).

        Returns:
            PSR 概率 [0, 1] — > 0.95 表示显著优于基准.
        """
        if n_observations < 3:
            return 0.5

        sr_diff = sharpe_observed - sharpe_benchmark
        # PSR 标准误 (含偏度/峰度修正)
        se_psr = math.sqrt(
            (1 - skewness * sr_diff + (kurtosis - 1) / 4 * sr_diff ** 2)
            / n_observations
        )

        if se_psr <= 0:
            return 1.0 if sr_diff > 0 else 0.0

        z_score = sr_diff / se_psr
        return DataSnoopingDefender._normal_cdf(z_score)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """标准正态 CDF (Abramowitz & Stegun 近似)."""
        if x < -8:
            return 0.0
        if x > 8:
            return 1.0

        # erf 近似
        def _erf(z: float) -> float:
            t = 1.0 / (1.0 + 0.3275911 * abs(z))
            poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
            return 1.0 - poly * math.exp(-z * z)

        return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
