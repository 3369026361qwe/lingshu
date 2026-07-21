"""
准备金方法引擎 — Chain Ladder + Bornhuetter-Ferguson (v4.0).

纯计算层：无状态、无 IO。
用于精算风险预留，支持损失三角形 (Loss Triangle) 分析。

数学基础:
    Chain Ladder (链梯法):
        累积损失 C_{i,j}: i = 事故年, j = 进展年
        进展因子: f_j = Σ_{i=1}^{n-j} C_{i,j+1} / Σ_{i=1}^{n-j} C_{i,j}
        最终损失: C_{i,n} = C_{i,n-i+1} · Π_{k=n-i+1}^{n-1} f_k

    Bornhuetter-Ferguson (BF法):
        最终损失 = 累计已发生 + 预期未发生
        其中 未发生 = 先验最终损失 × (1 - 1/Π_{k=j}^{n-1} f_k)

        需要先验最终损失估计 (通常来自 EVT / 风险预算结果)

    准备金 = 最终损失估计 - 已支付损失

Usage:
    from jingsuan.reserving import ReservingEngine, LossTriangle
    triangle = LossTriangle.from_matrix(data)
    ult = ReservingEngine.chain_ladder(triangle)
    reserve = ReservingEngine.bornhuetter_ferguson(triangle, prior_ultimate)
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class LossTriangle:
    """损失三角形 (累积损失).

    上三角结构: C_{i,j} where i + j <= n + 1
    为下三角填充 None.
    """
    data: list[list[Decimal | None]]  # [accident_year][development_year]
    n_accident_years: int
    n_development_years: int

    @classmethod
    def from_matrix(cls, cumulative_losses: list[list[Decimal]]) -> "LossTriangle":
        """从矩形矩阵构建损失三角形 (右上方截断).

        标准损失三角形: C_{i,j}  where i+j <= max_period.
        对于 n 个事故年, 每个事故年的进展年数 = n - i.
        """
        n = len(cumulative_losses)
        m = max(len(row) for row in cumulative_losses) if cumulative_losses else 0
        # Upper-triangle bound: i + j < n (standard square loss triangle)
        # For rectangular data, use min(n, m) as the truncation bound
        tri_bound = n  # square triangle: n accident years, n development years
        data = []
        for i in range(n):
            row = []
            for j in range(m):
                if i + j < tri_bound:  # 上三角: 已知数据
                    row.append(cumulative_losses[i][j] if j < len(cumulative_losses[i]) else None)
                else:
                    row.append(None)  # 下三角 (待预测)
            data.append(row)
        return cls(data=data, n_accident_years=n, n_development_years=m)

    @classmethod
    def from_incremental(cls, incremental_losses: list[list[Decimal]]) -> "LossTriangle":
        """从增量损失矩阵构建累积损失三角形."""
        cumulative = []
        for row in incremental_losses:
            cum_row = []
            running = Decimal("0")
            for val in row:
                running += val
                cum_row.append(running)
            cumulative.append(cum_row)
        return cls.from_matrix(cumulative)

    @property
    def latest_diagonal(self) -> list[Decimal]:
        """最新对角线 (最近评估日期的已知累积损失)."""
        result = []
        n = self.n_accident_years
        for i in range(n):
            j = n - 1 - i
            if j >= 0 and j < len(self.data[i]) and self.data[i][j] is not None:
                result.append(self.data[i][j])  # type: ignore[arg-type]
            else:
                result.append(Decimal("0"))
        return result


@dataclass
class ReservingResult:
    """准备金计算结果."""
    method: str                          # "chain_ladder" | "bornhuetter_ferguson"
    development_factors: list[Decimal]    # 进展因子 f_j
    cumulative_factors: list[Decimal]     # 累积进展因子 Π f_k
    ultimate_losses: list[Decimal]        # 每个事故年的最终损失
    reserves: list[Decimal]               # 每个事故年的准备金
    total_ultimate: Decimal               # 总最终损失
    total_reserve: Decimal                # 总准备金
    coefficient_of_variation: Decimal     # 变异系数 (不确定性度量)
    # BF-specific
    prior_ultimate: list[Decimal] | None = None
    credibility_factor: Decimal = Decimal("1")


class ReservingEngine:
    """准备金方法引擎 — Chain Ladder / Bornhuetter-Ferguson."""

    # ═══════════════════════════════════════════════════════
    # Chain Ladder
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def chain_ladder(
        triangle: LossTriangle,
        tail_factor: Decimal = Decimal("1.0"),
    ) -> ReservingResult:
        """Chain Ladder (链梯法) 准备金估计.

        Args:
            triangle: 损失三角形 (累积)
            tail_factor: 尾部因子 (>1 表示进展超过最后开发年仍有损失)

        Returns:
            ReservingResult
        """
        n = triangle.n_accident_years
        m = triangle.n_development_years

        # 进展因子 f_j (j = 0..m-2)
        dev_factors = []
        for j in range(m - 1):
            num = Decimal("0")
            den = Decimal("0")
            for i in range(n - j - 1):
                c_j = triangle.data[i][j]
                c_j1 = triangle.data[i][j + 1]
                if c_j is not None and c_j1 is not None and c_j > 0:
                    num += c_j1
                    den += c_j
            if den > 0:
                dev_factors.append(safe_divide(num, den, Decimal("1")))
            else:
                dev_factors.append(Decimal("1"))

        # 添加尾部因子
        dev_factors.append(tail_factor)

        # 累积进展因子
        cum_factors = [Decimal("1")] * m
        for j in range(m - 1, 0, -1):
            cum_factors[j - 1] = cum_factors[j] * dev_factors[j - 1]

        # 最终损失
        ultimate = []
        reserves = []
        for i in range(n):
            # 最新已知累积损失 (对角线)
            j_latest = n - 1 - i
            if j_latest < 0 or j_latest >= len(triangle.data[i]):
                ultimate.append(Decimal("0"))
                reserves.append(Decimal("0"))
                continue

            known = triangle.data[i][j_latest]
            if known is None:
                known = Decimal("0")

            # C_{i,n} = C_{i, j_latest} * Π_{k=j_latest}^{m-1} f_k
            ult = known * cum_factors[j_latest]
            ultimate.append(ult)
            reserves.append(ult - known)

        total_ultimate = sum(ultimate)
        total_reserve = sum(reserves)

        # Mack's CV approximation (simplified)
        cv = ReservingEngine._mack_cv(triangle, dev_factors, ultimate, n, m)

        return ReservingResult(
            method="chain_ladder",
            development_factors=dev_factors,
            cumulative_factors=cum_factors,
            ultimate_losses=ultimate,
            reserves=reserves,
            total_ultimate=total_ultimate,
            total_reserve=total_reserve,
            coefficient_of_variation=cv,
        )

    @staticmethod
    def _mack_cv(
        triangle: LossTriangle,
        dev_factors: list[Decimal],
        ultimate: list[Decimal],
        n: int,
        m: int,
    ) -> Decimal:
        """Mack's coefficient of variation (simplified).

        CV = sqrt(Σ Var(Reserve_i)) / Σ Reserve_i
        """
        if n <= 2:
            return Decimal("0")

        # Mack's σ²_j estimate
        sigma2 = []
        for j in range(m - 1):
            n_j = n - j - 1
            if n_j <= 1:
                sigma2.append(Decimal("0"))
                continue
            ss = Decimal("0")
            for i in range(n_j):
                c_j = triangle.data[i][j]
                c_j1 = triangle.data[i][j + 1]
                f_j = dev_factors[j]
                if c_j is not None and c_j1 is not None and c_j > 0:
                    residual = (c_j1 / c_j - f_j) ** 2
                    ss += c_j * residual
            sigma2.append(safe_divide(ss, Decimal(n_j - 1)))

        # Mack's MSEP approximation
        # MSEP(R_i) = C_{i,n}^2 * Σ_{k=n-i}^{n-1} (σ_k² / f_k²) * (1/C_{i,k} + 1/Σ_l C_{l,k})
        # Simplified: accumulate sigma^2 contributions weighted by ultimate loss
        total_reserve = sum(r for r in ultimate if r is not None)
        var_sum = Decimal("0")
        for i in range(1, n):
            ult_i = ultimate[i] if i < len(ultimate) else Decimal("0")
            if ult_i <= 0:
                continue
            j_latest = n - 1 - i
            # For each future development period, add variance contribution
            for k in range(j_latest, m - 1):
                if k < len(sigma2) and k < len(dev_factors):
                    f_k = dev_factors[k]
                    if f_k > 0:
                        # σ²_k / f_k² contribution scaled by ultimate^2
                        var_sum += ult_i * ult_i * safe_divide(sigma2[k], f_k * f_k)

        var_sqrt = var_sum.sqrt()
        return safe_divide(var_sqrt, total_reserve) if total_reserve > 0 else Decimal("0")

    # ═══════════════════════════════════════════════════════
    # Bornhuetter-Ferguson
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def bornhuetter_ferguson(
        triangle: LossTriangle,
        prior_ultimate: list[Decimal],
        credibility: Decimal = Decimal("0.5"),
        tail_factor: Decimal = Decimal("1.0"),
    ) -> ReservingResult:
        """Bornhuetter-Ferguson 方法.

        混合先验最终损失估计与基于数据的链梯法估计，按信度因子加权。

        BF 公式:
            Reserve_i = Prior_Ult_i × (1 - 1/Π_{k=j_latest}^{m-1} f_k)
                        + Credibility × (CL_Ult_i - Prior_Ult_i)

        Args:
            triangle: 损失三角形
            prior_ultimate: 先验最终损失估计 (每个事故年)
            credibility: 数据信度因子 (0=纯先验, 1=纯链梯法)
            tail_factor: 尾部因子

        Returns:
            ReservingResult
        """
        # First run Chain Ladder
        cl_result = ReservingEngine.chain_ladder(triangle, tail_factor)
        dev_factors = cl_result.development_factors
        cum_factors = cl_result.cumulative_factors

        n = triangle.n_accident_years
        ultimate = []
        reserves = []

        for i in range(n):
            j_latest = n - 1 - i
            if j_latest < 0 or j_latest >= len(triangle.data[i]):
                ultimate.append(Decimal("0"))
                reserves.append(Decimal("0"))
                continue

            known = triangle.data[i][j_latest]
            if known is None:
                known = Decimal("0")

            prior = prior_ultimate[i] if i < len(prior_ultimate) else known

            # %未报告: 1 - 1/Π f_k
            unreported_ratio = Decimal("1") - safe_divide(
                Decimal("1"), cum_factors[j_latest],
                Decimal("0"),
            )

            # BF 混合估计
            cl_ult = cl_result.ultimate_losses[i] if i < len(cl_result.ultimate_losses) else known
            bf_ult = known + prior * unreported_ratio + credibility * (cl_ult - prior)
            bf_ult = max(bf_ult, known)  # 最终损失不能小于已发生

            ultimate.append(bf_ult)
            reserves.append(bf_ult - known)

        total_ultimate = sum(ultimate)
        total_reserve = sum(reserves)

        return ReservingResult(
            method="bornhuetter_ferguson",
            development_factors=dev_factors,
            cumulative_factors=cum_factors,
            ultimate_losses=ultimate,
            reserves=reserves,
            total_ultimate=total_ultimate,
            total_reserve=total_reserve,
            coefficient_of_variation=cl_result.coefficient_of_variation,
            prior_ultimate=prior_ultimate,
            credibility_factor=credibility,
        )

    # ═══════════════════════════════════════════════════════
    # Cape Cod (capability ratio method)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def cape_cod(
        triangle: LossTriangle,
        premiums: list[Decimal],
        tail_factor: Decimal = Decimal("1.0"),
    ) -> ReservingResult:
        """Cape Cod 方法 — 基于保费的风险暴露调整.

        ELR (Expected Loss Ratio) = Σ known_losses / Σ (premium × %earned)

        Args:
            triangle: 损失三角形
            premiums: 各事故年的已赚保费
            tail_factor: 尾部因子

        Returns:
            ReservingResult
        """
        n = triangle.n_accident_years

        # 先运行 CL 获取进展因子
        cl_result = ReservingEngine.chain_ladder(triangle, tail_factor)
        cum_factors = cl_result.cumulative_factors

        # 计算 ELR
        earned_ratios = []
        weighted_known = Decimal("0")
        weighted_earned = Decimal("0")

        for i in range(n):
            j_latest = n - 1 - i
            if j_latest >= 0 and j_latest < len(cum_factors):
                earned_ratio = safe_divide(Decimal("1"), cum_factors[j_latest])
                earned_ratios.append(earned_ratio)

                known = triangle.data[i][j_latest]
                if known is not None:
                    prem = premiums[i] if i < len(premiums) else Decimal("1")
                    weighted_known += known
                    weighted_earned += prem * earned_ratio

        elr = safe_divide(weighted_known, weighted_earned) if weighted_earned > 0 else Decimal("0.6")

        # Cape Cod 最终损失 = premium × ELR
        prior_ult = []
        for i in range(n):
            prem = premiums[i] if i < len(premiums) else Decimal("1")
            prior_ult.append(prem * elr)

        # Use BF with credibility=1 (fully data-driven given prior)
        return ReservingEngine.bornhuetter_ferguson(
            triangle, prior_ult, credibility=Decimal("0.7"), tail_factor=tail_factor
        )

    # ═══════════════════════════════════════════════════════
    # Development Pattern Analysis
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def development_pattern(
        triangle: LossTriangle,
    ) -> dict:
        """分析损失进展模式.

        Returns:
            {dev_year: {mean, median, min, max, std}} 进展年统计
        """
        n = triangle.n_accident_years
        m = triangle.n_development_years

        result = {}
        for j in range(m):
            values = []
            for i in range(n - j):
                val = triangle.data[i][j]
                if val is not None:
                    values.append(float(val))
            if values:
                values.sort()
                n_vals = len(values)
                result[f"dev_year_{j}"] = {
                    "n_observations": n_vals,
                    "mean": Decimal(str(round(sum(values) / n_vals, 2))),
                    "median": Decimal(str(round(values[n_vals // 2], 2))),
                    "min": Decimal(str(round(values[0], 2))),
                    "max": Decimal(str(round(values[-1], 2))),
                    "std": Decimal(str(round(
                        math.sqrt(sum((v - sum(values) / n_vals) ** 2 for v in values) / max(1, n_vals - 1)),
                        2
                    ))),
                }
        return result

    # ═══════════════════════════════════════════════════════
    # Reserve Range (stochastic)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def reserve_range(
        triangle: LossTriangle,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict:
        """Bootstrap 准备金范围.

        Resamples residuals from Chain Ladder fit to generate
        reserve distribution.

        Returns:
            {mean, median, p5, p25, p75, p95, std, cv}
        """
        import random
        rng = random.Random(seed)

        # Baseline CL
        cl_base = ReservingEngine.chain_ladder(triangle)

        n = triangle.n_accident_years
        m = triangle.n_development_years

        # Compute Pearson residuals
        residuals = []
        for j in range(m - 1):
            for i in range(n - j - 1):
                c_j = triangle.data[i][j]
                c_j1 = triangle.data[i][j + 1]
                f_j = float(cl_base.development_factors[j])
                if c_j is not None and c_j1 is not None and float(c_j) > 0:
                    fitted = float(c_j) * f_j
                    if fitted > 0:
                        resid = (float(c_j1) - fitted) / math.sqrt(fitted)
                        residuals.append(resid)

        # Bootstrap
        reserve_samples = []
        for _ in range(n_bootstrap):
            # Resample residuals
            boot_resid = [rng.choice(residuals) for _ in range(len(residuals))]
            ridx = 0

            # Reconstruct triangle
            data_boot = [[None] * m for _ in range(n)]
            # First column stays the same
            for i in range(n):
                data_boot[i][0] = triangle.data[i][0]

            for j in range(m - 1):
                f_j = float(cl_base.development_factors[j])
                for i in range(n - j - 1):
                    c_j = float(data_boot[i][j]) if data_boot[i][j] is not None else 0.0
                    fitted = c_j * f_j
                    resid = boot_resid[ridx % len(boot_resid)] if residuals else 0.0
                    ridx += 1
                    c_j1 = max(0.0, fitted + resid * math.sqrt(max(fitted, 1.0)))
                    data_boot[i][j + 1] = Decimal(str(round(c_j1, 2)))

            # Chain Ladder on bootstrap
            try:
                tri_boot = LossTriangle(
                    data=data_boot,
                    n_accident_years=n,
                    n_development_years=m,
                )
                cl_boot = ReservingEngine.chain_ladder(tri_boot)
                reserve_samples.append(float(cl_boot.total_reserve))
            except Exception:
                continue

        if not reserve_samples:
            return {"mean": str(cl_base.total_reserve), "error": "bootstrap_failed"}

        reserve_samples.sort()
        n_s = len(reserve_samples)

        return {
            "mean": Decimal(str(round(sum(reserve_samples) / n_s, 2))),
            "median": Decimal(str(round(reserve_samples[n_s // 2], 2))),
            "p5": Decimal(str(round(reserve_samples[int(n_s * 0.05)], 2))),
            "p25": Decimal(str(round(reserve_samples[int(n_s * 0.25)], 2))),
            "p75": Decimal(str(round(reserve_samples[int(n_s * 0.75)], 2))),
            "p95": Decimal(str(round(reserve_samples[int(n_s * 0.95)], 2))),
            "std": Decimal(str(round(
                math.sqrt(sum((r - sum(reserve_samples) / n_s) ** 2 for r in reserve_samples) / max(1, n_s - 1)),
                2,
            ))),
        }
