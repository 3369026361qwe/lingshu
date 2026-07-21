"""
因子加权融合引擎。

将 35 个独立因子聚合为每只股票的综合得分。

流程:
  1. 因子权重分配 — 基于 IC/IR 或等权
  2. 方向处理 — IC 符号决定因子方向
  3. 截面标准化 — Z-Score + 行业中性化
  4. 加权求和 — Σ(direction × weight × z_score)
  5. Min-Max 归一化 — 全市场 [0, 1] 得分

Usage:
    # 默认权重 (DEFAULT_WEIGHTS 归一化)
    fusion = FactorFusion()
    # 数据库 IC 权重 (DB 依赖隔离在 from_db() 工厂方法中)
    fusion = FactorFusion.from_db()
    # 自定义权重注入
    fusion = FactorFusion(weights={'roe': {'dir': +1, 'weight': 0.15}, ...})
    scores = fusion.compute(factor_values_by_code, industry_map)
"""
import logging
from collections import defaultdict
from math import isinf, isnan
from statistics import mean, stdev

from sqlalchemy import text

from shujuku.session import SessionContext

_logger = logging.getLogger(__name__)


class FactorFusion:
    """因子加权融合引擎。"""

    # 默认因子权重（基于 IC/IR 质量评估结果）
    DEFAULT_WEIGHTS: dict[str, dict] = {
        # 质量因子 (IC正, 高IR)
        'roe':                  {'dir': +1, 'weight': 0.12},
        'roa':                  {'dir': +1, 'weight': 0.10},
        'net_margin':           {'dir': +1, 'weight': 0.06},
        'gross_margin':         {'dir': +1, 'weight': 0.03},
        'cashflow_to_revenue':  {'dir': +1, 'weight': 0.02},
        # 动量因子 (dir=-1=反转策略, 依据 A 股 2018-2025 Rank IC 检验结果:
        #   IC 均值 -0.03~-0.06, 即过去涨→未来跌; 高|IR| 表明反转效应稳定。
        #   对应 momentum_factors.py 中 direction=+1 (原始方向), 此处 dir=-1 反转。
        #   如需切换为趋势跟踪: 改 dir=+1 后重新运行 factor_validation.py)
        'momentum_3m':          {'dir': -1, 'weight': 0.08},
        'momentum_6m':          {'dir': -1, 'weight': 0.06},
        'momentum_12m1m':       {'dir': -1, 'weight': 0.05},
        'momentum_1m':          {'dir': -1, 'weight': 0.03},
        'turnover_momentum':    {'dir': -1, 'weight': 0.04},
        # Alpha 因子
        'oc_20':                {'dir': -1, 'weight': 0.06},
        'cntp_20':              {'dir': -1, 'weight': 0.04},
        'roc_5':                {'dir': -1, 'weight': 0.02},
        'corr_20':              {'dir': -1, 'weight': 0.02},
        # 波动率因子 (IC负, 低波好)
        'std_20':               {'dir': -1, 'weight': 0.03},
        'historical_vol':       {'dir': -1, 'weight': 0.02},
        # 极值因子
        'min_20':               {'dir': +1, 'weight': 0.04},
        'max_20':               {'dir': -1, 'weight': 0.01},
        # 估值因子
        'pb':                   {'dir': +1, 'weight': 0.04},
        'ps':                   {'dir': +1, 'weight': 0.04},
        'pe':                   {'dir': -1, 'weight': 0.01},
        # 情绪/另类
        'volume_ratio':         {'dir': +1, 'weight': 0.01},
        'turnover_anomaly':     {'dir': -1, 'weight': 0.01},
        # Alpha 统计
        'skew_20':              {'dir': -1, 'weight': 0.01},
        'turn_20':              {'dir': -1, 'weight': 0.01},
        'amp_20':               {'dir': -1, 'weight': 0.01},
        'hl_spread_20':         {'dir': -1, 'weight': 0.01},
        'vwap_20':              {'dir': -1, 'weight': 0.01},
        'kurt_20':              {'dir': +1, 'weight': 0.01},
        # DB-computed factors (from compute_factors_from_db.py)
        'vol_3m':               {'dir': -1, 'weight': 0.015},
        'rsi_14':               {'dir': +1, 'weight': 0.02},
        'ma5_gap':              {'dir': -1, 'weight': 0.01},
    }

    # 低优先级因子 — IC 不稳定或覆盖率低，仅在 from_db() 显式加载时启用。
    # 这些因子在默认融合中不参与计算（collective 贡献 < 3%），降低认知负担。
    LOW_PRIORITY_WEIGHTS: dict[str, dict] = {
        'money_flow':           {'dir': +1, 'weight': 0.005},
        'downside_vol':         {'dir': -1, 'weight': 0.005},
        'var_95':               {'dir': -1, 'weight': 0.005},
        'vstd_20':              {'dir': -1, 'weight': 0.005},
        'vma_5':                {'dir': +1, 'weight': 0.005},
        'rank_20':              {'dir': +1, 'weight': 0.005},
    }

    def __init__(self, min_valid_factors: int = 8, weights: dict[str, dict] | None = None):
        """初始化 FactorFusion。

        Args:
            min_valid_factors: 最少有效因子数（低于此值返回兜底分数）。
            weights: 外部注入的权重 {factor_name: {dir: ±1, weight: float}}。
                     若为 None 则使用 DEFAULT_WEIGHTS 归一化。
                     数据库权重通过 from_db() 类方法加载后传入。
        """
        self._min_valid = min_valid_factors
        if weights is not None:
            self._weights = dict(weights)
        else:
            raw = dict(self.DEFAULT_WEIGHTS)
            total = sum(cfg['weight'] for cfg in raw.values())
            self._weights: dict[str, dict] = {
                fn: {'dir': cfg['dir'], 'weight': cfg['weight'] / total}
                for fn, cfg in raw.items()
            }

    # ── 权重管理 ────────────────────────────────────────────

    def set_weights(self, weights: dict[str, dict]) -> None:
        """手动设置因子权重。{factor_name: {dir: ±1, weight: float}}"""
        self._weights = dict(weights)

    @classmethod
    def from_db(cls, min_valid_factors: int = 8) -> "FactorFusion":
        """从 factor_ic_record 表加载 IC/IR 权重并创建 FactorFusion 实例。

        数据库依赖完全隔离在此工厂方法中，__init__ 保持纯计算逻辑。
        向后兼容旧的 FactorFusion(use_db_weights=True) 调用：
            ff = FactorFusion.from_db()
        """

        try:
            with SessionContext() as s:
                rows = s.execute(text(
                    "SELECT factor_name, AVG(ic) as mean_ic, COUNT(*) as n "
                    "FROM factor_ic_record GROUP BY factor_name HAVING n >= 10"
                )).fetchall()
        except Exception as e:
            _logger.warning("Failed to query factor_ic_record: %s, using defaults", e)
            return cls(min_valid_factors=min_valid_factors)

        if not rows:
            _logger.warning("No IC records found, using default weights")
            return cls(min_valid_factors=min_valid_factors)

        factor_stats = {}
        for fn, mic, _n in rows:
            mic_f = float(mic)
            factor_stats[fn] = {
                'dir': +1 if mic_f > 0 else -1,
                'mean_ic': mic_f,
                'abs_ic': abs(mic_f),
            }

        total_abs = sum(s['abs_ic'] for s in factor_stats.values())
        if total_abs <= 0:
            return cls(min_valid_factors=min_valid_factors)

        new_weights = {}
        for fn, stats in factor_stats.items():
            new_weights[fn] = {
                'dir': stats['dir'],
                'weight': stats['abs_ic'] / total_abs,
            }
        _logger.info("Loaded %d factor IC weights from DB", len(new_weights))
        return cls(min_valid_factors=min_valid_factors, weights=new_weights)

    @property
    def active_factors(self) -> list[str]:
        """当前有非零权重的因子列表。"""
        return [fn for fn, cfg in self._weights.items() if cfg['weight'] > 0.001]

    # ── 融合计算 ────────────────────────────────────────────

    def compute(
        self,
        factor_values: dict[str, dict[str, float]],  # {factor_name: {code: value}}
        industry_map: dict[str, str] | None = None,  # {code: industry}
        do_industry_neutralize: bool = True,
    ) -> dict[str, float]:
        """计算全市场综合得分。

        Args:
            factor_values: 当日各因子在全市场的值
            industry_map: 行业映射
            do_industry_neutralize: 是否行业中性化

        Returns:
            {code: composite_score} 分数越高越好，约 [0, 1]
        """
        if not factor_values:
            return {}

        # 1. 收集所有股票代码
        all_codes = set()
        for fv in factor_values.values():
            all_codes.update(fv.keys())
        all_codes = sorted(all_codes)

        if len(all_codes) < 30:
            return {}

        # 2. 对每个因子做截面 Z-Score 标准化
        z_scores: dict[str, dict[str, float]] = {}  # {factor: {code: z}}
        for fname, fv in factor_values.items():
            if fname not in self._weights:
                continue
            z_scores[fname] = self._cross_sectional_zscore(fv)

        # 3. 行业中性化（在 Z-Score 层面）
        if do_industry_neutralize and industry_map:
            z_scores = self._industry_neutralize(z_scores, industry_map)

        # 4. 加权求和
        raw_scores: dict[str, float] = {}
        valid_counts: dict[str, int] = {}
        for code in all_codes:
            score = 0.0
            n_valid = 0
            for fname, cfg in self._weights.items():
                if fname not in z_scores:
                    continue
                zv = z_scores[fname].get(code)
                if zv is None or isnan(zv) or isinf(zv):
                    continue
                score += cfg['dir'] * cfg['weight'] * zv
                n_valid += 1

            if n_valid >= self._min_valid:
                raw_scores[code] = score
                valid_counts[code] = n_valid

        if not raw_scores:
            return {}

        # 5. 去极值（Winsorize 3σ）
        raw_scores = self._winsorize(raw_scores, sigma=3.0)

        # 6. Min-Max 归一化到 [0, 1]
        return self._minmax_norm(raw_scores)

    # ── 内部方法 ────────────────────────────────────────────

    @staticmethod
    def _cross_sectional_zscore(fv: dict[str, float]) -> dict[str, float]:
        """截面 Z-Score 标准化。"""
        vals = [v for v in fv.values() if not isnan(v) and not isinf(v) and abs(v) < 1e8]
        if len(vals) < 10:
            return {}

        mu = mean(vals)
        sigma = stdev(vals) if len(vals) > 1 else 1.0
        if sigma < 1e-12:
            sigma = 1.0

        return {c: (v - mu) / sigma for c, v in fv.items()
                if not isnan(v) and not isinf(v) and abs(v) < 1e8}

    @staticmethod
    def _industry_neutralize(
        z_scores: dict[str, dict[str, float]],
        industry_map: dict[str, str],
    ) -> dict[str, dict[str, float]]:
        """行业中性化：每个行业内独立做 Z-Score。"""
        result = {}
        for fname, fv in z_scores.items():
            # 按行业分组
            groups = defaultdict(list)
            for code, z in fv.items():
                ind = industry_map.get(code, '其他')
                groups[ind].append((code, z))

            # 行业内 Z-Score
            neutralized = {}
            for _ind, pairs in groups.items():
                vals_in_group = [p[1] for p in pairs]
                if len(vals_in_group) < 3:
                    # 行业太小，保持原值
                    for c, z in pairs:
                        neutralized[c] = z
                    continue

                mu_g = mean(vals_in_group)
                sigma_g = stdev(vals_in_group) if len(vals_in_group) > 1 else 1.0
                if sigma_g < 1e-12:
                    sigma_g = 1.0

                for c, z in pairs:
                    neutralized[c] = (z - mu_g) / sigma_g

            result[fname] = neutralized

        return result

    @staticmethod
    def _winsorize(scores: dict[str, float], sigma: float = 3.0) -> dict[str, float]:
        """3-sigma 去极值。"""
        vals = list(scores.values())
        if len(vals) < 10:
            return scores
        mu = mean(vals)
        std = stdev(vals) if len(vals) > 1 else 1.0
        lower = mu - sigma * std
        upper = mu + sigma * std

        return {
            c: max(lower, min(upper, v))
            for c, v in scores.items()
        }

    @staticmethod
    def _minmax_norm(scores: dict[str, float]) -> dict[str, float]:
        """Min-Max 归一化到 [0, 1]。"""
        vals = list(scores.values())
        v_min = min(vals)
        v_max = max(vals)
        if v_max - v_min < 1e-12:
            return {c: 0.5 for c in scores}
        return {c: (v - v_min) / (v_max - v_min) for c, v in scores.items()}
