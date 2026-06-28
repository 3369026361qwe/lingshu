"""因子评分函数 — 从 validate_factor 提取的 5 个独立评分维度。

每个函数签名：(result: dict, summary_dict: dict, fname: str) -> None
通过修改 result 字典来累积分数、检查描述和警告。
"""
from statistics import stdev

FORWARD_HORIZONS = [5, 10, 20, 40, 60]
DEFAULT_REGIME_ORDER = ['bull', 'bear', 'sideways']


def _score_ic_20d(result: dict, ic_decay_summary: dict, fname: str) -> None:
    """Check 1: IC at 20d (weight: 30%). Max score = 30."""
    result['max_score'] += 30
    if fname not in ic_decay_summary or 20 not in ic_decay_summary[fname]:
        result['checks']['ic_20d'] = 'NO DATA'
        return

    stats = ic_decay_summary[fname][20]
    ic_abs = abs(stats['mean_ic'])
    t_abs = abs(stats['t_stat'])
    desc = f"|IC|={ic_abs:.4f}, |t|={t_abs:.1f}"

    if ic_abs > 0.05 and t_abs > 3.0:
        result['score'] += 30
        result['checks']['ic_20d'] = desc + ' ✓ STRONG'
    elif ic_abs > 0.02 and t_abs > 2.0:
        result['score'] += 20
        result['checks']['ic_20d'] = desc + ' ✓ PASS'
    elif ic_abs > 0.01:
        result['score'] += 10
        result['checks']['ic_20d'] = desc + ' ⚠ WEAK'
        result['warnings'].append(f'IC_20d weak: |IC|={ic_abs:.4f}')
    else:
        result['checks']['ic_20d'] = desc + ' ✗ FAIL'
        result['warnings'].append(f'IC_20d fail: |IC|={ic_abs:.4f}')


def _score_ic_decay(result: dict, ic_decay_summary: dict, fname: str) -> None:
    """Check 2: IC decay profile (weight: 20%). Max score = 20."""
    result['max_score'] += 20
    if fname not in ic_decay_summary:
        result['checks']['ic_decay'] = 'NO DATA'
        return

    ic_profile = {}
    for h in FORWARD_HORIZONS:
        if h in ic_decay_summary[fname]:
            ic_profile[h] = abs(ic_decay_summary[fname][h]['mean_ic'])

    if not ic_profile:
        result['checks']['ic_decay'] = 'NO PROFILE'
        return

    peak_h = max(ic_profile, key=ic_profile.get)
    ic_peak = ic_profile[peak_h]
    ic_5d = ic_profile.get(5, 0)
    ic_60d = ic_profile.get(60, ic_peak)
    desc = f"peak@{peak_h}d={ic_peak:.4f}, 5d={ic_5d:.4f}, 60d={ic_60d:.4f}"

    decay_ratio = ic_60d / ic_peak if ic_peak > 0 else 0
    if ic_peak > 0.03 and decay_ratio > 0.5:
        result['score'] += 20
        result['checks']['ic_decay'] = desc + ' ✓ GOOD'
    elif ic_peak > 0.02 and decay_ratio > 0.3:
        result['score'] += 15
        result['checks']['ic_decay'] = desc + ' ✓ OK'
    elif ic_peak > 0.01:
        result['score'] += 8
        result['checks']['ic_decay'] = desc + ' ⚠ FAST_DECAY'
    else:
        result['score'] += 3
        result['checks']['ic_decay'] = desc + ' ✗ NO_POWER'


def _score_regime(result: dict, regime_summary: dict, fname: str) -> None:
    """Check 3: Regime robustness (weight: 15%). Max score = 15."""
    result['max_score'] += 15
    if fname not in regime_summary:
        result['checks']['regime'] = 'NO DATA'
        return

    regimes = regime_summary[fname]
    ic_by_regime = {r: regimes[r]['mean_ic'] for r in regimes}
    desc = ' | '.join(
        f"{r}: IC={ic_by_regime[r]:+.4f}"
        for r in DEFAULT_REGIME_ORDER if r in ic_by_regime
    )

    if len(ic_by_regime) >= 3:
        signs = [1 if v > 0 else -1 for v in ic_by_regime.values()]
        sign_consistent = len(set(signs)) == 1
        mag_variance = stdev(list(ic_by_regime.values())) if len(ic_by_regime) > 1 else 999

        if sign_consistent and mag_variance < 0.03:
            result['score'] += 15
            result['checks']['regime'] = desc + ' ✓ ROBUST'
        elif sign_consistent:
            result['score'] += 10
            result['checks']['regime'] = desc + ' ✓ STABLE_SIGN'
        else:
            result['score'] += 5
            result['checks']['regime'] = desc + ' ⚠ UNSTABLE'
            result['warnings'].append('IC sign varies across regimes')
    elif len(ic_by_regime) >= 2:
        result['score'] += 8
        result['checks']['regime'] = desc + ' ⚠ INCOMPLETE'


def _score_monotonicity(result: dict, monotonicity_scores: dict, fname: str) -> None:
    """Check 4: Monotonicity (weight: 15%). Max score = 15."""
    result['max_score'] += 15
    if fname not in monotonicity_scores:
        result['checks']['monotonicity'] = 'NO DATA'
        return

    mono = monotonicity_scores[fname]['mean_mono']
    desc = f"score={mono:.3f}"

    if mono > 0.75:
        result['score'] += 15
        result['checks']['monotonicity'] = desc + ' ✓ STRONG'
    elif mono > 0.60:
        result['score'] += 12
        result['checks']['monotonicity'] = desc + ' ✓ OK'
    elif mono > 0.50:
        result['score'] += 6
        result['checks']['monotonicity'] = desc + ' ⚠ MARGINAL'
    else:
        result['checks']['monotonicity'] = desc + ' ✗ NON-MONOTONIC'
        result['warnings'].append(f'Monotonicity low: {mono:.3f}')


def _score_stability(result: dict, autocorr_summary: dict, fname: str) -> None:
    """Check 5: Factor stability / turnover (weight: 20%). Max score = 20."""
    result['max_score'] += 20
    if fname not in autocorr_summary:
        result['checks']['stability'] = 'NO DATA'
        return

    ac = autocorr_summary[fname]
    desc = f"autocorr={ac['mean_autocorr']:.3f}, turnover≈{ac['implied_turnover']:.2%}"

    if ac['mean_autocorr'] > 0.85:
        result['score'] += 20
        result['checks']['stability'] = desc + ' ✓ STABLE'
    elif ac['mean_autocorr'] > 0.70:
        result['score'] += 15
        result['checks']['stability'] = desc + ' ✓ OK'
    elif ac['mean_autocorr'] > 0.50:
        result['score'] += 8
        result['checks']['stability'] = desc + ' ⚠ HIGH_TURNOVER'
        result['warnings'].append(f'High turnover: autocorr={ac["mean_autocorr"]:.3f}')
    else:
        result['score'] += 3
        result['checks']['stability'] = desc + ' ✗ UNSTABLE'
        result['warnings'].append(f'Very unstable: autocorr={ac["mean_autocorr"]:.3f}')


def compute_final_grade(score: int, max_score: int) -> dict:
    """根据最终分数百分比计算等级和操作建议。"""
    score_pct = score / max_score * 100 if max_score > 0 else 0
    if score_pct >= 80:
        grade, action = 'PASS ✓', 'KEEP'
    elif score_pct >= 55:
        grade, action = 'CONDITIONAL △', 'REVIEW'
    else:
        grade, action = 'FAIL ✗', 'DROP'
    return {'grade': grade, 'action': action, 'score_pct': score_pct}
