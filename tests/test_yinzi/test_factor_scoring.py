"""测试 factor_scoring 模块 — 5 个独立评分函数。"""
from yinzi.factor_scoring import (
    _score_ic_20d,
    _score_ic_decay,
    _score_monotonicity,
    _score_regime,
    _score_stability,
    compute_final_grade,
)


def _make_result():
    return {'name': 'test_factor', 'checks': {}, 'warnings': [], 'score': 0, 'max_score': 0}


# ── Check 1: IC at 20d ──────────────────────────────────────

def test_ic_20d_strong():
    r = _make_result()
    summary = {'test_factor': {20: {'mean_ic': 0.06, 't_stat': 4.0}}}
    _score_ic_20d(r, summary, 'test_factor')
    assert r['score'] == 30
    assert 'STRONG' in r['checks']['ic_20d']


def test_ic_20d_pass():
    r = _make_result()
    summary = {'test_factor': {20: {'mean_ic': 0.03, 't_stat': 2.5}}}
    _score_ic_20d(r, summary, 'test_factor')
    assert r['score'] == 20
    assert 'PASS' in r['checks']['ic_20d']


def test_ic_20d_weak():
    r = _make_result()
    summary = {'test_factor': {20: {'mean_ic': 0.015, 't_stat': 1.5}}}
    _score_ic_20d(r, summary, 'test_factor')
    assert r['score'] == 10
    assert 'WEAK' in r['checks']['ic_20d']
    assert any('weak' in w.lower() for w in r['warnings'])


def test_ic_20d_fail():
    r = _make_result()
    summary = {'test_factor': {20: {'mean_ic': 0.005, 't_stat': 0.5}}}
    _score_ic_20d(r, summary, 'test_factor')
    assert r['score'] == 0
    assert 'FAIL' in r['checks']['ic_20d']


def test_ic_20d_missing():
    r = _make_result()
    _score_ic_20d(r, {}, 'test_factor')
    assert r['score'] == 0
    assert r['checks']['ic_20d'] == 'NO DATA'


# ── Check 2: IC decay profile ───────────────────────────────

def test_ic_decay_good():
    r = _make_result()
    summary = {'test_factor': {
        5: {'mean_ic': 0.01},
        10: {'mean_ic': 0.02},
        20: {'mean_ic': 0.04},
        40: {'mean_ic': 0.035},
        60: {'mean_ic': 0.025},
    }}
    _score_ic_decay(r, summary, 'test_factor')
    assert r['score'] == 20  # peak 0.04 > 0.03, decay_ratio 0.025/0.04=0.625 > 0.5
    assert 'GOOD' in r['checks']['ic_decay']


def test_ic_decay_no_data():
    r = _make_result()
    _score_ic_decay(r, {}, 'test_factor')
    assert r['score'] == 0
    assert r['checks']['ic_decay'] == 'NO DATA'


def test_ic_decay_no_profile():
    r = _make_result()
    summary = {'test_factor': {}}  # empty
    _score_ic_decay(r, summary, 'test_factor')
    assert r['score'] == 0
    assert r['checks']['ic_decay'] == 'NO PROFILE'


# ── Check 3: Regime robustness ──────────────────────────────

def test_regime_robust():
    r = _make_result()
    summary = {'test_factor': {
        'bull': {'mean_ic': 0.05},
        'bear': {'mean_ic': 0.03},
        'sideways': {'mean_ic': 0.04},
    }}
    _score_regime(r, summary, 'test_factor')
    assert r['score'] == 15
    assert 'ROBUST' in r['checks']['regime']


def test_regime_unstable():
    r = _make_result()
    summary = {'test_factor': {
        'bull': {'mean_ic': 0.05},
        'bear': {'mean_ic': -0.03},
        'sideways': {'mean_ic': 0.01},
    }}
    _score_regime(r, summary, 'test_factor')
    assert r['score'] == 5
    assert 'UNSTABLE' in r['checks']['regime']
    assert len(r['warnings']) > 0


def test_regime_missing():
    r = _make_result()
    _score_regime(r, {}, 'test_factor')
    assert r['score'] == 0
    assert r['checks']['regime'] == 'NO DATA'


# ── Check 4: Monotonicity ───────────────────────────────────

def test_monotonicity_strong():
    r = _make_result()
    _score_monotonicity(r, {'test_factor': {'mean_mono': 0.80}}, 'test_factor')
    assert r['score'] == 15
    assert 'STRONG' in r['checks']['monotonicity']


def test_monotonicity_non_monotonic():
    r = _make_result()
    _score_monotonicity(r, {'test_factor': {'mean_mono': 0.40}}, 'test_factor')
    assert r['score'] == 0
    assert 'NON-MONOTONIC' in r['checks']['monotonicity']
    assert len(r['warnings']) > 0


def test_monotonicity_missing():
    r = _make_result()
    _score_monotonicity(r, {}, 'test_factor')
    assert r['score'] == 0
    assert r['checks']['monotonicity'] == 'NO DATA'


# ── Check 5: Stability ──────────────────────────────────────

def test_stability_stable():
    r = _make_result()
    _score_stability(r, {'test_factor': {'mean_autocorr': 0.90, 'implied_turnover': 0.05}}, 'test_factor')
    assert r['score'] == 20
    assert 'STABLE' in r['checks']['stability']


def test_stability_high_turnover():
    r = _make_result()
    _score_stability(r, {'test_factor': {'mean_autocorr': 0.55, 'implied_turnover': 0.45}}, 'test_factor')
    assert r['score'] == 8
    assert 'HIGH_TURNOVER' in r['checks']['stability']
    assert len(r['warnings']) > 0


def test_stability_missing():
    r = _make_result()
    _score_stability(r, {}, 'test_factor')
    assert r['score'] == 0
    assert r['checks']['stability'] == 'NO DATA'


# ── Final grade ─────────────────────────────────────────────

def test_final_grade_pass():
    g = compute_final_grade(85, 100)
    assert g['grade'] == 'PASS ✓'
    assert g['action'] == 'KEEP'
    assert g['score_pct'] == 85.0


def test_final_grade_conditional():
    g = compute_final_grade(60, 100)
    assert g['grade'] == 'CONDITIONAL △'
    assert g['action'] == 'REVIEW'


def test_final_grade_fail():
    g = compute_final_grade(40, 100)
    assert g['grade'] == 'FAIL ✗'
    assert g['action'] == 'DROP'


def test_final_grade_zero_max():
    g = compute_final_grade(0, 0)
    assert g['score_pct'] == 0
    assert g['action'] == 'DROP'
