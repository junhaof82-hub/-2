from __future__ import annotations

import math
import numpy as np

DEFAULT_COMPONENT_WEIGHTS = {
    'ml': 0.50, 'technical': 0.15, 'fundamental': 0.08, 'news': 0.12,
    'macro': 0.06, 'relative_strength': 0.05, 'options': 0.04,
}


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def model_quality(metric: dict) -> float:
    auc = metric.get('roc_auc', np.nan)
    bal = metric.get('balanced_accuracy', metric.get('accuracy', np.nan))
    brier = metric.get('brier_score', np.nan)
    parts = []
    if _finite(auc): parts.append(np.clip((float(auc) - .50) / .20, -1, 1))
    if _finite(bal): parts.append(np.clip((float(bal) - .50) / .18, -1, 1))
    if _finite(brier): parts.append(np.clip((.25 - float(brier)) / .10, -1, 1))
    if not parts:
        return 0.0
    return float(np.mean(parts))


def weighted_model_probability(model_probs: dict[str, float], model_metrics: dict[str, dict] | None = None) -> float:
    if not model_probs:
        return 0.5
    raw_weights = {}
    for name, p in model_probs.items():
        q = model_quality((model_metrics or {}).get(name, {}))
        # Even weak models keep a small vote; stronger calibrated out-of-sample models get more weight.
        raw_weights[name] = float(np.clip(0.35 + 0.65 * max(-0.15, q), 0.12, 1.0))
    den = sum(raw_weights.values()) or 1.0
    p = sum(float(model_probs[n]) * raw_weights[n] for n in model_probs) / den
    # Shrink extreme forecasts toward 50% unless the ensemble is strongly supported.
    avg_quality = np.mean([max(0, model_quality((model_metrics or {}).get(n, {}))) for n in model_probs])
    shrink = float(np.clip(0.55 + 0.45 * avg_quality, .55, 1.0))
    return float(np.clip(.5 + (p - .5) * shrink, .03, .97))


def model_agreement(model_probs: dict[str, float]) -> float:
    vals = np.asarray(list(model_probs.values()), dtype=float)
    if len(vals) <= 1:
        return 0.5
    dispersion = float(np.std(vals))
    return float(np.clip(1 - dispersion / .20, 0, 1))


def fundamental_probability(f: dict) -> float:
    score = 50.0
    rg = f.get('revenue_growth', np.nan); eg = f.get('earnings_growth', np.nan)
    gm = f.get('gross_margin', np.nan); om = f.get('operating_margin', np.nan); pe = f.get('forward_pe', np.nan)
    fcf = f.get('free_cash_flow', np.nan); surprise = f.get('eps_surprise', np.nan)
    if _finite(rg): score += np.clip(float(rg) * 80, -12, 12)
    if _finite(eg): score += np.clip(float(eg) * 60, -12, 12)
    if _finite(gm): score += np.clip((float(gm) - 0.35) * 20, -5, 8)
    if _finite(om): score += np.clip((float(om) - 0.15) * 25, -6, 8)
    if _finite(pe) and float(pe) > 55: score -= min(10, (float(pe) - 55) / 5)
    if _finite(fcf) and float(fcf) > 0: score += 5
    if _finite(surprise): score += np.clip(float(surprise) * 0.5, -8, 8)
    target = f.get('analyst_target_mean', np.nan); last = f.get('last_price', np.nan)
    if _finite(target) and _finite(last) and float(last) > 0:
        score += np.clip((float(target) / float(last) - 1) * 30, -5, 7)
    return float(np.clip(score / 100, 0.08, 0.92))


def macro_probability(m: dict) -> float:
    score = 50.0
    spy = m.get('spy_5d_return', np.nan); qqq = m.get('qqq_5d_return', np.nan)
    vix = m.get('vix', np.nan); ten = m.get('treasury_10y', np.nan)
    curve = m.get('yield_curve_10y2y', np.nan)
    if _finite(spy): score += np.clip(float(spy) * 180, -8, 8)
    if _finite(qqq): score += np.clip(float(qqq) * 220, -10, 10)
    if _finite(vix): score += 5 if float(vix) < 18 else (-8 if float(vix) > 28 else 0)
    if _finite(ten) and float(ten) > 5: score -= 6
    if _finite(curve) and float(curve) < -0.5: score -= 2
    if m.get('fed_policy_bias') == 'restrictive': score -= 3
    elif m.get('fed_policy_bias') == 'accommodative': score += 3
    return float(np.clip(score / 100, 0.08, 0.92))


def combine_components(ml_prob: float, technical_prob: float, fundamental_prob: float | None,
                       news_prob: float | None, macro_prob: float | None,
                       relative_strength_prob: float | None = None, options_prob: float | None = None,
                       weights: dict | None = None, data_quality: dict | None = None) -> tuple[float, dict]:
    weights = dict(weights or DEFAULT_COMPONENT_WEIGHTS)
    vals = {
        'ml': ml_prob, 'technical': technical_prob, 'fundamental': fundamental_prob,
        'news': news_prob, 'macro': macro_prob, 'relative_strength': relative_strength_prob,
        'options': options_prob,
    }
    available = {k: v for k, v in vals.items() if v is not None and _finite(v)}
    if data_quality:
        # Downweight news/options when coverage is thin rather than pretending missing data is neutral.
        news_cov = float(data_quality.get('news_coverage', 100)) / 100
        options_ok = bool(data_quality.get('options_available', True))
        if 'news' in available: weights['news'] *= float(np.clip(news_cov, .25, 1.0))
        if 'options' in available and not options_ok: weights['options'] *= .25
    norm = sum(weights[k] for k in available) or 1.0
    used_weights = {k: weights[k] / norm for k in available}
    p = sum(float(available[k]) * used_weights[k] for k in available)
    return float(np.clip(p, 0.03, 0.97)), used_weights
