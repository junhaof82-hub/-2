from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_params(close: pd.Series, lookback: int = 252) -> tuple[float, float]:
    log_returns = np.log(close / close.shift(1)).dropna().tail(lookback)
    if len(log_returns) < 20:
        return 0.0, 0.02
    # Robust drift: median/trimmed mean prevents one-off gaps dominating the simulation.
    q1, q99 = log_returns.quantile([.01, .99])
    clipped = log_returns.clip(q1, q99)
    return float(clipped.mean()), float(clipped.std(ddof=1))


def simulate_paths(last_price: float, mu: float, sigma: float, days: int = 5, simulations: int = 30000,
                   seed: int = 42, historical_returns: np.ndarray | None = None,
                   direction_prob: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if historical_returns is not None and len(historical_returns) >= 40:
        hist = np.asarray(historical_returns, dtype=float)
        hist = hist[np.isfinite(hist)]
        # Empirical bootstrap captures fat tails better than pure Gaussian GBM.
        idx = rng.integers(0, len(hist), size=(simulations, days))
        shocks = hist[idx].copy()
        if direction_prob is not None and np.isfinite(direction_prob):
            # Small model-conditioned drift tilt; deliberately capped to avoid circular overconfidence.
            tilt = np.clip((float(direction_prob) - .5) * sigma * .35, -sigma * .18, sigma * .18)
            shocks += tilt
    else:
        shocks = rng.normal(loc=mu, scale=sigma, size=(simulations, days))
    paths = last_price * np.exp(np.cumsum(shocks, axis=1))
    return paths


def _history(close: pd.Series, lookback: int = 756) -> np.ndarray:
    r = np.log(close / close.shift(1)).dropna().tail(lookback).to_numpy(dtype=float)
    if len(r) >= 50:
        lo, hi = np.quantile(r, [.005, .995]); r = np.clip(r, lo, hi)
    return r


def touch_probability(close: pd.Series, target_price: float, horizons=(1, 3, 5), simulations: int = 30000,
                      seed: int = 42, direction_prob: float | None = None) -> dict[int, float]:
    last = float(close.dropna().iloc[-1]); mu, sigma = estimate_params(close); max_h = max(horizons)
    paths = simulate_paths(last, mu, sigma, max_h, simulations, seed, _history(close), direction_prob)
    out = {}
    for h in horizons:
        sub = paths[:, :h]
        touched = (sub.max(axis=1) >= target_price) if target_price >= last else (sub.min(axis=1) <= target_price)
        out[h] = float(touched.mean())
    return out


def expected_range(close: pd.Series, days: int, confidence: float = 0.80, simulations: int = 30000,
                   seed: int = 42, direction_prob: float | None = None) -> tuple[float, float]:
    last = float(close.dropna().iloc[-1]); mu, sigma = estimate_params(close)
    paths = simulate_paths(last, mu, sigma, days, simulations, seed, _history(close), direction_prob)
    terminal = paths[:, -1]; alpha = (1 - confidence) / 2
    return float(np.quantile(terminal, alpha)), float(np.quantile(terminal, 1 - alpha))


def scenario_distribution(close: pd.Series, days: int, probability: float | None = None,
                          simulations: int = 30000, seed: int = 42) -> dict:
    last = float(close.dropna().iloc[-1]); mu, sigma = estimate_params(close)
    paths = simulate_paths(last, mu, sigma, days, simulations, seed, _history(close), probability)
    terminal = paths[:, -1]
    return {
        'p05': float(np.quantile(terminal, .05)), 'p10': float(np.quantile(terminal, .10)),
        'p25': float(np.quantile(terminal, .25)), 'median': float(np.quantile(terminal, .50)),
        'p75': float(np.quantile(terminal, .75)), 'p90': float(np.quantile(terminal, .90)),
        'p95': float(np.quantile(terminal, .95)), 'mean': float(np.mean(terminal)),
        'last': last,
    }
