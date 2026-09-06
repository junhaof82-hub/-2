# Stock AI MAX ULTRA v3 — Test Report

Build date: 2026-09-06

## Passed checks
- Python `py_compile`: PASS
- Synthetic OHLCV indicator pipeline: PASS
- RSI/MACD/ATR/ADX/Stochastic/MFI/OBV/volume features: PASS
- News scoring and recency weighting: PASS
- Empirical Monte Carlo bootstrap: PASS
- Four-model 1D/3D/5D training pipeline: PASS
- Chronological validation/test split: PASS
- Probability calibration fallback / shrinkage: PASS
- Full offline `analyze_max()` integration with mocked live providers: PASS
- Risk / confidence / data-quality bounds: PASS

## Cloud stability protections verified in code
- No heavy analysis on initial page load
- Network calls use explicit connect/read timeouts
- Retries are bounded to one retry
- Main analysis, news aggregation, intraday fetch and scanner use Streamlit-safe sequential execution
- Random Forest / Extra Trees use `n_jobs=1`
- Options analysis is opt-in and off by default
- Cached functions have TTL and max-entry limits
- Provider failures are isolated and returned in module-health tables
- Top-level exception shield renders a diagnostic instead of a blank page

## Note
Live external APIs were not called in the offline test environment. The code is designed to fail soft and fall back when a provider is unavailable or an API key is missing.
