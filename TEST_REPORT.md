# Test Report — Stock AI MAX v2

Date: 2026-09-06

## Passed

- `python -m compileall` — passed for the full project.
- `pytest -q` — **10 passed**.
- Synthetic end-to-end integration — passed:
  - 5-year-like OHLCV input
  - SPY / QQQ / VIX / 10Y context
  - 6 ML models
  - 1 / 3 / 5 day probabilities
  - probability calibration
  - ensemble
  - empirical Monte Carlo
  - SQLite persistence
  - news / fundamentals / options / relative-strength injection

Example synthetic integration output produced valid probabilities and confidence for all 3 horizons.

## Environment limitation during verification

`pip install -r requirements.txt` was executed in the build container. Installed packages were detected, but the container could not reach PyPI because DNS/network access was unavailable, so missing packages such as `yfinance` / `streamlit` could not be downloaded in this environment.

Because Streamlit was not installed in the build container and could not be downloaded, a live local Streamlit server could not be started here. The app source itself passed Python compilation.

On Streamlit Community Cloud / a normal Internet-connected Python 3.12 environment, `requirements.txt` is the deployment dependency file.

## Test suite

- technical indicator tests
- Monte Carlo tests
- SQLite tests
- six-model training tests
- chronological backtest tests
- walk-forward smoke test
- market-context feature alignment
- news deduplication
- import checks
