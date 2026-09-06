# Stock AI MAX ULTRA v3 — Cloud Safe

A single-file Streamlit build designed for GitHub web upload + Streamlit Community Cloud.

## Upload only these files to the ROOT of your GitHub repository
- `app.py`
- `requirements.txt`
- `README.md` (optional)

Deploy with:
- Branch: `main`
- Main file path: `app.py`
- Python: 3.12 recommended when Streamlit lets you choose

## Works without API keys
The default fallback uses Yahoo Finance plus official SEC EDGAR filings.

## Optional Streamlit Secrets
Open Streamlit → Manage app → Settings / Secrets and add only the sources you want:

```toml
MASSIVE_API_KEY = ""
ALPHA_VANTAGE_API_KEY = ""
FINNHUB_API_KEY = ""
MARKETAUX_API_KEY = ""
FRED_API_KEY = ""
SEC_USER_AGENT = "Your Name your-email@example.com"
```

Never commit real API keys to GitHub.

## What v3 adds
- Current snapshot: Massive/Polygon → Finnhub → Yahoo fallback
- 1h / 15m / 5m intraday signals with pre/post-market fallback
- Multi-source news: Yahoo + Alpha Vantage + Finnhub + Marketaux + SEC EDGAR
- Official SEC 8-K / 10-Q / 10-K / Form 4 / ownership / offering filings
- FRED macro option: CPI, Core CPI, PPI, payrolls, unemployment, Fed Funds, 2Y, 10Y
- Daily ML features include SPY / QQQ / SMH / VIX / 10Y market context
- 4-model ensemble with chronological train/validation/test and probability calibration
- Confidence-driven probability shrinkage to reduce overconfidence
- Recency-weighted empirical Monte Carlo
- Fast Scanner that does not retrain the full ensemble for every ticker
- Optional options snapshot only when requested

## Black-screen protection
- No heavy work on initial page load
- External requests have timeouts and bounded retries
- Main analysis/news/intraday run sequentially for Streamlit thread safety
- ML models use `n_jobs=1`
- Results are cached
- Options are off by default
- Scanner uses a lightweight ranking path
- Every provider fails independently
- A top-level crash shield renders an error panel instead of a blank page

No system can guarantee trading accuracy or capture every market-moving event. Real-time status depends on each data provider plan and exchange entitlements.
