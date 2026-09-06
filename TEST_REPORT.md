# Test Report — Stock AI MAX INSTITUTIONAL FINAL

Build date: 2026-09-06

Checks completed in the build environment:

- `app.py` Python bytecode compilation: PASS
- `streamlit_app.py` Python bytecode compilation: PASS
- Full AST parse: PASS
- Event taxonomy tests: PASS
- Bull/Base/Bear probability normalization: PASS
- Reliability grade calculation: PASS
- Forward-estimate intelligence calculation: PASS
- Dependency set intentionally kept lightweight: Streamlit, yfinance, pandas, numpy, plotly, scikit-learn, requests, urllib3

Cloud-safety design checks:

- No XGBoost / LightGBM / PyTorch / TensorFlow runtime dependency
- ML tree models use `n_jobs=1`
- BLAS thread caps are set before NumPy/sklearn import
- News source time budget included
- Supplemental fundamentals time budget included
- Total soft execution budget included
- Optional Options and AI Web run only on user request and only when budget remains
- Quote auto-refresh is isolated in a Streamlit fragment and does not retrain ML
- Top-level exception shield included
- Safe `streamlit_app.py` wrapper included

Limit: the build container does not have Streamlit/yfinance installed and has restricted package/network access, so an actual Community Cloud server boot could not be reproduced locally here. The code was compiled and pure computation components were tested. The target deployment installs dependencies from `requirements.txt`.
