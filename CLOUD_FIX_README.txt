Stock AI MAX v2.1 Cloud Fixed

Why this build exists:
- Streamlit cloud can restart a worker when CPU/RAM spikes.
- v2 trained 6 models x 3 horizons with multi-threaded tree models on first click.
- It also ran 30,000-path Monte Carlo and many network calls in one request.

Cloud Fixed defaults:
- 4 stable calibrated models x 3 horizons in cloud mode.
- All tree models restricted to one CPU thread.
- Monte Carlo reduced to 5,000 paths.
- Market/macro context uses batch downloads.
- FRED bulk downloads disabled unless FRED key is configured.
- Options disabled by default on cloud (can be enabled with ENABLE_OPTIONS=1).
- 3-year training history default.
- Full 6-model mode remains available with FULL_MODEL_MODE=1.

Recommended Streamlit main file: app.py
