from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, json
from prediction.predictor import StockPredictor

p = argparse.ArgumentParser()
p.add_argument('ticker')
p.add_argument('--target', type=float)
a = p.parse_args()
pred = StockPredictor()
if a.target is not None:
    print(json.dumps(pred.target_touch(a.ticker, a.target), indent=2, default=str))
else:
    r = pred.analyze(a.ticker)
    summary = {k:r[k] for k in ['ticker','last_price','probabilities','expected_ranges','bullish_score','risk_score','catalysts','risks']}
    print(json.dumps(summary, indent=2, default=str))
