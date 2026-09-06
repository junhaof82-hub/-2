from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from config import settings
from prediction.daily_learning import DailyLearningPipeline

p = argparse.ArgumentParser()
p.add_argument('tickers', nargs='*', default=list(settings.default_tickers))
a = p.parse_args()
pipeline = DailyLearningPipeline()
for ticker in a.tickers:
    try:
        r = pipeline.run(ticker)
        print(ticker, 'actuals_updated=', r['actuals_updated'], '5d_up=', round(r['prediction']['probabilities'][5]*100, 1), '%')
    except Exception as e:
        print(ticker, 'ERROR:', e)
