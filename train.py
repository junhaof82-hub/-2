from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import argparse
from data.market_data import MarketDataService
from data.market_context import MarketContextService
from prediction.trainer import ModelTrainer

p = argparse.ArgumentParser(); p.add_argument('ticker'); p.add_argument('--period', default='5y'); a = p.parse_args()
market = MarketDataService(); prices, provider = market.get_history(a.ticker, period=a.period, interval='1d')
context = MarketContextService(market).model_context(a.period)
trained = ModelTrainer(a.ticker).train(prices, horizons=(1,3,5), save=True, context_prices=context)
print(f'Trained {a.ticker.upper()} using {provider} + {len(context)} market context series: ' + ', '.join(f'{h}d' for h in trained))
