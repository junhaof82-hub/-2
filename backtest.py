from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import argparse, json
from data.market_data import MarketDataService
from data.market_context import MarketContextService
from backtest.backtest import run_backtest
from backtest.walk_forward import run_walk_forward

p = argparse.ArgumentParser(); p.add_argument('ticker'); p.add_argument('--horizon', type=int, default=1, choices=[1,3,5]); p.add_argument('--walk-forward', action='store_true'); a = p.parse_args()
market=MarketDataService(); prices, provider=market.get_history(a.ticker, period='5y', interval='1d'); context=MarketContextService(market).model_context('5y')
r = run_walk_forward(prices, horizon=a.horizon, context_prices=context) if a.walk_forward else run_backtest(prices, horizon=a.horizon, context_prices=context)
print(json.dumps(r, indent=2, default=str))
