import pandas as pd
from data.news import NewsService
from features.feature_engineering import build_feature_frame, available_feature_columns
from backtest.walk_forward import run_walk_forward


def test_context_features(prices):
    ctx = {
        'SPY': prices * 1.01,
        'QQQ': prices * 1.02,
        '^VIX': prices.assign(close=20 + prices['close'].pct_change().fillna(0).cumsum()),
        '^TNX': prices.assign(close=4 + prices['close'].pct_change().fillna(0).cumsum() * .1),
    }
    f = build_feature_frame(prices, horizons=(1,), context_prices=ctx)
    cols = available_feature_columns(f)
    assert 'spy_return_5d' in cols
    assert 'qqq_return_20d' in cols
    assert 'stock_vs_spy_5d' in cols


def test_news_dedupe():
    items = [
        {'title':'Nvidia beats earnings expectations', 'published_at':'2026-09-01T12:00:00Z'},
        {'title':'Nvidia beats earnings expectations!', 'published_at':'2026-09-01T12:01:00Z'},
        {'title':'Nvidia launches new AI product', 'published_at':'2026-09-01T13:00:00Z'},
    ]
    out = NewsService._dedupe(items)
    assert len(out) == 2


def test_walk_forward_smoke(prices):
    r = run_walk_forward(prices, horizon=1, step=80, max_windows=2)
    assert 'ensemble' in r
    assert r['_meta']['samples'] > 0
    assert 0 <= r['ensemble']['prediction_accuracy'] <= 1
