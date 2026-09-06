from backtest.backtest import run_backtest

def test_backtest(prices):
    r = run_backtest(prices, horizon=1)
    assert 'ensemble' in r
    for name in ['logistic_regression','random_forest','xgboost','lightgbm','ensemble']:
        assert 0 <= r[name]['prediction_accuracy'] <= 1
        assert 'sharpe_ratio' in r[name]
