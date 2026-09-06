from __future__ import annotations

import numpy as np
import pandas as pd
from features.feature_engineering import build_feature_frame, clean_training_data
from models.logistic_model import create_model as create_logistic
from models.random_forest import create_model as create_rf
from models.xgboost_model import create_model as create_xgb
from models.lightgbm_model import create_model as create_lgbm
from models.extra_trees_model import create_model as create_extra
from models.hist_gradient_model import create_model as create_hist
from backtest.metrics import classification_and_trading_metrics
from config import settings

FACTORIES = {
    'logistic_regression': create_logistic, 'random_forest': create_rf,
    'xgboost': create_xgb, 'lightgbm': create_lgbm,
    'extra_trees': create_extra, 'hist_gradient': create_hist,
}


def run_backtest(prices: pd.DataFrame, horizon: int = 1, train_ratio: float = 0.70, context_prices=None) -> dict:
    frame = build_feature_frame(prices, horizons=(horizon,), context_prices=context_prices)
    X, y = clean_training_data(frame, horizon)
    retcol = f'target_return_{horizon}d'
    returns = frame.loc[X.index, retcol].astype(float)
    if len(X) < 150: raise ValueError('Backtest requires at least 150 clean rows')
    split = int(len(X) * train_ratio); cal_start = max(60, int(split * .82))
    X_train, X_cal, X_test = X.iloc[:cal_start], X.iloc[cal_start:split], X.iloc[split:]
    y_train, y_cal, y_test = y.iloc[:cal_start], y.iloc[cal_start:split], y.iloc[split:]
    future_returns = returns.iloc[split:].values
    model_probs = {}; results = {}
    for name, factory in FACTORIES.items():
        model = factory(settings.random_seed).fit(X_train, y_train)
        model.fit_calibrator(X_cal, y_cal)
        p = model.predict_proba(X_test)
        results[name] = classification_and_trading_metrics(y_test.values, p, future_returns)
        model_probs[name] = p
    ensemble_p = np.mean(np.vstack([model_probs[n] for n in model_probs]), axis=0)
    results['ensemble'] = classification_and_trading_metrics(y_test.values, ensemble_p, future_returns)
    results['_meta'] = {'horizon': horizon, 'rows': len(X), 'train_rows': len(X_train), 'calibration_rows': len(X_cal), 'test_rows': len(X_test), 'method': 'chronological_holdout'}
    return results
