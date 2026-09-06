from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from features.feature_engineering import build_feature_frame, clean_training_data
from models.logistic_model import create_model as create_logistic
from models.random_forest import create_model as create_rf
from models.xgboost_model import create_model as create_xgb
from models.lightgbm_model import create_model as create_lgbm
from models.extra_trees_model import create_model as create_extra
from models.hist_gradient_model import create_model as create_hist
from backtest.metrics import classification_and_trading_metrics

FACTORIES = {
    'logistic_regression': create_logistic, 'random_forest': create_rf,
    'xgboost': create_xgb, 'lightgbm': create_lgbm,
    'extra_trees': create_extra, 'hist_gradient': create_hist,
}


def run_walk_forward(prices: pd.DataFrame, horizon: int = 1, context_prices=None,
                     initial_train: int | None = None, step: int = 25, max_windows: int | None = None) -> dict:
    """Expanding-window out-of-sample test. No future labels are used to train earlier windows."""
    frame = build_feature_frame(prices, horizons=(horizon,), context_prices=context_prices)
    X, y = clean_training_data(frame, horizon); returns = frame.loc[X.index, f'target_return_{horizon}d'].astype(float)
    n = len(X); initial_train = initial_train or max(220, int(n * .55))
    if n - initial_train < 40: raise ValueError('Walk-forward backtest needs more history')
    indices = list(range(initial_train, n, step))
    if max_windows and len(indices) > max_windows: indices = indices[-max_windows:]
    pred_store = {name: [] for name in FACTORIES}; y_store = []; r_store = []; date_store = []
    for window_no, start in enumerate(indices):
        end = min(n, start + step); train_X = X.iloc[:start]; train_y = y.iloc[:start]
        cal_n = max(30, int(len(train_X) * .14)); base_X = train_X.iloc[:-cal_n]; base_y = train_y.iloc[:-cal_n]; cal_X = train_X.iloc[-cal_n:]; cal_y = train_y.iloc[-cal_n:]
        test_X = X.iloc[start:end]
        if len(test_X) == 0: continue
        for name, factory in FACTORIES.items():
            model = factory(settings.random_seed + window_no).fit(base_X, base_y); model.fit_calibrator(cal_X, cal_y)
            pred_store[name].extend(model.predict_proba(test_X).tolist())
        y_store.extend(y.iloc[start:end].tolist()); r_store.extend(returns.iloc[start:end].tolist()); date_store.extend([str(x) for x in X.index[start:end]])
    y_arr = np.asarray(y_store, dtype=int); r_arr = np.asarray(r_store, dtype=float)
    results = {}
    for name, vals in pred_store.items(): results[name] = classification_and_trading_metrics(y_arr, np.asarray(vals), r_arr)
    ensemble = np.mean(np.vstack([np.asarray(pred_store[n]) for n in pred_store]), axis=0)
    results['ensemble'] = classification_and_trading_metrics(y_arr, ensemble, r_arr)
    results['_meta'] = {'horizon': horizon, 'samples': len(y_arr), 'windows': len(indices), 'step': step, 'method': 'expanding_walk_forward', 'start': date_store[0] if date_store else None, 'end': date_store[-1] if date_store else None}
    return results
