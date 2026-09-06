from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
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
from logging_config import get_logger

logger = get_logger(__name__)

MODEL_FACTORIES = {
    'logistic_regression': create_logistic,
    'random_forest': create_rf,
    'xgboost': create_xgb,
    'lightgbm': create_lgbm,
    'extra_trees': create_extra,
    'hist_gradient': create_hist,
}


@dataclass
class TrainedHorizon:
    horizon: int
    models: dict
    metrics: dict
    feature_columns: list[str]
    model_version: str = settings.model_version


class ModelTrainer:
    def __init__(self, ticker: str, model_dir: Path | None = None):
        self.ticker = ticker.upper()
        self.model_dir = Path(model_dir or settings.model_dir) / self.ticker
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, prices: pd.DataFrame, horizons=(1, 3, 5), save: bool = True,
              context_prices: dict[str, pd.DataFrame] | None = None) -> dict[int, TrainedHorizon]:
        frame = build_feature_frame(prices, tuple(horizons), context_prices=context_prices)
        trained = {}
        for h in horizons:
            X, y = clean_training_data(frame, h)
            if len(X) < 180 or y.nunique() < 2:
                # Preserve compatibility for short synthetic/test datasets.
                if len(X) < 120 or y.nunique() < 2:
                    raise ValueError(f'Not enough training data for {self.ticker} {h}d: {len(X)} rows')
            n = len(X)
            test_n = max(30, int(n * 0.18))
            cal_n = max(30, int(n * 0.14))
            if test_n + cal_n > n * .45:
                test_n = max(25, int(n * .18)); cal_n = max(20, int(n * .12))
            train_end = n - test_n - cal_n
            cal_end = n - test_n
            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_cal, y_cal = X.iloc[train_end:cal_end], y.iloc[train_end:cal_end]
            X_test, y_test = X.iloc[cal_end:], y.iloc[cal_end:]

            models, metrics = {}, {}
            active_factories = MODEL_FACTORIES if settings.full_model_mode else {k:v for k,v in MODEL_FACTORIES.items() if k in ('logistic_regression','random_forest','extra_trees','hist_gradient')}
            for name, factory in active_factories.items():
                try:
                    m = factory(settings.random_seed).fit(X_train, y_train)
                    m.fit_calibrator(X_cal, y_cal)
                    met = m.evaluate(X_test, y_test)
                    met['train_rows'] = int(len(X_train)); met['calibration_rows'] = int(len(X_cal)); met['test_rows'] = int(len(X_test))
                    models[name] = m; metrics[name] = met
                    if save:
                        m.save(self.model_dir / f'{name}_{h}d.joblib')
                except Exception as e:
                    logger.exception('Training failed %s %s %sd: %s', self.ticker, name, h, e)
            if not models:
                raise RuntimeError(f'All models failed for {self.ticker} {h}d')
            meta = {
                'ticker': self.ticker, 'horizon': h, 'feature_columns': list(X.columns),
                'metrics': metrics, 'model_version': settings.model_version,
                'rows': n, 'train_end': train_end, 'cal_end': cal_end,
            }
            if save:
                (self.model_dir / f'meta_{h}d.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
            trained[h] = TrainedHorizon(h, models, metrics, list(X.columns), settings.model_version)
        return trained

    def load(self, horizon: int) -> TrainedHorizon | None:
        meta_path = self.model_dir / f'meta_{horizon}d.json'
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            return None
        if meta.get('model_version') != settings.model_version:
            logger.info('Ignoring stale model bundle %s %sd (%s != %s)', self.ticker, horizon, meta.get('model_version'), settings.model_version)
            return None
        models = {}
        active_factories = MODEL_FACTORIES if settings.full_model_mode else {k:v for k,v in MODEL_FACTORIES.items() if k in ('logistic_regression','random_forest','extra_trees','hist_gradient')}
        for name, factory in active_factories.items():
            p = self.model_dir / f'{name}_{horizon}d.joblib'
            if p.exists():
                try:
                    models[name] = factory(settings.random_seed).load(p)
                except Exception as e:
                    logger.warning('Could not load %s: %s', p, e)
        if not models:
            return None
        return TrainedHorizon(horizon, models, meta.get('metrics', {}), meta.get('feature_columns', []), meta.get('model_version', ''))
