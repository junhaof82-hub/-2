from __future__ import annotations

from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, brier_score_loss, log_loss

from data.market_data import MarketDataService
from data.market_context import MarketContextService
from prediction.predictor import StockPredictor
from prediction.trainer import ModelTrainer, MODEL_FACTORIES
from storage.database import Database
from logging_config import get_logger

logger = get_logger(__name__)


class DailyLearningPipeline:
    def __init__(self):
        self.market = MarketDataService(); self.db = Database(); self.predictor = StockPredictor(self.market, self.db); self.context = MarketContextService(self.market)

    def update_actuals(self, ticker: str) -> int:
        ticker = ticker.upper(); hist = self.db.prediction_history(ticker)
        if hist.empty: return 0
        prices, _ = self.market.get_history(ticker, period='5y', interval='1d'); daily = prices['close'].copy(); daily.index = pd.DatetimeIndex(daily.index).tz_convert(None).normalize()
        updated = 0
        with self.db.connect() as conn:
            for _, row in hist[hist['actual_up'].isna()].iterrows():
                ts = pd.Timestamp(row['prediction_date']); pred_date = ts.tz_convert(None).normalize() if ts.tzinfo else ts.normalize()
                future = daily[daily.index > pred_date]; h = int(row['horizon'])
                if len(future) < h: continue
                base_candidates = daily[daily.index <= pred_date]
                if len(base_candidates) == 0: continue
                base = float(base_candidates.iloc[-1]); final = float(future.iloc[h-1]); ret = final/base - 1; up = int(ret > 0)
                now = datetime.now(timezone.utc).isoformat()
                conn.execute('UPDATE predictions SET actual_up=?, actual_return=?, evaluated_at=? WHERE id=?', (up, ret, now, int(row['id'])))
                conn.execute('INSERT OR REPLACE INTO actual_results VALUES(?,?,?,?,?,?,?)', (ticker, row['prediction_date'], h, future.index[h-1].isoformat(), up, ret, now)); updated += 1
        return updated

    def refresh_live_accuracy(self, ticker: str) -> None:
        hist = self.db.prediction_history(ticker.upper()); evald = hist.dropna(subset=['actual_up']).copy()
        if evald.empty: return
        for h, group in evald.groupby('horizon'):
            y = group['actual_up'].astype(int).to_numpy()
            series = {
                'ensemble': group['probability'].astype(float).to_numpy(),
                'technical_model': (group['technical_score'].fillna(50).astype(float)/100).to_numpy(),
                'news_model': ((group['news_score'].fillna(0).astype(float)+100)/200).clip(0,1).to_numpy(),
            }
            parsed = [json.loads(x) if x else {} for x in group['model_probs_json'].tolist()]
            for name in MODEL_FACTORIES:
                vals = np.asarray([d.get(name, np.nan) for d in parsed], dtype=float)
                if np.isfinite(vals).sum() == len(vals): series[name] = vals
            for name, pvals in series.items():
                pred = (pvals >= .5).astype(int); auc = float(roc_auc_score(y, pvals)) if len(np.unique(y)) > 1 else float('nan')
                metrics = {
                    'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
                    'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred, zero_division=0)),
                    'roc_auc': auc, 'brier_score': float(brier_score_loss(y, pvals)), 'log_loss': float(log_loss(y, pvals, labels=[0,1])),
                }
                self.db.save_model_accuracy(ticker.upper(), int(h), name, metrics, len(group))

    def run(self, ticker: str) -> dict:
        ticker = ticker.upper(); updated = self.update_actuals(ticker); self.refresh_live_accuracy(ticker)
        prices, _ = self.market.get_history(ticker, period='5y', interval='1d'); context = self.context.model_context('5y')
        ModelTrainer(ticker).train(prices, horizons=(1,3,5), save=True, context_prices=context)
        prediction = self.predictor.analyze(ticker, period='5y', force_train=False, save=True); self.db.save_daily_snapshot(prediction)
        return {'ticker': ticker, 'actuals_updated': updated, 'prediction': prediction}
