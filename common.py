from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, log_loss, matthews_corrcoef,
)


@dataclass
class ModelResult:
    name: str
    probability: float
    metrics: dict


class BaseProbModel:
    name = 'base'

    def __init__(self, model):
        self.model = model
        self.metrics_: dict = {}
        self.calibrator = None

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def _raw_proba(self, X) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            p = self.model.predict_proba(X)[:, 1]
        else:
            score = self.model.decision_function(X)
            p = 1 / (1 + np.exp(-score))
        return np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)

    def fit_calibrator(self, X, y):
        """Platt-style probability calibration on a chronological validation block."""
        if len(X) < 30 or len(np.unique(y)) < 2:
            return self
        p = self._raw_proba(X)
        # Logit input makes calibration more stable near 0/1.
        z = np.log(p / (1 - p)).reshape(-1, 1)
        try:
            cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)
            cal.fit(z, y)
            self.calibrator = cal
        except Exception:
            self.calibrator = None
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self._raw_proba(X)
        if self.calibrator is None:
            return p
        z = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(self.calibrator.predict_proba(z)[:, 1], 1e-4, 1 - 1e-4)

    def evaluate(self, X, y) -> dict:
        p = self.predict_proba(X)
        pred = (p >= 0.5).astype(int)
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float('nan')
        self.metrics_ = {
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'roc_auc': float(auc),
            'brier_score': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'mcc': float(matthews_corrcoef(y, pred)) if len(set(y)) > 1 else 0.0,
        }
        return self.metrics_

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'metrics': self.metrics_, 'calibrator': self.calibrator}, path)

    def load(self, path: str | Path):
        data = joblib.load(path)
        self.model = data['model']; self.metrics_ = data.get('metrics', {})
        self.calibrator = data.get('calibrator')
        return self
