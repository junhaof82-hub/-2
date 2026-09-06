from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, log_loss, matthews_corrcoef,
)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0: return 0.0
    peak = np.maximum.accumulate(equity); dd = equity / np.where(peak == 0, 1, peak) - 1
    return float(dd.min())


def classification_and_trading_metrics(y_true, probabilities, future_returns, threshold=0.5, transaction_cost=0.0005) -> dict:
    y_true = np.asarray(y_true, dtype=int); probabilities = np.asarray(probabilities, dtype=float); future_returns = np.asarray(future_returns, dtype=float)
    pred = (probabilities >= threshold).astype(int)
    active = pred == 1
    strategy_returns = np.where(active, future_returns - transaction_cost, 0.0)
    equity = np.cumprod(1 + strategy_returns)
    auc = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else float('nan')
    std = strategy_returns.std(ddof=1) if len(strategy_returns) > 1 else 0.0
    sharpe = float(strategy_returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    wins = (future_returns[active] > transaction_cost).mean() if active.any() else 0.0
    return {
        'prediction_accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'win_rate': float(wins), 'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)), 'roc_auc': float(auc),
        'brier_score': float(brier_score_loss(y_true, probabilities)),
        'log_loss': float(log_loss(y_true, probabilities, labels=[0, 1])),
        'mcc': float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else 0.0,
        'average_return': float(strategy_returns.mean()), 'max_drawdown': max_drawdown(equity),
        'sharpe_ratio': sharpe, 'trades': int(active.sum()), 'samples': int(len(y_true)),
        'avg_probability': float(np.mean(probabilities)),
    }
