from __future__ import annotations

import numpy as np
import pandas as pd
from prediction.predictor import StockPredictor
from logging_config import get_logger

logger = get_logger(__name__)


def scan(tickers: list[str], predictor: StockPredictor | None = None, horizon: int = 5) -> pd.DataFrame:
    predictor = predictor or StockPredictor(); rows = []
    for ticker in tickers:
        try:
            r = predictor.analyze(ticker, save=True)
            d = r.get('scenario_distributions', {}).get(horizon, {})
            last = r['last_price']; median = float(d.get('median', np.mean(r['expected_ranges'][horizon])))
            expected_mid_return = (median / last - 1) * 100
            conf = float(r.get('confidence', {}).get(horizon, 50)); quality = float(r.get('data_quality', 50))
            rows.append({
                'Ticker': ticker.upper(), 'Up Probability': r['probabilities'][horizon] * 100,
                'Expected Return %': expected_mid_return, 'Risk': r['risk_score'], 'Confidence': conf,
                'Data Quality': quality, 'News Coverage': r.get('news', {}).get('coverage_score', 0),
                'Relative Strength': r.get('relative_strength', {}).get('score', 50),
                'Bullish Score': r['bullish_score'],
            })
        except Exception as e:
            logger.warning('Scanner failed %s: %s', ticker, e)
    if not rows:
        return pd.DataFrame(columns=['Rank','Ticker','Up Probability','Expected Return %','Risk','Confidence'])
    df = pd.DataFrame(rows)
    df['rank_score'] = (
        df['Up Probability'] + .30 * df['Expected Return %'] + .12 * df['Confidence'] +
        .05 * df['Data Quality'] + .04 * df['Relative Strength'] - 1.55 * df['Risk']
    )
    df = df.sort_values('rank_score', ascending=False).reset_index(drop=True); df.insert(0, 'Rank', range(1, len(df)+1))
    return df.drop(columns=['rank_score'])
