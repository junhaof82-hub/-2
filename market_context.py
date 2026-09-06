from __future__ import annotations

import numpy as np
import pandas as pd
from config import settings
from data.market_data import MarketDataService
from logging_config import get_logger

logger = get_logger(__name__)

SECTOR_ETFS = {
    'Technology': 'XLK', 'Financial Services': 'XLF', 'Healthcare': 'XLV',
    'Industrials': 'XLI', 'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP',
    'Energy': 'XLE', 'Basic Materials': 'XLB', 'Real Estate': 'XLRE',
    'Utilities': 'XLU', 'Communication Services': 'XLC',
}


class MarketContextService:
    def __init__(self, market: MarketDataService | None = None):
        self.market = market or MarketDataService()

    def metadata(self, ticker: str) -> dict:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            sector = info.get('sector') or ''
            industry = info.get('industry') or ''
            sector_etf = 'SMH' if 'Semiconductor' in industry else SECTOR_ETFS.get(sector)
            return {
                'sector': sector, 'industry': industry, 'sector_etf': sector_etf,
                'beta': info.get('beta'), 'short_percent_float': info.get('shortPercentOfFloat'),
                'recommendation_mean': info.get('recommendationMean'),
                'analyst_target_mean': info.get('targetMeanPrice'),
                'analyst_count': info.get('numberOfAnalystOpinions'),
            }
        except Exception as e:
            logger.warning('Metadata failed for %s: %s', ticker, e)
            return {'sector': '', 'industry': '', 'sector_etf': None}

    def relative_strength(self, ticker: str, metadata: dict | None = None) -> dict:
        metadata = metadata or self.metadata(ticker)
        symbols = {'stock': ticker, 'spy': 'SPY', 'qqq': 'QQQ'}
        if metadata.get('sector_etf'):
            symbols['sector'] = metadata['sector_etf']
        returns = {}
        for key, symbol in symbols.items():
            try:
                h, _ = self.market.get_history(symbol, period='6mo', interval='1d')
                c = h['close'].dropna()
                returns[key] = {
                    n: float(c.iloc[-1] / c.iloc[-(n + 1)] - 1) if len(c) > n else np.nan
                    for n in (5, 20, 60)
                }
            except Exception as e:
                logger.warning('Relative strength fetch failed %s: %s', symbol, e)
                returns[key] = {5: np.nan, 20: np.nan, 60: np.nan}
        rs = {}
        for n in (5, 20, 60):
            s = returns['stock'][n]
            q = returns['qqq'][n]
            p = returns['spy'][n]
            sec = returns.get('sector', {}).get(n, np.nan)
            rs[f'vs_qqq_{n}d'] = s - q if np.isfinite(s) and np.isfinite(q) else np.nan
            rs[f'vs_spy_{n}d'] = s - p if np.isfinite(s) and np.isfinite(p) else np.nan
            rs[f'vs_sector_{n}d'] = s - sec if np.isfinite(s) and np.isfinite(sec) else np.nan
        score = 50.0
        for k, v in rs.items():
            if np.isfinite(v):
                weight = 120 if '5d' in k else (80 if '20d' in k else 45)
                score += float(np.clip(v * weight, -7, 7))
        return {'returns': returns, 'relative': rs, 'score': float(np.clip(score, 0, 100))}

    def model_context(self, period: str = '5y') -> dict[str, object]:
        symbols = ('SPY', 'QQQ', '^VIX', '^TNX')
        if settings.cloud_safe_mode:
            try:
                import yfinance as yf
                raw = yf.download(list(symbols), period=period, interval='1d', auto_adjust=False, progress=False, threads=False, group_by='column')
                out = {}
                for sym in symbols:
                    try:
                        if not isinstance(raw.columns, pd.MultiIndex):
                            continue
                        cols = {}
                        for field, target in [('Open','open'),('High','high'),('Low','low'),('Close','close'),('Volume','volume')]:
                            if (field, sym) in raw.columns: cols[target] = raw[(field, sym)]
                            elif (sym, field) in raw.columns: cols[target] = raw[(sym, field)]
                        if {'open','high','low','close'}.issubset(cols):
                            df = pd.DataFrame(cols).dropna(subset=['close'])
                            if 'volume' not in df: df['volume'] = 0
                            out[sym] = df
                    except Exception as e:
                        logger.warning('Context parse failed %s: %s', sym, e)
                if out: return out
            except Exception as e:
                logger.warning('Batch model context failed: %s', e)
        out = {}
        for symbol in symbols:
            try:
                out[symbol], _ = self.market.get_history(symbol, period=period, interval='1d')
            except Exception as e:
                logger.warning('Historical model context unavailable %s: %s', symbol, e)
        return out

