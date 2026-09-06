from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

FRED_SERIES = {
    'cpi': 'CPIAUCSL', 'core_cpi': 'CPILFESL', 'ppi': 'PPIACO',
    'nonfarm_payrolls': 'PAYEMS', 'unemployment_rate': 'UNRATE',
    'fed_funds_rate': 'FEDFUNDS', 'treasury_2y_fred': 'DGS2', 'treasury_10y_fred': 'DGS10',
    'real_10y': 'DFII10', 'financial_conditions': 'NFCI',
}
MARKET_PROXIES = {
    'spy': 'SPY', 'qqq': 'QQQ', 'vix': '^VIX', 'treasury_10y': '^TNX',
    'dollar_index': 'DX-Y.NYB', 'oil': 'CL=F', 'gold': 'GC=F', 'semiconductors': 'SMH',
}


class MacroService:
    def _market_snapshot(self) -> dict:
        out = {}
        try:
            import yfinance as yf
            tickers = list(MARKET_PROXIES.values())
            raw = yf.download(tickers, period='2mo', interval='1d', auto_adjust=False, progress=False, threads=False, group_by='column')
            for name, ticker in MARKET_PROXIES.items():
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ('Close', ticker) in raw.columns:
                            c = pd.to_numeric(raw[('Close', ticker)], errors='coerce').dropna()
                        elif (ticker, 'Close') in raw.columns:
                            c = pd.to_numeric(raw[(ticker, 'Close')], errors='coerce').dropna()
                        else:
                            raise KeyError(ticker)
                    else:
                        c = pd.to_numeric(raw['Close'], errors='coerce').dropna()
                    out[name] = float(c.iloc[-1]) if len(c) else np.nan
                    out[f'{name}_5d_return'] = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else np.nan
                    out[f'{name}_20d_return'] = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else np.nan
                except Exception as e:
                    logger.warning('Macro proxy parse failed %s: %s', ticker, e)
                    out[name] = np.nan; out[f'{name}_5d_return'] = np.nan; out[f'{name}_20d_return'] = np.nan
            return out
        except Exception as e:
            logger.warning('Batch macro proxies failed: %s', e)
            for name in MARKET_PROXIES:
                out[name] = np.nan; out[f'{name}_5d_return'] = np.nan; out[f'{name}_20d_return'] = np.nan
            return out

    def _fred_series(self, series_id: str, limit: int = 30) -> pd.Series:
        if settings.fred_api_key:
            r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
                'series_id': series_id, 'api_key': settings.fred_api_key, 'file_type': 'json',
                'sort_order': 'desc', 'limit': limit,
            }, timeout=settings.request_timeout)
            r.raise_for_status(); obs = r.json().get('observations', [])
            rows = [(x.get('date'), x.get('value')) for x in reversed(obs)]
            s = pd.Series([pd.to_numeric(v, errors='coerce') for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
            return s.dropna()
        r = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}', timeout=settings.request_timeout)
        r.raise_for_status(); df = pd.read_csv(io.StringIO(r.text))
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(), index=pd.to_datetime(df.iloc[:, 0]))
        return s.dropna().tail(max(limit, 30))

    def get_snapshot(self) -> dict:
        out = self._market_snapshot(); series_cache = {}
        fetch_fred = bool(settings.fred_api_key) or (not settings.cloud_safe_mode) or settings.enable_fred_without_key
        for name, sid in FRED_SERIES.items():
            if not fetch_fred:
                out[name] = np.nan
                continue
            try:
                s = self._fred_series(sid, 30); series_cache[name] = s
                out[name] = float(s.iloc[-1]) if len(s) else np.nan
            except Exception as e:
                logger.warning('FRED series %s failed: %s', sid, e); out[name] = np.nan
        # Macro momentum / surprise proxies from release-to-release changes.
        for name in ('cpi', 'core_cpi', 'ppi'):
            s = series_cache.get(name)
            if s is not None and len(s) >= 13:
                out[f'{name}_yoy'] = float(s.iloc[-1] / s.iloc[-13] - 1) * 100
                out[f'{name}_mom'] = float(s.iloc[-1] / s.iloc[-2] - 1) * 100
        s = series_cache.get('nonfarm_payrolls')
        if s is not None and len(s) >= 2:
            out['nonfarm_change_thousands'] = float(s.iloc[-1] - s.iloc[-2])
        two = out.get('treasury_2y_fred', np.nan); tenf = out.get('treasury_10y_fred', np.nan)
        if np.isfinite(two) and np.isfinite(tenf):
            out['yield_curve_10y2y'] = float(tenf - two)
        fed = out.get('fed_funds_rate', np.nan)
        if np.isfinite(fed):
            out['fed_policy_bias'] = 'restrictive' if fed >= 4 else ('neutral' if fed >= 2 else 'accommodative')
        else:
            out['fed_policy_bias'] = 'unknown'
        return out
