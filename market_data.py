from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Iterable
import pandas as pd
import numpy as np
import requests

from config import settings
from data.base import DataProviderError, MarketDataProvider, TTLCache
from logging_config import get_logger

logger = get_logger(__name__)
CACHE = TTLCache(settings.cache_ttl_seconds)

INTERVAL_MAP = {
    '1d': '1d', '1h': '1h', '60m': '1h', '15m': '15m', '5m': '5m',
}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance can return MultiIndex columns for download()
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    aliases = {'adjusted_close': 'adj_close', 'adj_close': 'adj_close'}
    out = out.rename(columns=aliases)
    needed = ['open', 'high', 'low', 'close', 'volume']
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
    out = out[needed + [c for c in out.columns if c not in needed]]
    out.index = pd.to_datetime(out.index, utc=True, errors='coerce')
    out = out[~out.index.isna()].sort_index()
    for c in needed:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out.dropna(subset=['close'])


class YFinanceProvider(MarketDataProvider):
    name = 'yfinance'

    def history(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        interval = INTERVAL_MAP.get(interval, interval)
        key = f'yf:{ticker}:{period}:{interval}'
        cached = CACHE.get(key)
        if cached is not None:
            return cached
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False, actions=False)
            df = normalize_ohlcv(df)
            if df.empty:
                raise DataProviderError(f'No yfinance data for {ticker}')
            CACHE.set(key, df)
            return df
        except Exception as e:
            raise DataProviderError(f'yfinance failed for {ticker}: {e}') from e


class OpenBBProvider(MarketDataProvider):
    name = 'openbb'

    def history(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        try:
            from openbb import obb  # optional dependency
        except Exception as e:
            raise DataProviderError('OpenBB is not installed') from e
        days = {'1mo': 35, '3mo': 100, '6mo': 200, '1y': 370, '2y': 740, '5y': 1850}.get(period, 740)
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days)
        try:
            result = obb.equity.price.historical(
                symbol=ticker, start_date=str(start), end_date=str(end),
                interval=INTERVAL_MAP.get(interval, interval), provider='yfinance'
            )
            return normalize_ohlcv(result.to_df())
        except Exception as e:
            raise DataProviderError(f'OpenBB failed for {ticker}: {e}') from e


class AlphaVantageProvider(MarketDataProvider):
    name = 'alpha_vantage'

    def _get(self, params: dict) -> dict:
        last_error = None
        for attempt in range(settings.max_retries):
            try:
                r = requests.get('https://www.alphavantage.co/query', params=params, timeout=settings.request_timeout)
                r.raise_for_status()
                payload = r.json()
                if 'Note' in payload or 'Information' in payload:
                    raise DataProviderError(payload.get('Note') or payload.get('Information'))
                return payload
            except Exception as e:
                last_error = e
                time.sleep(min(8, 2 ** attempt))
        raise DataProviderError(f'Alpha Vantage request failed: {last_error}')

    def history(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        if not settings.alpha_vantage_api_key:
            raise DataProviderError('ALPHA_VANTAGE_API_KEY is missing')
        if interval == '1d':
            params = {'function': 'TIME_SERIES_DAILY', 'symbol': ticker, 'outputsize': 'full', 'apikey': settings.alpha_vantage_api_key}
            key_name = 'Time Series (Daily)'
        else:
            av_int = {'1h': '60min', '60m': '60min', '15m': '15min', '5m': '5min'}.get(interval)
            if not av_int:
                raise DataProviderError(f'Unsupported Alpha Vantage interval: {interval}')
            params = {'function': 'TIME_SERIES_INTRADAY', 'symbol': ticker, 'interval': av_int, 'outputsize': 'full', 'apikey': settings.alpha_vantage_api_key}
            key_name = f'Time Series ({av_int})'
        payload = self._get(params)
        series = payload.get(key_name, {})
        if not series:
            raise DataProviderError(f'No Alpha Vantage data for {ticker}')
        df = pd.DataFrame.from_dict(series, orient='index')
        df = df.rename(columns={
            '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'
        })
        return normalize_ohlcv(df)


class FMPProvider(MarketDataProvider):
    name = 'fmp'

    def history(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        if not settings.fmp_api_key:
            raise DataProviderError('FMP_API_KEY is missing')
        if interval == '1d':
            url = f'https://financialmodelingprep.com/stable/historical-price-eod/full'
            params = {'symbol': ticker, 'apikey': settings.fmp_api_key}
        else:
            fmp_int = {'1h': '1hour', '60m': '1hour', '15m': '15min', '5m': '5min'}.get(interval)
            if not fmp_int:
                raise DataProviderError(f'Unsupported FMP interval: {interval}')
            url = f'https://financialmodelingprep.com/stable/historical-chart/{fmp_int}'
            params = {'symbol': ticker, 'apikey': settings.fmp_api_key}
        r = requests.get(url, params=params, timeout=settings.request_timeout)
        r.raise_for_status()
        payload = r.json()
        rows = payload if isinstance(payload, list) else payload.get('historical', payload.get('data', []))
        if not rows:
            raise DataProviderError(f'No FMP data for {ticker}')
        df = pd.DataFrame(rows)
        idx = 'date' if 'date' in df.columns else 'datetime'
        if idx in df.columns:
            df = df.set_index(idx)
        return normalize_ohlcv(df)


class PolygonProvider(MarketDataProvider):
    name = 'polygon'

    def history(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        if not settings.polygon_api_key:
            raise DataProviderError('POLYGON_API_KEY is missing')
        mult, span = {'1d': (1, 'day'), '1h': (1, 'hour'), '60m': (1, 'hour'), '15m': (15, 'minute'), '5m': (5, 'minute')}.get(interval, (1, 'day'))
        days = {'1mo': 35, '3mo': 100, '6mo': 200, '1y': 370, '2y': 740, '5y': 1850}.get(period, 740)
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days)
        payload = None
        last_error = None
        # Polygon rebranded to Massive; support both current and legacy API hostnames.
        for base in ('https://api.massive.com', 'https://api.polygon.io'):
            try:
                url = f'{base}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start}/{end}'
                r = requests.get(url, params={'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': settings.polygon_api_key}, timeout=settings.request_timeout)
                r.raise_for_status()
                payload = r.json()
                if payload.get('results'):
                    break
            except Exception as e:
                last_error = e
                continue
        rows = (payload or {}).get('results', [])
        if not rows:
            raise DataProviderError(f'No Polygon/Massive data for {ticker}: {last_error or payload}')
        df = pd.DataFrame(rows).rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return normalize_ohlcv(df.set_index('timestamp'))


class MarketDataService:
    def __init__(self, provider_order: Iterable[str] | None = None):
        self.providers = {
            'openbb': OpenBBProvider(), 'yfinance': YFinanceProvider(), 'alpha_vantage': AlphaVantageProvider(),
            'fmp': FMPProvider(), 'polygon': PolygonProvider(),
        }
        self.provider_order = tuple(provider_order or settings.default_provider_order)

    def get_history(self, ticker: str, period: str = '2y', interval: str = '1d', preferred: str | None = None) -> tuple[pd.DataFrame, str]:
        order = ([preferred] if preferred else []) + [p for p in self.provider_order if p != preferred]
        errors: list[str] = []
        for name in order:
            provider = self.providers.get(name)
            if not provider:
                continue
            try:
                df = provider.history(ticker.upper(), period=period, interval=interval)
                if not df.empty:
                    return df, name
            except Exception as e:
                errors.append(f'{name}: {e}')
                logger.warning('Provider %s failed for %s: %s', name, ticker, e)
                time.sleep(0.15)
        raise DataProviderError('All market data providers failed: ' + ' | '.join(errors))
