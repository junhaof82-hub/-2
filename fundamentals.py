from __future__ import annotations

import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

FIELDS = [
    'revenue', 'eps', 'eps_surprise', 'revenue_surprise', 'forward_pe', 'market_cap',
    'free_cash_flow', 'gross_margin', 'operating_margin', 'earnings_growth', 'revenue_growth',
    'debt_to_equity', 'current_ratio', 'return_on_equity', 'beta', 'short_percent_float',
    'held_percent_institutions', 'recommendation_mean', 'analyst_target_mean', 'analyst_count',
    'next_earnings_date', 'days_to_earnings',
]


def _safe(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _days_to(ts) -> float:
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None: t = t.tz_localize('UTC')
        else: t = t.tz_convert('UTC')
        return float((t - pd.Timestamp.now(tz='UTC')).total_seconds() / 86400)
    except Exception:
        return np.nan


class FundamentalsService:
    def _from_yfinance(self, ticker: str) -> dict:
        try:
            import yfinance as yf
        except Exception as e:
            logger.warning('yfinance unavailable: %s', e)
            return {f: np.nan for f in FIELDS} | {'provider': 'none'}
        t = yf.Ticker(ticker); info = {}
        try: info = t.info or {}
        except Exception as e: logger.warning('yfinance info failed for %s: %s', ticker, e)
        eps_surprise = np.nan; next_earnings = None
        try:
            ed = t.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                now = pd.Timestamp.now(tz='UTC')
                idx = pd.DatetimeIndex(ed.index)
                if idx.tz is None: idx = idx.tz_localize('UTC')
                future = idx[idx >= now]
                if len(future): next_earnings = future.min().isoformat()
                past_mask = idx < now
                if past_mask.any():
                    past = ed.loc[past_mask]
                    col = next((c for c in past.columns if 'Surprise' in str(c)), None)
                    if col: eps_surprise = _safe(past.iloc[0][col])
        except Exception:
            pass
        return {
            'revenue': _safe(info.get('totalRevenue')), 'eps': _safe(info.get('trailingEps') or info.get('forwardEps')),
            'eps_surprise': eps_surprise, 'revenue_surprise': np.nan, 'forward_pe': _safe(info.get('forwardPE')),
            'market_cap': _safe(info.get('marketCap')), 'free_cash_flow': _safe(info.get('freeCashflow')),
            'gross_margin': _safe(info.get('grossMargins')), 'operating_margin': _safe(info.get('operatingMargins')),
            'earnings_growth': _safe(info.get('earningsGrowth')), 'revenue_growth': _safe(info.get('revenueGrowth')),
            'debt_to_equity': _safe(info.get('debtToEquity')), 'current_ratio': _safe(info.get('currentRatio')),
            'return_on_equity': _safe(info.get('returnOnEquity')), 'beta': _safe(info.get('beta')),
            'short_percent_float': _safe(info.get('shortPercentOfFloat')),
            'held_percent_institutions': _safe(info.get('heldPercentInstitutions')),
            'recommendation_mean': _safe(info.get('recommendationMean')),
            'analyst_target_mean': _safe(info.get('targetMeanPrice')),
            'analyst_count': _safe(info.get('numberOfAnalystOpinions')),
            'next_earnings_date': next_earnings, 'days_to_earnings': _days_to(next_earnings) if next_earnings else np.nan,
            'provider': 'yfinance',
        }

    def _from_fmp(self, ticker: str) -> dict:
        if not settings.fmp_api_key: return {}
        try:
            profile = requests.get('https://financialmodelingprep.com/stable/profile', params={'symbol': ticker, 'apikey': settings.fmp_api_key}, timeout=settings.request_timeout).json()
            ratios = requests.get('https://financialmodelingprep.com/stable/ratios-ttm', params={'symbol': ticker, 'apikey': settings.fmp_api_key}, timeout=settings.request_timeout).json()
            income = requests.get('https://financialmodelingprep.com/stable/income-statement', params={'symbol': ticker, 'limit': 2, 'apikey': settings.fmp_api_key}, timeout=settings.request_timeout).json()
            cash = requests.get('https://financialmodelingprep.com/stable/cash-flow-statement', params={'symbol': ticker, 'limit': 1, 'apikey': settings.fmp_api_key}, timeout=settings.request_timeout).json()
            p = profile[0] if isinstance(profile, list) and profile else {}; r = ratios[0] if isinstance(ratios, list) and ratios else {}
            i0 = income[0] if isinstance(income, list) and income else {}; i1 = income[1] if isinstance(income, list) and len(income) > 1 else {}; c0 = cash[0] if isinstance(cash, list) and cash else {}
            rev0, rev1 = _safe(i0.get('revenue')), _safe(i1.get('revenue'))
            growth = (rev0 / rev1 - 1) if np.isfinite(rev0) and np.isfinite(rev1) and rev1 else np.nan
            return {
                'revenue': rev0, 'eps': _safe(i0.get('eps')),
                'forward_pe': _safe(r.get('priceToEarningsRatioTTM') or r.get('priceEarningsRatioTTM')),
                'market_cap': _safe(p.get('marketCap')), 'free_cash_flow': _safe(c0.get('freeCashFlow')),
                'gross_margin': _safe(r.get('grossProfitMarginTTM')), 'operating_margin': _safe(r.get('operatingProfitMarginTTM')),
                'revenue_growth': growth, 'debt_to_equity': _safe(r.get('debtEquityRatioTTM')),
                'current_ratio': _safe(r.get('currentRatioTTM')), 'return_on_equity': _safe(r.get('returnOnEquityTTM')),
                'beta': _safe(p.get('beta')), 'provider': 'fmp',
            }
        except Exception as e:
            logger.warning('FMP fundamentals failed for %s: %s', ticker, e); return {}

    def get(self, ticker: str) -> dict:
        base = self._from_yfinance(ticker.upper()); fmp = self._from_fmp(ticker.upper())
        for k, v in fmp.items():
            if k == 'provider': continue
            if isinstance(v, (int, float, np.number)) and np.isfinite(v): base[k] = v
            elif v not in (None, '') and not isinstance(v, (int, float, np.number)): base[k] = v
        if fmp: base['provider'] = 'fmp+yfinance'
        for f in FIELDS: base.setdefault(f, np.nan)
        return base
