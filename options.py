from __future__ import annotations

from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
from logging_config import get_logger

logger = get_logger(__name__)


class OptionsService:
    """Short-horizon options sentiment from yfinance. Gracefully returns unavailable data."""

    def get(self, ticker: str) -> dict:
        from config import settings
        ticker = ticker.upper()
        empty = {
            'available': False, 'expiry': None, 'days_to_expiry': None,
            'put_call_volume': np.nan, 'put_call_open_interest': np.nan,
            'atm_iv': np.nan, 'implied_move_pct': np.nan, 'options_score': 50.0,
        }
        if settings.cloud_safe_mode and not settings.enable_options:
            return empty
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            expiries = list(t.options or [])
            if not expiries:
                return empty
            today = datetime.now(timezone.utc).date()
            expiry = min(expiries, key=lambda x: abs((datetime.fromisoformat(x).date() - today).days))
            dte = max(1, (datetime.fromisoformat(expiry).date() - today).days)
            chain = t.option_chain(expiry)
            calls, puts = chain.calls.copy(), chain.puts.copy()
            hist = t.history(period='5d', interval='1d', auto_adjust=False)
            spot = float(pd.to_numeric(hist['Close'], errors='coerce').dropna().iloc[-1])

            call_vol = float(pd.to_numeric(calls.get('volume', 0), errors='coerce').fillna(0).sum())
            put_vol = float(pd.to_numeric(puts.get('volume', 0), errors='coerce').fillna(0).sum())
            call_oi = float(pd.to_numeric(calls.get('openInterest', 0), errors='coerce').fillna(0).sum())
            put_oi = float(pd.to_numeric(puts.get('openInterest', 0), errors='coerce').fillna(0).sum())
            pcv = put_vol / call_vol if call_vol > 0 else np.nan
            pcoi = put_oi / call_oi if call_oi > 0 else np.nan

            ivs = []
            for df in (calls, puts):
                if 'strike' not in df or 'impliedVolatility' not in df:
                    continue
                d = df.copy()
                d['distance'] = (pd.to_numeric(d['strike'], errors='coerce') / spot - 1).abs()
                near = d.nsmallest(min(4, len(d)), 'distance')
                vals = pd.to_numeric(near['impliedVolatility'], errors='coerce').dropna()
                ivs.extend(vals.tolist())
            atm_iv = float(np.median(ivs)) if ivs else np.nan
            implied_move = float(atm_iv * math.sqrt(dte / 365)) if np.isfinite(atm_iv) else np.nan

            score = 50.0
            if np.isfinite(pcv):
                if pcv < 0.65: score += 10
                elif pcv > 1.35: score -= 10
            if np.isfinite(pcoi):
                if pcoi < 0.75: score += 6
                elif pcoi > 1.30: score -= 6
            if np.isfinite(implied_move) and implied_move > 0.12:
                score -= 4  # high uncertainty/event risk rather than direction
            return {
                'available': True, 'expiry': expiry, 'days_to_expiry': dte,
                'put_call_volume': pcv, 'put_call_open_interest': pcoi,
                'atm_iv': atm_iv, 'implied_move_pct': implied_move,
                'options_score': float(np.clip(score, 0, 100)),
            }
        except Exception as e:
            logger.warning('Options analysis failed for %s: %s', ticker, e)
            return empty
