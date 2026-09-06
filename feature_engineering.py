from __future__ import annotations

import numpy as np
import pandas as pd
from features.technical import add_technical_indicators

BASE_FEATURE_COLUMNS = [
    'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
    'rsi_5', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'macd_hist_change',
    'bb_width', 'bb_position', 'atr_pct', 'volume_ratio', 'volume_z20', 'obv_slope_10',
    'momentum_10', 'roc_10', 'close_vs_ema_9', 'close_vs_ema_20', 'close_vs_ema_50',
    'close_vs_ema_100', 'close_vs_ema_200', 'ema_20_slope_5', 'ema_50_slope_10',
    'ema_200_slope_20', 'dist_support_20', 'dist_resistance_20', 'position_20', 'position_50',
    'adx_14', 'di_spread', 'stoch_k', 'stoch_d', 'mfi_14', 'gap_pct',
    'intraday_range_pct', 'close_location', 'vwap_distance',
]
CONTEXT_FEATURE_COLUMNS = [
    'spy_return_1d', 'spy_return_5d', 'spy_return_20d',
    'qqq_return_1d', 'qqq_return_5d', 'qqq_return_20d',
    'vix_level', 'vix_change_1d', 'vix_change_5d',
    'tnx_level', 'tnx_change_5d', 'stock_vs_spy_5d', 'stock_vs_qqq_5d',
    'stock_vs_spy_20d', 'stock_vs_qqq_20d',
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS


def _align_context(base: pd.DataFrame, context_prices: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    if not context_prices:
        return base
    out = base.copy()
    # Align by calendar date, avoiding timezone / close-time mismatches across providers.
    base_dates = pd.Index(pd.DatetimeIndex(out.index).tz_convert('UTC').date if pd.DatetimeIndex(out.index).tz is not None else pd.DatetimeIndex(out.index).date)
    for symbol, prefix in [('SPY', 'spy'), ('QQQ', 'qqq')]:
        ctx = context_prices.get(symbol)
        if ctx is None or ctx.empty:
            continue
        c = pd.to_numeric(ctx['close'], errors='coerce')
        s = pd.DataFrame(index=pd.Index(pd.DatetimeIndex(ctx.index).date))
        s[f'{prefix}_return_1d'] = c.pct_change().to_numpy()
        s[f'{prefix}_return_5d'] = c.pct_change(5).to_numpy()
        s[f'{prefix}_return_20d'] = c.pct_change(20).to_numpy()
        for col in s.columns:
            mapping = s[col].groupby(level=0).last()
            out[col] = pd.Series(base_dates.map(mapping), index=out.index, dtype=float)
    vix = context_prices.get('^VIX')
    if vix is not None and not vix.empty:
        c = pd.to_numeric(vix['close'], errors='coerce')
        temp = pd.DataFrame({'vix_level': c, 'vix_change_1d': c.pct_change(), 'vix_change_5d': c.pct_change(5)}, index=pd.Index(pd.DatetimeIndex(vix.index).date))
        for col in temp:
            mapping = temp[col].groupby(level=0).last(); out[col] = pd.Series(base_dates.map(mapping), index=out.index, dtype=float)
    tnx = context_prices.get('^TNX')
    if tnx is not None and not tnx.empty:
        c = pd.to_numeric(tnx['close'], errors='coerce')
        temp = pd.DataFrame({'tnx_level': c, 'tnx_change_5d': c.pct_change(5)}, index=pd.Index(pd.DatetimeIndex(tnx.index).date))
        for col in temp:
            mapping = temp[col].groupby(level=0).last(); out[col] = pd.Series(base_dates.map(mapping), index=out.index, dtype=float)
    if 'spy_return_5d' in out:
        out['stock_vs_spy_5d'] = out['return_5d'] - out['spy_return_5d']
        out['stock_vs_spy_20d'] = out['return_20d'] - out.get('spy_return_20d')
    if 'qqq_return_5d' in out:
        out['stock_vs_qqq_5d'] = out['return_5d'] - out['qqq_return_5d']
        out['stock_vs_qqq_20d'] = out['return_20d'] - out.get('qqq_return_20d')
    return out


def build_feature_frame(prices: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5), context_prices: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    out = add_technical_indicators(prices)
    out = _align_context(out, context_prices)
    for h in horizons:
        future_return = out['close'].shift(-h) / out['close'] - 1
        out[f'target_return_{h}d'] = future_return
        out[f'target_up_{h}d'] = (future_return > 0).astype(float)
        out.loc[future_return.isna(), f'target_up_{h}d'] = np.nan
    return out


def available_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in frame.columns and frame[c].notna().sum() >= max(20, int(len(frame) * 0.25))]


def clean_training_data(frame: pd.DataFrame, horizon: int, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_columns or available_feature_columns(frame)
    target = f'target_up_{horizon}d'
    data = frame[cols + [target]].replace([np.inf, -np.inf], np.nan)
    # Keep rows even with some missing features; model pipelines impute. Require target and at least 60% feature coverage.
    min_non_na = max(1, int(len(cols) * .60))
    data = data[data[cols].notna().sum(axis=1) >= min_non_na].dropna(subset=[target])
    return data[cols].astype(float), data[target].astype(int)


def latest_features(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    cols = feature_columns or available_feature_columns(frame)
    safe = frame.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).ffill()
    row = safe.iloc[[-1]].copy()
    return row.astype(float)
