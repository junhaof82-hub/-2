from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df['high'], df['low'], df['close']
    up = high.diff(); down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = atr(df, 1)
    tr_s = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df['high'] + df['low'] + df['close']) / 3
    flow = typical * df['volume'].fillna(0)
    direction = typical.diff()
    pos = flow.where(direction > 0, 0.0).rolling(period, min_periods=max(3, period//3)).sum()
    neg = flow.where(direction < 0, 0.0).rolling(period, min_periods=max(3, period//3)).sum().abs()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_index()
    c = pd.to_numeric(out['close'], errors='coerce')
    o = pd.to_numeric(out['open'], errors='coerce')
    h = pd.to_numeric(out['high'], errors='coerce')
    l = pd.to_numeric(out['low'], errors='coerce')
    v = pd.to_numeric(out['volume'], errors='coerce').fillna(0)

    for n in (1, 2, 3, 5, 10, 20, 60):
        out[f'return_{n}d'] = c.pct_change(n)
    out['log_return'] = np.log(c / c.shift(1))
    for n in (5, 10, 20, 60):
        out[f'volatility_{n}'] = out['log_return'].rolling(n, min_periods=max(3, n//3)).std() * np.sqrt(252)
    out['rsi_14'] = rsi(c, 14)
    out['rsi_5'] = rsi(c, 5)

    for n in (9, 20, 50, 100, 200):
        out[f'ema_{n}'] = c.ewm(span=n, adjust=False).mean()
    for n in (20, 50, 100, 200):
        out[f'sma_{n}'] = c.rolling(n, min_periods=max(2, n // 4)).mean()

    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    out['macd'] = ema12 - ema26
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']
    out['macd_hist_change'] = out['macd_hist'].diff()

    mid = c.rolling(20, min_periods=5).mean(); std = c.rolling(20, min_periods=5).std()
    out['bb_mid'] = mid; out['bb_upper'] = mid + 2 * std; out['bb_lower'] = mid - 2 * std
    out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / mid.replace(0, np.nan)
    out['bb_position'] = (c - out['bb_lower']) / (out['bb_upper'] - out['bb_lower']).replace(0, np.nan)
    out['atr_14'] = atr(out, 14)

    adx14, plus_di, minus_di = adx(out, 14)
    out['adx_14'] = adx14; out['plus_di_14'] = plus_di; out['minus_di_14'] = minus_di
    out['di_spread'] = (plus_di - minus_di) / 100

    low14 = l.rolling(14, min_periods=5).min(); high14 = h.rolling(14, min_periods=5).max()
    out['stoch_k'] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    out['stoch_d'] = out['stoch_k'].rolling(3, min_periods=1).mean()
    out['mfi_14'] = mfi(out, 14)

    typical = (h + l + c) / 3
    idx = pd.DatetimeIndex(out.index)
    if len(out) and len(idx) > 1 and (idx.to_series().diff().dropna().median() < pd.Timedelta(hours=20)):
        session = pd.Series(idx.date, index=out.index)
        pv = typical * v
        out['vwap'] = pv.groupby(session).cumsum() / v.groupby(session).cumsum().replace(0, np.nan)
    else:
        out['vwap'] = (typical * v).rolling(20, min_periods=1).sum() / v.rolling(20, min_periods=1).sum().replace(0, np.nan)

    out['volume_sma_20'] = v.rolling(20, min_periods=2).mean()
    out['volume_ratio'] = v / out['volume_sma_20'].replace(0, np.nan)
    volume_std = v.rolling(20, min_periods=5).std().replace(0, np.nan)
    out['volume_z20'] = (v - out['volume_sma_20']) / volume_std
    signed = np.sign(c.diff()).fillna(0)
    out['obv'] = (signed * v).cumsum()
    out['obv_slope_10'] = out['obv'].pct_change(10).replace([np.inf, -np.inf], np.nan)

    out['momentum_10'] = c - c.shift(10)
    out['roc_10'] = c.pct_change(10) * 100
    for n in (20, 50):
        out[f'support_{n}'] = l.rolling(n, min_periods=max(5, n//4)).min()
        out[f'resistance_{n}'] = h.rolling(n, min_periods=max(5, n//4)).max()
        out[f'position_{n}'] = (c - out[f'support_{n}']) / (out[f'resistance_{n}'] - out[f'support_{n}']).replace(0, np.nan)

    for n in (9, 20, 50, 100, 200):
        out[f'close_vs_ema_{n}'] = c / out[f'ema_{n}'] - 1
    out['ema_20_slope_5'] = out['ema_20'].pct_change(5)
    out['ema_50_slope_10'] = out['ema_50'].pct_change(10)
    out['ema_200_slope_20'] = out['ema_200'].pct_change(20)
    out['atr_pct'] = out['atr_14'] / c.replace(0, np.nan)
    out['dist_support_20'] = c / out['support_20'] - 1
    out['dist_resistance_20'] = c / out['resistance_20'] - 1
    out['gap_pct'] = o / c.shift(1) - 1
    out['intraday_range_pct'] = (h - l) / c.shift(1).replace(0, np.nan)
    out['close_location'] = (c - l) / (h - l).replace(0, np.nan)
    out['vwap_distance'] = c / out['vwap'].replace(0, np.nan) - 1
    return out.replace([np.inf, -np.inf], np.nan)


def technical_score(latest: pd.Series) -> tuple[float, list[str], list[str]]:
    score = 50.0; bullish: list[str] = []; bearish: list[str] = []
    r = float(latest.get('rsi_14', 50) if pd.notna(latest.get('rsi_14', 50)) else 50)
    if 52 < r < 70: score += 7; bullish.append('RSI positive')
    elif r >= 76: score -= 7; bearish.append('RSI overbought')
    elif r < 35: score -= 6; bearish.append('RSI weak')
    if float(latest.get('macd_hist', 0) or 0) > 0:
        score += 7; bullish.append('MACD bullish')
    else:
        score -= 5; bearish.append('MACD bearish')
    if float(latest.get('macd_hist_change', 0) or 0) > 0:
        score += 3; bullish.append('MACD momentum improving')
    for n, pts in ((20, 7), (50, 6), (200, 7)):
        if float(latest.get(f'close_vs_ema_{n}', 0) or 0) > 0:
            score += pts; bullish.append(f'Above EMA {n}')
        else:
            score -= pts; bearish.append(f'Below EMA {n}')
    adxv = float(latest.get('adx_14', 0) or 0); di = float(latest.get('di_spread', 0) or 0)
    if adxv > 25 and di > 0: score += 5; bullish.append('Strong positive trend (ADX)')
    elif adxv > 25 and di < 0: score -= 5; bearish.append('Strong negative trend (ADX)')
    if float(latest.get('volume_ratio', 1) or 1) > 1.4 and float(latest.get('return_1d', 0) or 0) > 0:
        score += 5; bullish.append('High-volume advance')
    if float(latest.get('dist_resistance_20', -1) or -1) > -0.01:
        score -= 2; bearish.append('Near resistance')
    if float(latest.get('vwap_distance', 0) or 0) > 0:
        score += 2; bullish.append('Above VWAP')
    else:
        score -= 2; bearish.append('Below VWAP')
    return float(np.clip(score, 0, 100)), list(dict.fromkeys(bullish))[:8], list(dict.fromkeys(bearish))[:8]
