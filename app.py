from __future__ import annotations

# Stock AI MAX ULTRA v3 — single-file Streamlit Cloud build.
# Design goal: maximum useful coverage while failing soft instead of blank-screening.

import os

# Keep Community Cloud from spawning too many BLAS / tree threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import io
import json
import math
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_VERSION = "4.0.0"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

st.set_page_config(
    page_title="Stock AI MAX ULTRA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container{padding-top:1.0rem;max-width:1580px}
.hero{padding:18px 22px;border:1px solid rgba(128,128,128,.28);border-radius:18px;margin-bottom:14px;background:rgba(90,90,110,.05)}
.small{font-size:.86rem;opacity:.76}.tiny{font-size:.78rem;opacity:.68}
.pill{display:inline-block;padding:3px 9px;border:1px solid rgba(128,128,128,.30);border-radius:999px;margin:2px}
.good{color:#19c37d}.warn{color:#f5a623}.bad{color:#ff4b4b}
[data-testid="stMetricValue"]{font-size:1.48rem}
[data-testid="stMetricDelta"]{font-size:.82rem}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Configuration / secrets
# -----------------------------------------------------------------------------

def _secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, None)
        if v is not None:
            return str(v).strip()
    except Exception:
        pass
    return str(os.getenv(name, default) or "").strip()


def api_keys() -> dict[str, str]:
    massive = _secret("MASSIVE_API_KEY") or _secret("POLYGON_API_KEY")
    return {
        "massive": massive,
        "alpha": _secret("ALPHA_VANTAGE_API_KEY"),
        "finnhub": _secret("FINNHUB_API_KEY"),
        "marketaux": _secret("MARKETAUX_API_KEY"),
        "fred": _secret("FRED_API_KEY"),
        "sec_agent": _secret("SEC_USER_AGENT", "StockAIMAX research-app contact@example.com"),
        "openai": _secret("OPENAI_API_KEY"),
        "openai_model": _secret("OPENAI_MODEL", "gpt-5"),
    }


# -----------------------------------------------------------------------------
# Network safety: timeouts, retry caps, isolation
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        status=1,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=6, pool_maxsize=6)
    s.mount("https://", adapter)
    s.headers.update({"Accept": "application/json", "User-Agent": "StockAIMAX/4.0"})
    return s


def safe_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: tuple[float, float] = (3.5, 8.0),
) -> tuple[Any | None, str | None]:
    try:
        r = http_session().get(url, params=params or {}, headers=headers or {}, timeout=timeout)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def safe_text(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: tuple[float, float] = (3.5, 8.0),
) -> tuple[str | None, str | None]:
    try:
        r = http_session().get(url, params=params or {}, headers=headers or {}, timeout=timeout)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return r.text, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def timestamp_age_seconds(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        x = float(v)
        # Normalize ns/ms/s epoch values.
        if x > 1e17: x /= 1e9
        elif x > 1e14: x /= 1e6
        elif x > 1e11: x /= 1e3
        return max(0.0, datetime.now(UTC).timestamp() - x)
    except Exception:
        return None


def safe_call(name: str, fn: Callable[[], Any]) -> tuple[str, Any | None, str | None, float]:
    t0 = time.perf_counter()
    try:
        out = fn()
        return name, out, None, time.perf_counter() - t0
    except Exception as e:
        return name, None, f"{type(e).__name__}: {e}", time.perf_counter() - t0


# -----------------------------------------------------------------------------
# Market data
# -----------------------------------------------------------------------------

def _clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
    ren = {c: str(c).title() for c in x.columns}
    x = x.rename(columns=ren)
    wanted = [c for c in ["Open", "High", "Low", "Close", "Volume", "Vwap"] if c in x.columns]
    if "Close" not in wanted:
        return pd.DataFrame()
    x = x[wanted].copy()
    for c in wanted:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"]).sort_index()
    return x[~x.index.duplicated(keep="last")]


@st.cache_data(ttl=600, max_entries=128, show_spinner=False)
def yf_history(ticker: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    try:
        # Ticker.history handles pre/post more consistently than download.
        x = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=prepost,
            timeout=8,
        )
        return _clean_ohlcv(x)
    except TypeError:
        # Older/newer yfinance versions can differ in timeout handling.
        try:
            x = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=False, actions=False, prepost=prepost
            )
            return _clean_ohlcv(x)
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, max_entries=24, show_spinner=False)
def yf_multi_daily(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(
            list(tickers),
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        for t in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0):
                    out[t] = _clean_ohlcv(raw[t])
                elif len(tickers) == 1:
                    out[t] = _clean_ohlcv(raw)
                else:
                    out[t] = pd.DataFrame()
            except Exception:
                out[t] = pd.DataFrame()
    except Exception:
        pass
    for t in tickers:
        if t not in out or out[t].empty:
            out[t] = yf_history(t, period, "1d", False)
    return out


def _massive_host_urls(path: str) -> list[str]:
    return [f"https://api.massive.com{path}", f"https://api.polygon.io{path}"]


@st.cache_data(ttl=20, max_entries=64, show_spinner=False)
def massive_snapshot(ticker: str, api_key: str) -> dict[str, Any]:
    """Prefer Massive unified snapshot (includes session/extended-hours fields), then v2 fallback."""
    if not api_key:
        return {}
    # New unified endpoint.
    for host in ("https://api.massive.com", "https://api.polygon.io"):
        data, err = safe_json(f"{host}/v3/snapshot", params={"ticker": ticker.upper(), "apiKey": api_key}, timeout=(3, 6))
        if not err and isinstance(data, dict) and data.get("results"):
            r = data["results"][0] or {}
            sess = r.get("session") or {}
            trade = r.get("last_trade") or {}
            quote = r.get("last_quote") or {}
            price = trade.get("price") or sess.get("price") or sess.get("close")
            if price is not None:
                updated = trade.get("last_updated") or sess.get("last_updated") or quote.get("last_updated")
                return {
                    "price": float(price),
                    "change_pct": float(sess.get("change_percent", np.nan)),
                    "bid": quote.get("bid"), "ask": quote.get("ask"),
                    "volume": sess.get("volume"), "vwap": sess.get("vwap"),
                    "market_status": r.get("market_status"),
                    "regular_change_pct": sess.get("regular_trading_change_percent"),
                    "early_change_pct": sess.get("early_trading_change_percent"),
                    "late_change_pct": sess.get("late_trading_change_percent"),
                    "updated": updated,
                    "source": "Massive Unified Snapshot",
                    "raw": r,
                }
    # Legacy v2 snapshot fallback.
    path = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
    for url in _massive_host_urls(path):
        data, err = safe_json(url, params={"apiKey": api_key}, timeout=(3, 6))
        if not err and isinstance(data, dict) and data.get("ticker"):
            t = data["ticker"]
            lt = t.get("lastTrade") or {}
            lq = t.get("lastQuote") or {}
            day = t.get("day") or {}
            price = lt.get("p") or day.get("c") or lq.get("P") or lq.get("p")
            return {
                "price": float(price) if price is not None else np.nan,
                "change_pct": float(t.get("todaysChangePerc", np.nan)),
                "bid": lq.get("p"), "ask": lq.get("P"), "volume": day.get("v"),
                "updated": t.get("updated") or data.get("updated"),
                "source": "Massive/Polygon Snapshot", "raw": t,
            }
    return {}


@st.cache_data(ttl=20, max_entries=64, show_spinner=False)
def finnhub_quote(ticker: str, api_key: str) -> dict[str, Any]:
    if not api_key:
        return {}
    data, err = safe_json(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": ticker.upper(), "token": api_key},
        timeout=(3, 6),
    )
    if err or not isinstance(data, dict) or not data.get("c"):
        return {}
    return {
        "price": float(data.get("c")),
        "change_pct": float(data.get("dp", np.nan)),
        "high": data.get("h"),
        "low": data.get("l"),
        "open": data.get("o"),
        "prev_close": data.get("pc"),
        "updated": int(data.get("t") or 0) * 1_000_000_000,
        "source": "Finnhub Quote",
    }


@st.cache_data(ttl=20, max_entries=64, show_spinner=False)
def current_snapshot(ticker: str) -> dict[str, Any]:
    keys = api_keys()
    if keys["massive"]:
        x = massive_snapshot(ticker, keys["massive"])
        if x:
            return x
    if keys["finnhub"]:
        x = finnhub_quote(ticker, keys["finnhub"])
        if x:
            return x
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info
        price = (fi.get("last_price") or fi.get("lastPrice")) if hasattr(fi, "get") else getattr(fi, "last_price", None)
        prev = (fi.get("previous_close") or fi.get("previousClose")) if hasattr(fi, "get") else getattr(fi, "previous_close", None)
        if price:
            dp = ((float(price) / float(prev)) - 1) * 100 if prev else np.nan
            return {"price": float(price), "change_pct": dp, "source": "Yahoo Finance", "updated": None}
    except Exception:
        pass
    d = yf_history(ticker, "5d", "1d", False)
    if not d.empty:
        p = float(d.Close.iloc[-1])
        prev = float(d.Close.iloc[-2]) if len(d) > 1 else np.nan
        return {"price": p, "change_pct": (p / prev - 1) * 100 if prev else np.nan, "source": "Yahoo Finance EOD", "updated": None}
    return {}


@st.cache_data(ttl=300, max_entries=128, show_spinner=False)
def massive_bars(ticker: str, interval: str, api_key: str) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    spec = {
        "5m": (5, "minute", 10),
        "15m": (15, "minute", 20),
        "1h": (1, "hour", 90),
    }
    if interval not in spec:
        return pd.DataFrame()
    mult, span, days = spec[interval]
    end = datetime.now(ET).date()
    start = end - timedelta(days=days)
    path = f"/v2/aggs/ticker/{ticker.upper()}/range/{mult}/{span}/{start.isoformat()}/{end.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}
    for url in _massive_host_urls(path):
        data, err = safe_json(url, params=params, timeout=(3, 7))
        if err or not isinstance(data, dict) or not data.get("results"):
            continue
        r = pd.DataFrame(data["results"])
        if r.empty:
            continue
        idx = pd.to_datetime(r["t"], unit="ms", utc=True).dt.tz_convert(ET)
        out = pd.DataFrame(
            {
                "Open": pd.to_numeric(r.get("o"), errors="coerce").to_numpy(),
                "High": pd.to_numeric(r.get("h"), errors="coerce").to_numpy(),
                "Low": pd.to_numeric(r.get("l"), errors="coerce").to_numpy(),
                "Close": pd.to_numeric(r.get("c"), errors="coerce").to_numpy(),
                "Volume": pd.to_numeric(r.get("v"), errors="coerce").to_numpy(),
                "Vwap": pd.to_numeric(r.get("vw"), errors="coerce").to_numpy() if "vw" in r else np.full(len(r), np.nan),
            },
            index=pd.DatetimeIndex(idx.to_numpy()),
        )
        return _clean_ohlcv(out)
    return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=128, show_spinner=False)
def intraday_bars(ticker: str, interval: str) -> tuple[pd.DataFrame, str]:
    keys = api_keys()
    if keys["massive"]:
        d = massive_bars(ticker, interval, keys["massive"])
        if not d.empty:
            return d, "Massive/Polygon"
    period = {"5m": "5d", "15m": "10d", "1h": "3mo"}.get(interval, "5d")
    d = yf_history(ticker, period, interval, True)
    return d, "Yahoo Finance"


# -----------------------------------------------------------------------------
# Technical indicators and features
# -----------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 3)).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["Close"].astype(float)
    h = x["High"].astype(float) if "High" in x else c
    l = x["Low"].astype(float) if "Low" in x else c
    v = x["Volume"].astype(float) if "Volume" in x else pd.Series(np.nan, index=x.index)
    ret = c.pct_change()
    x["ret1"] = ret
    for n in (2, 3, 5, 10, 20, 60):
        x[f"ret{n}"] = c.pct_change(n)
    for n in (9, 20, 50, 100, 200):
        x[f"ema{n}"] = ema(c, n)
    for n in (20, 50, 200):
        x[f"sma{n}"] = sma(c, n)

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1 / 14, adjust=False).mean() / loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    x["rsi14"] = 100 - 100 / (1 + rs)

    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    x["macd"] = macd
    x["macd_signal"] = sig
    x["macd_hist"] = macd - sig

    mid = sma(c, 20)
    std = c.rolling(20).std()
    x["bb_mid"] = mid
    x["bb_up"] = mid + 2 * std
    x["bb_low"] = mid - 2 * std
    x["bb_pos"] = (c - x["bb_low"]) / (x["bb_up"] - x["bb_low"]).replace(0, np.nan)
    x["bb_width"] = (x["bb_up"] - x["bb_low"]) / mid.replace(0, np.nan)

    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    x["atr_pct"] = x["atr14"] / c

    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    x["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    x["di_spread"] = (plus_di - minus_di) / 100

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    x["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    x["stoch_d"] = x["stoch_k"].rolling(3).mean()

    tp = (h + l + c) / 3
    if v.notna().any():
        x["vwap20"] = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        x["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
        x["volume_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
        signed = np.sign(c.diff()).fillna(0) * v.fillna(0)
        x["obv"] = signed.cumsum()
        x["obv_trend"] = x["obv"].pct_change(20).replace([np.inf, -np.inf], np.nan)
        pos_money = (tp * v).where(tp.diff() > 0, 0.0)
        neg_money = (tp * v).where(tp.diff() < 0, 0.0).abs()
        mfr = pos_money.rolling(14).sum() / neg_money.rolling(14).sum().replace(0, np.nan)
        x["mfi14"] = 100 - 100 / (1 + mfr)
    else:
        for col in ("vwap20", "volume_ratio", "volume_z", "obv", "obv_trend", "mfi14"):
            x[col] = np.nan

    x["roc10"] = c.pct_change(10) * 100
    x["volatility10"] = ret.rolling(10).std() * np.sqrt(252)
    x["volatility20"] = ret.rolling(20).std() * np.sqrt(252)
    x["volatility60"] = ret.rolling(60).std() * np.sqrt(252)
    x["support20"] = l.rolling(20).min()
    x["resistance20"] = h.rolling(20).max()
    x["support60"] = l.rolling(60).min()
    x["resistance60"] = h.rolling(60).max()
    x["range_pos20"] = (c - x["support20"]) / (x["resistance20"] - x["support20"]).replace(0, np.nan)
    x["range_pos60"] = (c - x["support60"]) / (x["resistance60"] - x["support60"]).replace(0, np.nan)
    for n in (20, 50, 200):
        x[f"dist_ema{n}"] = c / x[f"ema{n}"] - 1
    x["trend20"] = x["ema20"].pct_change(10)
    x["trend50"] = x["ema50"].pct_change(20)
    x["gap"] = (x["Open"] / c.shift(1) - 1) if "Open" in x else np.nan
    return x.replace([np.inf, -np.inf], np.nan)


BASE_FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ret60",
    "rsi14", "macd_hist", "bb_pos", "bb_width", "atr_pct", "adx14", "di_spread",
    "stoch_k", "volume_ratio", "volume_z", "obv_trend", "mfi14", "roc10",
    "volatility10", "volatility20", "volatility60", "range_pos20", "range_pos60",
    "dist_ema20", "dist_ema50", "dist_ema200", "trend20", "trend50", "gap",
]

MARKET_FEATURES = [
    "spy_ret1", "spy_ret5", "spy_ret20", "qqq_ret1", "qqq_ret5", "qqq_ret20",
    "smh_ret5", "smh_ret20", "vix_level", "vix_chg5", "tnx_level", "tnx_chg5",
    "rel_spy20", "rel_qqq20", "rel_smh20", "corr_qqq60",
]

FEATURES = BASE_FEATURES + MARKET_FEATURES


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def build_daily_feature_frame(ticker: str, period: str = "3y") -> tuple[pd.DataFrame, dict[str, Any]]:
    stock = yf_history(ticker, period, "1d", False)
    if stock.empty:
        return pd.DataFrame(), {"error": "no stock history"}
    ind = add_indicators(stock)
    ctx = yf_multi_daily(("SPY", "QQQ", "SMH", "^VIX", "^TNX"), period)

    def sclose(sym: str) -> pd.Series:
        d = ctx.get(sym, pd.DataFrame())
        if d.empty:
            return pd.Series(index=ind.index, dtype=float)
        s = d["Close"].copy()
        try:
            if getattr(s.index, "tz", None) is not None and getattr(ind.index, "tz", None) is None:
                s.index = s.index.tz_localize(None)
            elif getattr(s.index, "tz", None) is None and getattr(ind.index, "tz", None) is not None:
                s.index = s.index.tz_localize(ind.index.tz)
        except Exception:
            pass
        return s.reindex(ind.index, method="ffill")

    spy, qqq, smh, vix, tnx = [sclose(s) for s in ("SPY", "QQQ", "SMH", "^VIX", "^TNX")]
    for name, s in (("spy", spy), ("qqq", qqq), ("smh", smh)):
        ind[f"{name}_ret1"] = s.pct_change()
        ind[f"{name}_ret5"] = s.pct_change(5)
        ind[f"{name}_ret20"] = s.pct_change(20)
    ind["vix_level"] = vix
    ind["vix_chg5"] = vix.pct_change(5)
    ind["tnx_level"] = tnx
    ind["tnx_chg5"] = tnx.pct_change(5)
    ind["rel_spy20"] = ind["ret20"] - ind["spy_ret20"]
    ind["rel_qqq20"] = ind["ret20"] - ind["qqq_ret20"]
    ind["rel_smh20"] = ind["ret20"] - ind["smh_ret20"]
    ind["corr_qqq60"] = ind["ret1"].rolling(60).corr(ind["qqq_ret1"])
    meta = {"stock_rows": len(stock), "market_sources": [k for k, v in ctx.items() if not v.empty]}
    return ind.replace([np.inf, -np.inf], np.nan), meta


def technical_score(ind: pd.DataFrame) -> tuple[float, list[str]]:
    if ind.empty:
        return 50.0, []
    x = ind.iloc[-1]
    s = 50.0
    notes: list[str] = []
    close = float(x.get("Close", np.nan))
    for n, pts in ((20, 7), (50, 6), (200, 5)):
        e = x.get(f"ema{n}")
        if pd.notna(e) and pd.notna(close):
            if close > e:
                s += pts
                notes.append(f"价格 > EMA{n}")
            else:
                s -= pts
    rsi = x.get("rsi14", 50)
    if pd.notna(rsi):
        if 52 <= rsi <= 68:
            s += 5
        elif rsi > 80:
            s -= 7
            notes.append("RSI 极度超买")
        elif rsi < 32:
            s += 3
            notes.append("RSI 超卖")
    if x.get("macd_hist", 0) > 0:
        s += 5
    else:
        s -= 4
    adx = x.get("adx14", 0)
    if pd.notna(adx) and adx > 25:
        s += 3 if x.get("di_spread", 0) > 0 else -3
    vr = x.get("volume_ratio")
    if pd.notna(vr) and vr > 1.5:
        s += 3 if x.get("ret1", 0) > 0 else -3
    return float(np.clip(s, 0, 100)), notes


# -----------------------------------------------------------------------------
# Intraday multi-timeframe signal
# -----------------------------------------------------------------------------

def intraday_signal_from_df(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or len(df) < 25:
        return {"score": 50.0, "trend": "No data", "rows": len(df)}
    x = add_indicators(df)
    z = x.iloc[-1]
    score = 50.0
    close = z.get("Close", np.nan)
    e9, e20 = z.get("ema9"), z.get("ema20")
    if pd.notna(close) and pd.notna(e9): score += 8 if close > e9 else -8
    if pd.notna(e9) and pd.notna(e20): score += 7 if e9 > e20 else -7
    rsi = z.get("rsi14")
    if pd.notna(rsi):
        score += 4 if 52 <= rsi <= 70 else (-5 if rsi > 82 else (3 if rsi < 30 else 0))
    score += 5 if z.get("macd_hist", 0) > 0 else -5
    vr = z.get("volume_ratio")
    if pd.notna(vr) and vr > 1.4:
        score += 4 if z.get("ret1", 0) > 0 else -4
    score = float(np.clip(score, 0, 100))
    trend = "Bullish" if score >= 60 else ("Bearish" if score <= 40 else "Neutral")
    return {
        "score": score,
        "trend": trend,
        "rows": len(df),
        "rsi": float(rsi) if pd.notna(rsi) else np.nan,
        "ret": float(z.get("ret1", np.nan)),
        "volume_ratio": float(vr) if pd.notna(vr) else np.nan,
        "last": float(close) if pd.notna(close) else np.nan,
    }


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def intraday_bundle(ticker: str) -> dict[str, Any]:
    specs = ["1h", "15m", "5m"]
    signals = {}
    # Deliberately sequential: Streamlit does not officially support arbitrary app threads.
    # Caches + strict network timeouts keep this bounded without ScriptRunContext risk.
    for iv in specs:
        try:
            d, src = intraday_bars(ticker, iv)
        except Exception:
            d, src = pd.DataFrame(), "failed"
        sig = intraday_signal_from_df(d)
        sig["source"] = src
        signals[iv] = sig
    valid = [(iv, signals[iv]["score"]) for iv in specs if signals[iv]["rows"] >= 25]
    weights = {"1h": 0.45, "15m": 0.35, "5m": 0.20}
    if valid:
        w = np.array([weights[iv] for iv, _ in valid], float)
        sc = float(np.average([v for _, v in valid], weights=w))
    else:
        sc = 50.0
    return {"score": sc, "signals": signals}


def market_session_status() -> str:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return "Weekend / Closed"
    mins = now.hour * 60 + now.minute
    if 4 * 60 <= mins < 9 * 60 + 30:
        return "Pre-market"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "Regular session"
    if 16 * 60 <= mins < 20 * 60:
        return "After-hours"
    return "Closed"


# -----------------------------------------------------------------------------
# News: Yahoo + Alpha Vantage + Finnhub + Marketaux + SEC filings
# -----------------------------------------------------------------------------

POS_WORDS = {
    "beat", "beats", "growth", "record", "upgrade", "surge", "strong", "profit", "bullish",
    "demand", "partnership", "approval", "buyback", "raises", "raised", "outperform", "accelerates",
    "expands", "launch", "wins", "optimistic", "guidance raised", "contract", "breakthrough",
}
NEG_WORDS = {
    "miss", "misses", "downgrade", "fall", "weak", "loss", "lawsuit", "probe", "investigation",
    "recall", "cuts", "warning", "bearish", "decline", "slump", "risk", "delay", "antitrust",
    "tariff", "fraud", "guidance cut", "offering", "dilution", "shortfall",
}
CATALYST_WORDS = {
    "earnings", "guidance", "ai", "artificial intelligence", "deal", "acquisition", "merger", "approval",
    "launch", "contract", "partnership", "buyback", "dividend", "upgrade", "analyst", "product",
}
RISK_WORDS = {
    "lawsuit", "probe", "investigation", "downgrade", "tariff", "antitrust", "recall", "warning",
    "debt", "valuation", "delay", "offering", "dilution", "fraud", "short seller",
}
HIGH_QUALITY_SOURCES = {"reuters", "bloomberg", "associated press", "wall street journal", "wsj", "sec", "company filing"}


def _parse_pub(v: Any) -> datetime | None:
    if v is None or v == "": return None
    if isinstance(v, (int, float)):
        try: return datetime.fromtimestamp(float(v), tz=UTC)
        except Exception: return None
    sv = str(v)
    if re.fullmatch(r"\d{8}T\d{4,6}", sv):
        try:
            fmt = "%Y%m%dT%H%M%S" if len(sv) == 15 else "%Y%m%dT%H%M"
            return datetime.strptime(sv, fmt).replace(tzinfo=UTC)
        except Exception:
            pass
    try:
        d = pd.to_datetime(v, utc=True)
        return d.to_pydatetime()
    except Exception:
        return None


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())[:180]


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def yahoo_news(ticker: str, limit: int = 40) -> list[dict[str, Any]]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        raw = []
    out = []
    for n in raw:
        c = n.get("content") if isinstance(n, dict) else None
        if isinstance(c, dict):
            title = c.get("title") or ""
            summary = c.get("summary") or c.get("description") or ""
            provider = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else "Yahoo Finance"
            url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else ""
            pub = c.get("pubDate") or c.get("displayTime")
        else:
            title = n.get("title", "") if isinstance(n, dict) else ""
            summary = n.get("summary", "") if isinstance(n, dict) else ""
            provider = n.get("publisher", "Yahoo Finance") if isinstance(n, dict) else "Yahoo Finance"
            url = n.get("link", "") if isinstance(n, dict) else ""
            pub = n.get("providerPublishTime") if isinstance(n, dict) else None
        if title:
            out.append({"title": title, "summary": summary, "provider": provider or "Yahoo Finance", "url": url or "", "published": pub, "native_sentiment": None, "source_family": "Yahoo"})
        if len(out) >= limit:
            break
    return out


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def alpha_news(ticker: str, api_key: str) -> list[dict[str, Any]]:
    if not api_key: return []
    since = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y%m%dT%H%M")
    data, err = safe_json(
        "https://www.alphavantage.co/query",
        params={"function": "NEWS_SENTIMENT", "tickers": ticker.upper(), "time_from": since, "sort": "LATEST", "limit": 100, "apikey": api_key},
        timeout=(3, 8),
    )
    if err or not isinstance(data, dict): return []
    out = []
    for n in data.get("feed", []) or []:
        sent = None
        for ts in n.get("ticker_sentiment", []) or []:
            if str(ts.get("ticker", "")).upper() == ticker.upper():
                try: sent = float(ts.get("ticker_sentiment_score"))
                except Exception: pass
        out.append({
            "title": n.get("title", ""), "summary": n.get("summary", ""), "provider": n.get("source", "Alpha Vantage"),
            "url": n.get("url", ""), "published": n.get("time_published"), "native_sentiment": sent,
            "source_family": "Alpha Vantage",
        })
    return out


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def finnhub_news(ticker: str, api_key: str) -> list[dict[str, Any]]:
    if not api_key: return []
    today = datetime.now(UTC).date()
    data, err = safe_json(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": ticker.upper(), "from": (today - timedelta(days=7)).isoformat(), "to": today.isoformat(), "token": api_key},
        timeout=(3, 8),
    )
    if err or not isinstance(data, list): return []
    return [
        {"title": n.get("headline", ""), "summary": n.get("summary", ""), "provider": n.get("source", "Finnhub"), "url": n.get("url", ""), "published": n.get("datetime"), "native_sentiment": None, "source_family": "Finnhub"}
        for n in data[:100]
        if n.get("headline")
    ]


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def marketaux_news(ticker: str, api_key: str) -> list[dict[str, Any]]:
    if not api_key: return []
    since = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M")
    data, err = safe_json(
        "https://api.marketaux.com/v1/news/all",
        params={"symbols": ticker.upper(), "filter_entities": "true", "language": "en", "published_after": since, "limit": 50, "api_token": api_key},
        timeout=(3, 8),
    )
    if err or not isinstance(data, dict): return []
    out = []
    for n in data.get("data", []) or []:
        sent = None
        for ent in n.get("entities", []) or []:
            if str(ent.get("symbol", "")).upper() == ticker.upper():
                try: sent = float(ent.get("sentiment_score"))
                except Exception: pass
        out.append({
            "title": n.get("title", ""), "summary": n.get("description") or n.get("snippet") or "",
            "provider": n.get("source", "Marketaux") if isinstance(n.get("source"), str) else (n.get("source") or {}).get("name", "Marketaux"),
            "url": n.get("url", ""), "published": n.get("published_at"), "native_sentiment": sent,
            "source_family": "Marketaux",
        })
    return out


@st.cache_data(ttl=86400, max_entries=1, show_spinner=False)
def sec_ticker_map(user_agent: str) -> dict[str, int]:
    data, err = safe_json(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=(3, 8),
    )
    if err or not isinstance(data, dict): return {}
    out = {}
    for row in data.values():
        try: out[str(row.get("ticker", "")).upper()] = int(row.get("cik_str"))
        except Exception: pass
    return out


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def sec_filings(ticker: str, user_agent: str) -> list[dict[str, Any]]:
    cik = sec_ticker_map(user_agent).get(ticker.upper())
    if not cik: return []
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    data, err = safe_json(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=(3, 8))
    if err or not isinstance(data, dict): return []
    recent = ((data.get("filings") or {}).get("recent") or {})
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    acc = recent.get("accessionNumber", []) or []
    docs = recent.get("primaryDocument", []) or []
    company = data.get("name", ticker)
    wanted = {"8-K", "10-Q", "10-K", "4", "13D", "13G", "S-3", "424B5", "8-K/A", "10-Q/A", "10-K/A"}
    out = []
    for i, form in enumerate(forms[:80]):
        if form not in wanted: continue
        filing_date = dates[i] if i < len(dates) else ""
        try:
            if filing_date and (datetime.now().date() - datetime.fromisoformat(filing_date).date()).days > 21: continue
        except Exception: pass
        accession = acc[i].replace("-", "") if i < len(acc) else ""
        doc = docs[i] if i < len(docs) else ""
        link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}" if accession and doc else ""
        label = {"8-K": "8-K material event", "10-Q": "Quarterly report", "10-K": "Annual report", "4": "Insider transaction", "13D": "13D ownership", "13G": "13G ownership", "S-3": "Shelf registration", "424B5": "Prospectus supplement"}.get(form, form)
        out.append({"title": f"SEC {form}: {company} — {label}", "summary": f"Official SEC filing dated {filing_date}.", "provider": "SEC EDGAR", "url": link, "published": filing_date, "native_sentiment": None, "source_family": "SEC", "form": form})
    return out


def score_news(items: list[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now(UTC)
    scored = []
    for it in items:
        title = str(it.get("title", ""))
        summary = str(it.get("summary", ""))
        text = (title + " " + summary).lower()
        toks = set(re.findall(r"[a-z]+", text))
        lex = (len(toks & POS_WORDS) - len(toks & NEG_WORDS)) * 18.0
        native = it.get("native_sentiment")
        if native is not None and pd.notna(native):
            # Both Alpha Vantage and Marketaux sentiment are roughly centered around zero.
            native100 = float(np.clip(float(native) * 100, -100, 100))
            sentiment = 0.65 * native100 + 0.35 * lex
        else:
            sentiment = lex
        sentiment = float(np.clip(sentiment, -100, 100))

        form = it.get("form", "")
        importance = 2.0
        importance += 1.7 * sum(1 for k in CATALYST_WORDS if k in text)
        importance += 1.5 * sum(1 for k in RISK_WORDS if k in text)
        if form in {"8-K", "10-Q", "10-K", "S-3", "424B5"}: importance += 4
        importance = float(np.clip(importance, 0, 10))

        pub = _parse_pub(it.get("published"))
        age_h = max(0.0, (now - pub).total_seconds() / 3600) if pub else 72.0
        recency = math.exp(-age_h / 72.0)
        provider = str(it.get("provider", "")).lower()
        quality = 1.15 if any(k in provider for k in HIGH_QUALITY_SOURCES) else 1.0
        weight = max(0.15, importance / 10) * (0.35 + 0.65 * recency) * quality
        scored.append({**it, "score": sentiment, "importance": importance, "age_hours": age_h, "weight": weight})

    if scored:
        w = np.array([s["weight"] for s in scored], float)
        vals = np.array([s["score"] for s in scored], float)
        overall = float(np.average(vals, weights=w))
    else:
        overall = 0.0
    cats = [k for k in CATALYST_WORDS if any(k in (str(s.get("title", "")) + " " + str(s.get("summary", ""))).lower() for s in scored)]
    risks = [k for k in RISK_WORDS if any(k in (str(s.get("title", "")) + " " + str(s.get("summary", ""))).lower() for s in scored)]
    return {"score": overall, "items": sorted(scored, key=lambda z: (z["age_hours"], -z["importance"])), "catalysts": cats[:10], "risks": risks[:10]}


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def news_radar(ticker: str) -> dict[str, Any]:
    keys = api_keys()
    tasks: list[tuple[str, Callable[[], Any]]] = [
        ("Yahoo", lambda: yahoo_news(ticker, 50)),
        ("SEC", lambda: sec_filings(ticker, keys["sec_agent"])),
    ]
    if keys["alpha"]: tasks.append(("Alpha Vantage", lambda: alpha_news(ticker, keys["alpha"])))
    if keys["finnhub"]: tasks.append(("Finnhub", lambda: finnhub_news(ticker, keys["finnhub"])))
    if keys["marketaux"]: tasks.append(("Marketaux", lambda: marketaux_news(ticker, keys["marketaux"])))

    items: list[dict[str, Any]] = []
    status = []
    for name, fn in tasks:
        name, out, err, elapsed = safe_call(name, fn)
        n = len(out) if isinstance(out, list) else 0
        status.append({"Provider": name, "Items": n, "Seconds": elapsed, "Status": "OK" if not err else err})
        if isinstance(out, list): items.extend(out)

    dedup: dict[str, dict[str, Any]] = {}
    for it in items:
        k = _norm_title(str(it.get("title", "")))
        if not k: continue
        if k not in dedup or len(str(it.get("summary", ""))) > len(str(dedup[k].get("summary", ""))):
            dedup[k] = it
    result = score_news(list(dedup.values()))
    result["provider_status"] = status
    result["provider_count"] = sum(1 for x in status if x["Items"] > 0)
    result["article_count"] = len(result["items"])
    return result


# -----------------------------------------------------------------------------
# Fundamentals / earnings / analyst context
# -----------------------------------------------------------------------------

@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def company_info(ticker: str) -> dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info or {}
        return {str(k): v for k, v in info.items()}
    except Exception:
        return {}


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def finnhub_recommendations(ticker: str, api_key: str) -> list[dict[str, Any]]:
    if not api_key: return []
    data, err = safe_json("https://finnhub.io/api/v1/stock/recommendation", params={"symbol": ticker.upper(), "token": api_key}, timeout=(3, 7))
    return data[:6] if not err and isinstance(data, list) else []


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def finnhub_earnings(ticker: str, api_key: str) -> list[dict[str, Any]]:
    if not api_key: return []
    today = datetime.now(UTC).date()
    data, err = safe_json(
        "https://finnhub.io/api/v1/calendar/earnings",
        params={"from": (today - timedelta(days=30)).isoformat(), "to": (today + timedelta(days=90)).isoformat(), "symbol": ticker.upper(), "token": api_key},
        timeout=(3, 7),
    )
    if err or not isinstance(data, dict): return []
    return data.get("earningsCalendar", []) or []


def fundamental_score(info: dict[str, Any]) -> tuple[float, list[str], dict[str, Any]]:
    s = 50.0
    notes = []
    fields = {
        "Revenue Growth": info.get("revenueGrowth"),
        "Earnings Growth": info.get("earningsGrowth"),
        "Gross Margin": info.get("grossMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Forward P/E": info.get("forwardPE"),
        "Market Cap": info.get("marketCap"),
        "Free Cash Flow": info.get("freeCashflow"),
        "Trailing EPS": info.get("trailingEps"),
        "Forward EPS": info.get("forwardEps"),
    }
    rg, eg, gm, om, pe, fcf = [info.get(k) for k in ("revenueGrowth", "earningsGrowth", "grossMargins", "operatingMargins", "forwardPE", "freeCashflow")]
    if isinstance(rg, (int, float)):
        s += float(np.clip(rg * 45, -12, 12)); notes.append(f"Revenue growth {rg*100:.1f}%")
    if isinstance(eg, (int, float)):
        s += float(np.clip(eg * 30, -10, 10)); notes.append(f"Earnings growth {eg*100:.1f}%")
    if isinstance(gm, (int, float)):
        s += float(np.clip((gm - 0.30) * 12, -4, 5))
    if isinstance(om, (int, float)):
        s += float(np.clip((om - 0.10) * 20, -5, 7)); notes.append(f"Operating margin {om*100:.1f}%")
    if isinstance(pe, (int, float)) and pe > 0:
        s += 4 if pe < 25 else (-6 if pe > 70 else 0); notes.append(f"Forward P/E {pe:.1f}")
    if isinstance(fcf, (int, float)):
        s += 4 if fcf > 0 else -6
    return float(np.clip(s, 0, 100)), notes, fields


# -----------------------------------------------------------------------------
# Institutional fundamentals / earnings / analyst / valuation intelligence
# -----------------------------------------------------------------------------

@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def alpha_overview(ticker: str, api_key: str) -> dict[str, Any]:
    if not api_key:
        return {}
    data, err = safe_json(
        "https://www.alphavantage.co/query",
        params={"function": "OVERVIEW", "symbol": ticker.upper(), "apikey": api_key},
        timeout=(3, 8),
    )
    return data if not err and isinstance(data, dict) and data.get("Symbol") else {}


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def alpha_earnings_history(ticker: str, api_key: str) -> list[dict[str, Any]]:
    if not api_key:
        return []
    data, err = safe_json(
        "https://www.alphavantage.co/query",
        params={"function": "EARNINGS", "symbol": ticker.upper(), "apikey": api_key},
        timeout=(3, 8),
    )
    if err or not isinstance(data, dict):
        return []
    return (data.get("quarterlyEarnings") or [])[:12]


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def finnhub_metrics(ticker: str, api_key: str) -> dict[str, Any]:
    if not api_key:
        return {}
    data, err = safe_json(
        "https://finnhub.io/api/v1/stock/metric",
        params={"symbol": ticker.upper(), "metric": "all", "token": api_key},
        timeout=(3, 8),
    )
    return data if not err and isinstance(data, dict) else {}


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def finnhub_price_target(ticker: str, api_key: str) -> dict[str, Any]:
    if not api_key:
        return {}
    data, err = safe_json(
        "https://finnhub.io/api/v1/stock/price-target",
        params={"symbol": ticker.upper(), "token": api_key},
        timeout=(3, 7),
    )
    return data if not err and isinstance(data, dict) else {}


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def finnhub_upgrades(ticker: str, api_key: str) -> list[dict[str, Any]]:
    if not api_key:
        return []
    today = datetime.now(UTC).date()
    data, err = safe_json(
        "https://finnhub.io/api/v1/stock/upgrade-downgrade",
        params={"symbol": ticker.upper(), "from": (today - timedelta(days=30)).isoformat(), "to": today.isoformat(), "token": api_key},
        timeout=(3, 7),
    )
    return data[:20] if not err and isinstance(data, list) else []


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def finnhub_peers(ticker: str, api_key: str) -> list[str]:
    if not api_key:
        return []
    data, err = safe_json(
        "https://finnhub.io/api/v1/stock/peers",
        params={"symbol": ticker.upper(), "grouping": "industry", "token": api_key},
        timeout=(3, 7),
    )
    return [str(x).upper() for x in data[:12]] if not err and isinstance(data, list) else []


def _num(*vals: Any) -> float | None:
    for v in vals:
        try:
            if v is None or v == "" or str(v).lower() in {"none", "nan", "-"}:
                continue
            x = float(v)
            if np.isfinite(x):
                return x
        except Exception:
            continue
    return None


def _metric_from_finnhub(bundle: dict[str, Any], *names: str) -> float | None:
    m = bundle.get("metric") if isinstance(bundle, dict) else {}
    if not isinstance(m, dict):
        return None
    return _num(*[m.get(n) for n in names])


def build_fundamental_bundle(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    keys = api_keys()
    alpha = alpha_overview(ticker, keys["alpha"]) if keys["alpha"] else {}
    fin = finnhub_metrics(ticker, keys["finnhub"]) if keys["finnhub"] else {}
    rec = finnhub_recommendations(ticker, keys["finnhub"]) if keys["finnhub"] else []
    cal = finnhub_earnings(ticker, keys["finnhub"]) if keys["finnhub"] else []
    pt = finnhub_price_target(ticker, keys["finnhub"]) if keys["finnhub"] else {}
    upgrades = finnhub_upgrades(ticker, keys["finnhub"]) if keys["finnhub"] else []
    hist = alpha_earnings_history(ticker, keys["alpha"]) if keys["alpha"] else []

    metrics = {
        "market_cap": _num(info.get("marketCap"), alpha.get("MarketCapitalization")),
        "enterprise_value": _num(info.get("enterpriseValue")),
        "revenue_growth": _num(info.get("revenueGrowth"), alpha.get("QuarterlyRevenueGrowthYOY")),
        "earnings_growth": _num(info.get("earningsGrowth"), info.get("earningsQuarterlyGrowth"), alpha.get("QuarterlyEarningsGrowthYOY")),
        "gross_margin": _num(info.get("grossMargins"), alpha.get("GrossProfitTTM") and None),
        "operating_margin": _num(info.get("operatingMargins"), alpha.get("OperatingMarginTTM")),
        "profit_margin": _num(info.get("profitMargins"), alpha.get("ProfitMargin")),
        "forward_pe": _num(info.get("forwardPE"), alpha.get("ForwardPE"), _metric_from_finnhub(fin, "peExclExtraAnnual")),
        "trailing_pe": _num(info.get("trailingPE"), alpha.get("PERatio"), _metric_from_finnhub(fin, "peTTM", "peBasicExclExtraTTM")),
        "price_to_sales": _num(info.get("priceToSalesTrailing12Months"), alpha.get("PriceToSalesRatioTTM"), _metric_from_finnhub(fin, "psTTM")),
        "price_to_book": _num(info.get("priceToBook"), alpha.get("PriceToBookRatio"), _metric_from_finnhub(fin, "pbQuarterly")),
        "ev_to_ebitda": _num(info.get("enterpriseToEbitda"), alpha.get("EVToEBITDA"), _metric_from_finnhub(fin, "evEbitdaTTM")),
        "free_cash_flow": _num(info.get("freeCashflow")),
        "operating_cash_flow": _num(info.get("operatingCashflow")),
        "total_cash": _num(info.get("totalCash")),
        "total_debt": _num(info.get("totalDebt")),
        "shares": _num(info.get("sharesOutstanding")),
        "forward_eps": _num(info.get("forwardEps")),
        "trailing_eps": _num(info.get("trailingEps"), alpha.get("EPS")),
        "beta": _num(info.get("beta"), alpha.get("Beta"), 1.0),
        "roe": _num(info.get("returnOnEquity"), alpha.get("ReturnOnEquityTTM"), _metric_from_finnhub(fin, "roeTTM")),
        "roa": _num(info.get("returnOnAssets"), alpha.get("ReturnOnAssetsTTM"), _metric_from_finnhub(fin, "roaTTM")),
        "analyst_target_mean": _num(pt.get("targetMean"), info.get("targetMeanPrice"), alpha.get("AnalystTargetPrice")),
        "analyst_target_median": _num(pt.get("targetMedian")),
        "analyst_target_high": _num(pt.get("targetHigh"), info.get("targetHighPrice")),
        "analyst_target_low": _num(pt.get("targetLow"), info.get("targetLowPrice")),
        "analyst_count": _num(pt.get("numberAnalysts"), info.get("numberOfAnalystOpinions")),
    }
    return {"metrics": metrics, "alpha": alpha, "finnhub": fin, "recommendations": rec, "price_target": pt, "upgrades": upgrades, "earnings_calendar": cal, "earnings_history": hist}


def earnings_intelligence(bundle: dict[str, Any]) -> dict[str, Any]:
    hist = bundle.get("earnings_history") or []
    cal = bundle.get("earnings_calendar") or []
    surprises = []
    for x in hist[:8]:
        est = _num(x.get("estimatedEPS")); act = _num(x.get("reportedEPS")); pct = _num(x.get("surprisePercentage"))
        if pct is None and est not in (None, 0) and act is not None:
            pct = (act / est - 1) * 100
        if pct is not None:
            surprises.append(float(pct))
    # Finnhub can provide recent actual/estimate too.
    for x in cal:
        try:
            d = pd.to_datetime(x.get("date"), utc=True)
        except Exception:
            continue
        if d <= pd.Timestamp.now(tz="UTC"):
            act = _num(x.get("epsActual")); est = _num(x.get("epsEstimate"))
            if act is not None and est not in (None, 0):
                surprises.append((act / est - 1) * 100)
    avg_surprise = float(np.mean(surprises[:6])) if surprises else 0.0
    beat_rate = float(np.mean(np.array(surprises[:8]) > 0)) if surprises else 0.5
    score = float(np.clip(50 + np.clip(avg_surprise, -25, 25) * 0.65 + (beat_rate - 0.5) * 18, 20, 80))

    now = datetime.now(UTC).date()
    upcoming = []
    for x in cal:
        try:
            d = datetime.fromisoformat(str(x.get("date"))[:10]).date()
        except Exception:
            continue
        if d >= now:
            upcoming.append((d, x))
    upcoming.sort(key=lambda z: z[0])
    next_event = upcoming[0][1] if upcoming else None
    days_to = (upcoming[0][0] - now).days if upcoming else None
    event_risk = 0.0
    if days_to is not None:
        event_risk = 10.0 if days_to <= 2 else (7.0 if days_to <= 5 else (4.0 if days_to <= 10 else 1.0))
    return {"score": score, "avg_eps_surprise_pct": avg_surprise, "beat_rate": beat_rate, "next": next_event, "days_to_next": days_to, "event_risk": event_risk, "surprises": surprises[:8]}


def analyst_intelligence(bundle: dict[str, Any], spot: float | None) -> dict[str, Any]:
    rec = bundle.get("recommendations") or []
    latest = rec[0] if rec else {}
    sb = _num(latest.get("strongBuy"), 0) or 0
    b = _num(latest.get("buy"), 0) or 0
    h = _num(latest.get("hold"), 0) or 0
    s = _num(latest.get("sell"), 0) or 0
    ss = _num(latest.get("strongSell"), 0) or 0
    total = sb + b + h + s + ss
    rec_score = 50.0 if total <= 0 else 50 + 45 * ((sb + 0.6*b - 0.6*s - ss) / total)
    target = bundle.get("metrics", {}).get("analyst_target_mean")
    upside = None
    if spot and target and spot > 0:
        upside = target / spot - 1
        rec_score += float(np.clip(upside * 55, -12, 12))
    return {"score": float(np.clip(rec_score, 10, 90)), "latest": latest, "target_upside": upside}


def valuation_engine(spot: float, bundle: dict[str, Any], macro: dict[str, Any]) -> dict[str, Any]:
    m = bundle.get("metrics") or {}
    if not spot or not np.isfinite(spot):
        return {"score": 50.0, "confidence": 20.0, "methods": []}

    rg = _num(m.get("revenue_growth"), 0.05) or 0.05
    eg = _num(m.get("earnings_growth"), rg) or rg
    growth = float(np.clip(np.nanmedian([rg, eg]), -0.05, 0.28))
    beta = float(np.clip(_num(m.get("beta"), 1.0) or 1.0, 0.4, 2.5))
    rf = 0.042
    for row in macro.get("rows", []) if isinstance(macro, dict) else []:
        if row.get("Asset") == "^TNX" and _num(row.get("Last")) is not None:
            rf = float(np.clip(float(row["Last"]) / 100, 0.0, 0.10))
            break
    wacc = float(np.clip(rf + beta * 0.05, 0.07, 0.15))

    methods = []
    fwd_eps = _num(m.get("forward_eps"))
    roe = _num(m.get("roe"), 0.12) or 0.12
    opm = _num(m.get("operating_margin"), 0.12) or 0.12
    quality_adj = np.clip((roe - 0.12) * 22 + (opm - 0.12) * 18, -5, 7)
    fair_pe = float(np.clip(17 + growth * 62 + quality_adj, 10, 46))
    if fwd_eps is not None and fwd_eps > 0:
        methods.append({"name": "Growth-adjusted Forward P/E", "fair": fwd_eps * fair_pe, "weight": 1.0, "detail": f"fair P/E≈{fair_pe:.1f}x"})

    fcf = _num(m.get("free_cash_flow")); shares = _num(m.get("shares")); cash = _num(m.get("total_cash"), 0) or 0; debt = _num(m.get("total_debt"), 0) or 0
    if fcf is not None and fcf > 0 and shares is not None and shares > 0:
        fair_yield = float(np.clip(0.055 - growth * 0.07 + (beta - 1) * 0.008, 0.025, 0.09))
        methods.append({"name": "FCF Yield", "fair": (fcf / shares) / fair_yield, "weight": 0.85, "detail": f"fair FCF yield≈{fair_yield*100:.1f}%"})

        def dcf_value(g0: float, w: float, tg: float) -> float | None:
            if w <= tg + 0.01:
                return None
            pv = 0.0; cur = fcf
            for yr in range(1, 6):
                # Fade growth toward terminal growth.
                gy = g0 + (tg - g0) * (yr / 6)
                cur *= (1 + gy)
                pv += cur / ((1 + w) ** yr)
            terminal = cur * (1 + tg) / (w - tg)
            ev = pv + terminal / ((1 + w) ** 5)
            return (ev + cash - debt) / shares
        bear = dcf_value(float(np.clip(growth * 0.55, -0.03, 0.12)), min(0.17, wacc + 0.018), 0.020)
        base = dcf_value(float(np.clip(growth * 0.85, 0.00, 0.18)), wacc, 0.025)
        bull = dcf_value(float(np.clip(growth * 1.15, 0.01, 0.24)), max(0.065, wacc - 0.012), 0.030)
        if base and base > 0:
            methods.append({"name": "5Y FCF DCF", "fair": base, "bear": bear, "bull": bull, "weight": 1.15, "detail": f"WACC≈{wacc*100:.1f}%, g≈{growth*100:.1f}%"})

    target = _num(m.get("analyst_target_mean"))
    if target is not None and target > 0:
        methods.append({"name": "Analyst target consensus", "fair": target, "weight": 0.75, "detail": "external analyst consensus"})

    vals = [(x["fair"], x["weight"]) for x in methods if _num(x.get("fair")) and x["fair"] > 0]
    if not vals:
        return {"score": 50.0, "confidence": 25.0, "methods": [], "growth_assumption": growth, "wacc": wacc}
    fair = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    arr = np.array([v for v, _ in vals], float)
    low = float(np.quantile(arr, 0.20)) if len(arr) > 1 else fair * 0.86
    high = float(np.quantile(arr, 0.80)) if len(arr) > 1 else fair * 1.14
    upside = fair / spot - 1
    score = float(np.clip(50 + upside * 85, 5, 95))
    dispersion = float(np.std(arr / fair)) if len(arr) > 1 else 0.18
    conf = float(np.clip(42 + len(vals) * 12 - dispersion * 120, 25, 90))
    return {"score": score, "confidence": conf, "fair_value": fair, "fair_low": low, "fair_high": high, "upside": upside, "methods": methods, "growth_assumption": growth, "wacc": wacc}


@st.cache_data(ttl=300, max_entries=32, show_spinner=False)
def openai_web_brief(ticker: str, api_key: str, model: str) -> dict[str, Any]:
    """Optional fresh web intelligence. It is narrative-only and never controls the quantitative probability."""
    if not api_key:
        return {}
    prompt = f"""Search the web for the most recent market-moving information about {ticker}.
Prioritize the last 24 hours, then the last 7 days. Verify: company news, earnings/guidance,
SEC filings, analyst upgrades/downgrades/price-target changes, material legal/regulatory items,
product/customer/AI demand developments, and macro/sector events directly relevant to the stock.
Distinguish confirmed facts from rumor. Return a concise Chinese brief with: Latest confirmed events,
Bullish catalysts, Bearish risks, Earnings/valuation implications, and What could invalidate the view.
Use source links/citations when available. Do not make a guaranteed price prediction."""
    payload = {
        "model": model or "gpt-5",
        "tools": [{"type": "web_search"}],
        "input": prompt,
        "max_output_tokens": 1400,
    }
    try:
        r = http_session().post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload), timeout=(4, 24),
        )
        if r.status_code >= 400:
            return {"error": f"OpenAI HTTP {r.status_code}: {r.text[:350]}"}
        data = r.json()
        texts = []
        if data.get("output_text"):
            texts.append(str(data["output_text"]))
        for item in data.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and c.get("text"):
                    texts.append(str(c["text"]))
        text = "\n".join(texts).strip()
        return {"text": text, "model": model or "gpt-5", "response_id": data.get("id")} if text else {"error": "OpenAI returned no text."}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# -----------------------------------------------------------------------------
# Macro context: market proxies + optional official FRED
# -----------------------------------------------------------------------------

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "PPI": "PPIACO",
    "Nonfarm Payrolls": "PAYEMS",
    "Unemployment": "UNRATE",
    "Fed Funds": "FEDFUNDS",
    "10Y Treasury": "DGS10",
    "2Y Treasury": "DGS2",
}


@st.cache_data(ttl=3600, max_entries=24, show_spinner=False)
def fred_latest(api_key: str) -> list[dict[str, Any]]:
    if not api_key: return []
    out = []
    for label, sid in FRED_SERIES.items():
        data, err = safe_json(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": sid, "api_key": api_key, "file_type": "json", "sort_order": "desc", "limit": 4},
            timeout=(3, 6),
        )
        if err or not isinstance(data, dict): continue
        vals = []
        for o in data.get("observations", []) or []:
            try:
                if o.get("value") != ".": vals.append((o.get("date"), float(o.get("value"))))
            except Exception: pass
        if vals:
            latest = vals[0]
            prev = vals[1] if len(vals) > 1 else (None, np.nan)
            out.append({"Series": label, "Date": latest[0], "Value": latest[1], "Previous": prev[1], "Change": latest[1] - prev[1] if pd.notna(prev[1]) else np.nan})
    return out


@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def macro_snapshot() -> dict[str, Any]:
    tickers = ("SPY", "QQQ", "IWM", "SMH", "^VIX", "^TNX", "DX-Y.NYB", "CL=F")
    d = yf_multi_daily(tickers, "6mo")
    rows = []
    score = 50.0
    for t in tickers:
        x = d.get(t, pd.DataFrame())
        if len(x) >= 21:
            last = float(x.Close.iloc[-1]); r1 = float(x.Close.pct_change().iloc[-1]); r5 = float(x.Close.pct_change(5).iloc[-1]); r20 = float(x.Close.pct_change(20).iloc[-1])
            rows.append({"Asset": t, "Last": last, "1D": r1, "5D": r5, "20D": r20})
    mp = {r["Asset"]: r for r in rows}
    score += float(np.clip(mp.get("SPY", {}).get("20D", 0) * 100, -7, 7))
    score += float(np.clip(mp.get("QQQ", {}).get("20D", 0) * 110, -8, 8))
    score += float(np.clip(mp.get("SMH", {}).get("20D", 0) * 80, -6, 6))
    vix = mp.get("^VIX", {}).get("Last")
    if vix is not None:
        score += 7 if vix < 17 else (-9 if vix > 30 else (-4 if vix > 24 else 0))
    tnx5 = mp.get("^TNX", {}).get("5D", 0)
    score += float(np.clip(-tnx5 * 80, -5, 5))
    dxy20 = mp.get("DX-Y.NYB", {}).get("20D", 0)
    score += float(np.clip(-dxy20 * 50, -3, 3))
    fred = fred_latest(api_keys()["fred"])
    regime = "Risk-on" if score >= 58 else ("Risk-off" if score <= 42 else "Mixed")
    return {"score": float(np.clip(score, 0, 100)), "rows": rows, "fred": fred, "regime": regime}


# -----------------------------------------------------------------------------
# ML ensemble with chronological holdout + probability calibration
# -----------------------------------------------------------------------------

@dataclass
class HorizonResult:
    horizon: int
    raw_probability: float
    model_probs: dict[str, float]
    model_metrics: dict[str, dict[str, float]]
    ensemble_metrics: dict[str, float]
    expected_hist_return: float
    dispersion: float


def model_bank() -> dict[str, Any]:
    return {
        "Logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=450, C=0.7, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=140, max_depth=7, min_samples_leaf=6, max_features="sqrt",
            random_state=42, n_jobs=1, class_weight="balanced_subsample",
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=160, max_depth=8, min_samples_leaf=5, max_features="sqrt",
            random_state=42, n_jobs=1, class_weight="balanced",
        ),
        "Hist Gradient": HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.055,
            l2_regularization=0.35, random_state=42,
        ),
    }


def make_dataset(ind: pd.DataFrame, horizon: int) -> pd.DataFrame:
    d = ind.copy()
    d["future_return"] = d["Close"].shift(-horizon) / d["Close"] - 1
    d["target"] = (d["future_return"] > 0).astype(int)
    present = [f for f in FEATURES if f in d.columns]
    if len(present) < 25:
        return pd.DataFrame()
    return d.dropna(subset=present + ["future_return"]).copy()


def _calibrate_prob(val_p: np.ndarray, val_y: np.ndarray, p: np.ndarray | float) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(p, dtype=float))
    try:
        if len(val_p) >= 60 and len(np.unique(val_y)) == 2 and len(np.unique(np.round(val_p, 4))) >= 12:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.03, y_max=0.97)
            iso.fit(val_p, val_y)
            return np.asarray(iso.predict(arr), dtype=float)
    except Exception:
        pass
    # Fail-safe: shrink raw probability toward 0.5 instead of overclaiming.
    return 0.5 + (arr - 0.5) * 0.82


@st.cache_data(ttl=7200, max_entries=64, show_spinner=False)
def train_horizons(ticker: str) -> dict[int, HorizonResult | None]:
    ind, _ = build_daily_feature_frame(ticker, "3y")
    out: dict[int, HorizonResult | None] = {}
    if ind.empty:
        return {1: None, 3: None, 5: None}
    features = [f for f in FEATURES if f in ind.columns]
    latest = ind[features].dropna().tail(1)
    if latest.empty:
        return {1: None, 3: None, 5: None}

    for horizon in (1, 3, 5):
        d = make_dataset(ind, horizon)
        if len(d) < 360:
            out[horizon] = None
            continue
        n = len(d)
        i1, i2 = int(n * 0.65), int(n * 0.80)
        train, val, test = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
        Xtr, ytr = train[features].fillna(0), train["target"].astype(int)
        Xv, yv = val[features].fillna(0), val["target"].astype(int)
        Xte, yte = test[features].fillna(0), test["target"].astype(int)
        latest_x = latest[features].fillna(0)

        probs: dict[str, float] = {}
        metrics: dict[str, dict[str, float]] = {}
        test_prob_list = []
        test_weight_list = []
        latest_prob_list = []
        latest_weight_list = []

        for name, base in model_bank().items():
            try:
                m = clone(base)
                m.fit(Xtr, ytr)
                pv = m.predict_proba(Xv)[:, 1]
                pt_raw = m.predict_proba(Xte)[:, 1]
                pl_raw = float(m.predict_proba(latest_x)[:, 1][0])
                pt = _calibrate_prob(pv, yv.values, pt_raw)
                pl = float(_calibrate_prob(pv, yv.values, pl_raw)[0])
                pred = (pt >= 0.5).astype(int)
                acc = float(accuracy_score(yte, pred))
                try: auc = float(roc_auc_score(yte, pt))
                except Exception: auc = 0.5
                brier = float(brier_score_loss(yte, np.clip(pt, 0.001, 0.999)))
                precision = float(precision_score(yte, pred, zero_division=0))
                recall = float(recall_score(yte, pred, zero_division=0))
                # Reward discrimination + calibration, but keep a minimum model voice.
                weight = max(0.06, (auc - 0.47) * 1.6 + (0.27 - brier) * 1.3 + (acc - 0.48) * 0.4)
                probs[name] = pl
                metrics[name] = {"accuracy": acc, "auc": auc, "brier": brier, "precision": precision, "recall": recall, "weight": weight}
                test_prob_list.append(pt); test_weight_list.append(weight)
                latest_prob_list.append(pl); latest_weight_list.append(weight)
            except Exception:
                continue

        if not latest_prob_list:
            out[horizon] = None
            continue
        latest_ens = float(np.average(latest_prob_list, weights=latest_weight_list))
        test_ens = np.average(np.vstack(test_prob_list), axis=0, weights=np.asarray(test_weight_list))
        pred = (test_ens >= 0.5).astype(int)
        try: ens_auc = float(roc_auc_score(yte, test_ens))
        except Exception: ens_auc = 0.5
        ens = {
            "accuracy": float(accuracy_score(yte, pred)),
            "auc": ens_auc,
            "brier": float(brier_score_loss(yte, np.clip(test_ens, 0.001, 0.999))),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "test_n": int(len(test)),
        }
        out[horizon] = HorizonResult(
            horizon=horizon,
            raw_probability=latest_ens,
            model_probs=probs,
            model_metrics=metrics,
            ensemble_metrics=ens,
            expected_hist_return=float(d["future_return"].tail(400).mean()),
            dispersion=float(np.std(latest_prob_list)),
        )
    return out


# -----------------------------------------------------------------------------
# Monte Carlo / empirical bootstrap
# -----------------------------------------------------------------------------

def bootstrap_paths(close: pd.Series, horizon: int, n_paths: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray] | None:
    rets = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(756).values
    if len(rets) < 100: return None
    # Recency-weighted empirical bootstrap preserves fat tails better than a Gaussian assumption.
    n = len(rets)
    w = np.exp(np.linspace(-2.0, 0.0, n)); w /= w.sum()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(rets, size=(n_paths, horizon), replace=True, p=w)
    start = float(close.iloc[-1])
    paths = start * np.cumprod(1 + sampled, axis=1)
    return paths, paths[:, -1]


def range_stats(close: pd.Series, horizon: int) -> dict[str, float]:
    sim = bootstrap_paths(close, horizon)
    if sim is None: return {}
    _, term = sim
    q = np.quantile(term, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {"p05": q[0], "p10": q[1], "p25": q[2], "p50": q[3], "p75": q[4], "p90": q[5], "p95": q[6], "up_prob": float(np.mean(term > close.iloc[-1]))}


def touch_probability(close: pd.Series, target: float, horizon: int) -> float:
    sim = bootstrap_paths(close, horizon)
    if sim is None: return np.nan
    paths, _ = sim; start = float(close.iloc[-1])
    if target >= start: return float(np.mean(np.max(paths, axis=1) >= target))
    return float(np.mean(np.min(paths, axis=1) <= target))


# -----------------------------------------------------------------------------
# Optional options snapshot — only runs when user requests it
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900, max_entries=32, show_spinner=False)
def options_snapshot(ticker: str) -> dict[str, Any]:
    try:
        tk = yf.Ticker(ticker)
        exps = list(tk.options or [])
        if not exps: return {}
        now = datetime.now().date()
        parsed = [(e, datetime.fromisoformat(e).date()) for e in exps]
        future = [(e, d) for e, d in parsed if d >= now]
        if not future: return {}
        # Prefer roughly one month to expiry.
        exp, dte_date = min(future, key=lambda z: abs((z[1] - now).days - 30))
        ch = tk.option_chain(exp)
        calls, puts = ch.calls.copy(), ch.puts.copy()
        spot = current_snapshot(ticker).get("price")
        if not spot or not np.isfinite(spot):
            hist = yf_history(ticker, "5d", "1d", False); spot = float(hist.Close.iloc[-1]) if not hist.empty else np.nan
        call_vol, put_vol = calls.get("volume", pd.Series(dtype=float)).fillna(0).sum(), puts.get("volume", pd.Series(dtype=float)).fillna(0).sum()
        call_oi, put_oi = calls.get("openInterest", pd.Series(dtype=float)).fillna(0).sum(), puts.get("openInterest", pd.Series(dtype=float)).fillna(0).sum()
        atm_iv = np.nan
        if np.isfinite(spot):
            rows = []
            for df in (calls, puts):
                if not df.empty and "strike" in df and "impliedVolatility" in df:
                    t = df.iloc[(df["strike"] - spot).abs().argsort()[:2]]
                    rows.extend(pd.to_numeric(t["impliedVolatility"], errors="coerce").dropna().tolist())
            if rows: atm_iv = float(np.median(rows))
        dte = max(1, (dte_date - now).days)
        implied_move = float(spot * atm_iv * math.sqrt(dte / 365)) if np.isfinite(spot) and np.isfinite(atm_iv) else np.nan
        return {
            "expiry": exp, "dte": dte, "put_call_volume": float(put_vol / call_vol) if call_vol else np.nan,
            "put_call_oi": float(put_oi / call_oi) if call_oi else np.nan, "atm_iv": atm_iv,
            "implied_move": implied_move, "spot": float(spot) if np.isfinite(spot) else np.nan,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# -----------------------------------------------------------------------------
# Main fusion
# -----------------------------------------------------------------------------

def _fuse_probability(
    h: int, ml: float, tech: float, intra: float, fund: float, news: float, macro: float,
    earnings: float = 0.5, analyst: float = 0.5, valuation: float = 0.5,
) -> float:
    # Horizon-aware ensemble. Long horizon increases fundamentals/valuation; 1D prioritizes live tape/news.
    if h == 1:
        w = {"ml": .46, "tech": .13, "intra": .16, "fund": .035, "news": .09, "macro": .06, "earn": .035, "analyst": .025, "value": .01}
    elif h == 3:
        w = {"ml": .52, "tech": .13, "intra": .09, "fund": .055, "news": .065, "macro": .055, "earn": .045, "analyst": .025, "value": .015}
    else:
        w = {"ml": .55, "tech": .11, "intra": .045, "fund": .075, "news": .05, "macro": .045, "earn": .045, "analyst": .035, "value": .045}
    vals = {"ml": ml, "tech": tech, "intra": intra, "fund": fund, "news": news, "macro": macro, "earn": earnings, "analyst": analyst, "value": valuation}
    raw = sum(w[k] * float(np.clip(vals[k], 0, 1)) for k in w)
    return float(np.clip(raw, 0.05, 0.95))


def _quality_score(raw_rows: int, news_count: int, provider_count: int, has_info: bool, intra: dict[str, Any], hres: dict[int, HorizonResult | None]) -> float:
    q = 46.0
    q += min(18, raw_rows / 50)
    q += min(12, news_count * 0.6)
    q += min(10, provider_count * 2.5)
    q += 6 if has_info else 0
    q += 5 if sum(1 for s in intra.get("signals", {}).values() if s.get("rows", 0) >= 25) >= 2 else 0
    q += 8 if all(hres.get(h) is not None for h in (1, 3, 5)) else 0
    return float(np.clip(q, 20, 100))


@st.cache_data(ttl=600, max_entries=48, show_spinner=False)
def analyze_max(ticker: str, include_options: bool = False, include_ai_web: bool = False) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", ticker):
        raise ValueError("股票代码格式不正确。")

    ind, feature_meta = build_daily_feature_frame(ticker, "3y")
    if ind.empty or len(ind) < 280:
        raise ValueError("无法取得足够历史数据，请检查代码或稍后重试。")

    # Independent blocks: one failure never blanks the page. Sequential execution is intentional
    # because Streamlit's docs do not officially support arbitrary app threads.
    tasks = [
        ("snapshot", lambda: current_snapshot(ticker)),
        ("intraday", lambda: intraday_bundle(ticker)),
        ("news", lambda: news_radar(ticker)),
        ("macro", lambda: macro_snapshot()),
        ("info", lambda: company_info(ticker)),
        ("models", lambda: train_horizons(ticker)),
    ]
    results: dict[str, Any] = {}
    health = []
    for name, fn in tasks:
        name, out, err, elapsed = safe_call(name, fn)
        results[name] = out
        health.append({"Module": name, "Seconds": elapsed, "Status": "OK" if not err else err})

    snap = results.get("snapshot") or {}
    intra = results.get("intraday") or {"score": 50, "signals": {}}
    news = results.get("news") or {"score": 0, "items": [], "catalysts": [], "risks": [], "provider_count": 0, "article_count": 0, "provider_status": []}
    macro = results.get("macro") or {"score": 50, "rows": [], "fred": [], "regime": "Unknown"}
    info = results.get("info") or {}
    hres = results.get("models") or {1: None, 3: None, 5: None}

    tech, tech_notes = technical_score(ind)
    fund, fund_notes, fund_fields = fundamental_score(info)
    fund_bundle = build_fundamental_bundle(ticker, info)
    earn_intel = earnings_intelligence(fund_bundle)
    spot_for_value = _num(snap.get("price"), float(ind["Close"].iloc[-1])) or float(ind["Close"].iloc[-1])
    analyst_intel = analyst_intelligence(fund_bundle, spot_for_value)
    valuation = valuation_engine(spot_for_value, fund_bundle, macro)
    news01 = float(np.clip(50 + news.get("score", 0) * 0.38, 0, 100))
    macro01 = float(macro.get("score", 50))
    intra01 = float(intra.get("score", 50))

    fused: dict[int, float] = {}
    for h in (1, 3, 5):
        ml = hres[h].raw_probability if hres.get(h) else 0.5
        p = _fuse_probability(
            h, ml, tech / 100, intra01 / 100, fund / 100, news01 / 100, macro01 / 100,
            earn_intel.get("score", 50) / 100, analyst_intel.get("score", 50) / 100,
            valuation.get("score", 50) / 100,
        )
        fused[h] = p

    dispersion = float(np.mean([hres[h].dispersion for h in (1, 3, 5) if hres.get(h)])) if any(hres.get(h) for h in (1, 3, 5)) else 0.18
    quality = _quality_score(len(ind), news.get("article_count", 0), news.get("provider_count", 0), bool(info), intra, hres)
    if fund_bundle.get("alpha"): quality += 3
    if fund_bundle.get("finnhub"): quality += 3
    quality = float(np.clip(quality, 20, 100))
    model_quality = float(np.mean([hres[h].ensemble_metrics.get("auc", 0.5) for h in (1, 3, 5) if hres.get(h)])) if any(hres.get(h) for h in (1, 3, 5)) else 0.5
    confidence = 50 + (quality - 60) * 0.35 + (model_quality - 0.5) * 70 - dispersion * 95
    confidence = float(np.clip(confidence, 25, 92))

    # Reliability shrinkage: lower confidence means the displayed probability moves toward 50%.
    shrink = 0.55 + 0.45 * confidence / 100
    for h in fused:
        fused[h] = float(np.clip(0.5 + (fused[h] - 0.5) * shrink, 0.12, 0.88))

    vol = float(ind["volatility20"].iloc[-1]) if pd.notna(ind["volatility20"].iloc[-1]) else 0.35
    vix = np.nan
    for row in macro.get("rows", []):
        if row.get("Asset") == "^VIX": vix = row.get("Last", np.nan)
    risk = 2.7 + vol * 6.5 + (100 - confidence) / 24 + (1.0 if pd.notna(vix) and vix > 25 else 0) + earn_intel.get("event_risk", 0) * 0.12
    risk = float(np.clip(risk, 1, 10))

    close = ind["Close"].dropna()
    ranges = {h: range_stats(close, h) for h in (1, 3, 5)}
    opt = options_snapshot(ticker) if include_options else {}
    ai_web = {}
    if include_ai_web and api_keys()["openai"]:
        _, ai_web, ai_err, ai_elapsed = safe_call("openai_web", lambda: openai_web_brief(ticker, api_keys()["openai"], api_keys()["openai_model"]))
        health.append({"Module": "openai_web", "Seconds": ai_elapsed, "Status": "OK" if not ai_err and not (isinstance(ai_web, dict) and ai_web.get("error")) else (ai_err or ai_web.get("error"))})
    return {
        "ticker": ticker, "version": APP_VERSION, "generated_at": datetime.now(UTC).isoformat(),
        "snapshot": snap, "ind": ind, "feature_meta": feature_meta, "intraday": intra, "news": news,
        "macro": macro, "info": info, "fund_fields": fund_fields, "fund_bundle": fund_bundle,
        "earnings_intel": earn_intel, "analyst_intel": analyst_intel, "valuation": valuation,
        "hres": hres, "fused": fused,
        "tech": tech, "tech_notes": tech_notes, "fund": fund, "fund_notes": fund_notes,
        "news01": news01, "macro01": macro01, "confidence": confidence, "data_quality": quality,
        "risk": risk, "ranges": ranges, "options": opt, "ai_web": ai_web, "health": health,
    }


# -----------------------------------------------------------------------------
# Fast scanner (does NOT train 12 models per ticker)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900, max_entries=128, show_spinner=False)
def fast_scan_one(ticker: str) -> dict[str, Any] | None:
    d = yf_history(ticker, "1y", "1d", False)
    if len(d) < 120: return None
    ind = add_indicators(d)
    tech, _ = technical_score(ind)
    last = ind.iloc[-1]
    r20 = float(last.get("ret20", 0)); r5 = float(last.get("ret5", 0)); vol = float(last.get("volatility20", 0.4))
    q = current_snapshot(ticker)
    score = float(np.clip(50 + (tech - 50) * 0.55 + np.clip(r20 * 130, -12, 12) + np.clip(r5 * 80, -7, 7), 5, 95))
    # Convert ranking score to a conservative directional probability, not a full MAX probability.
    p = float(np.clip(0.5 + (score - 50) / 180, 0.28, 0.72))
    return {"Ticker": ticker, "Fast Up Score": p, "Technical": tech, "20D Return": r20, "5D Return": r5, "Volatility": vol, "Price": q.get("price", float(d.Close.iloc[-1])), "Source": q.get("source", "Yahoo")}


# -----------------------------------------------------------------------------
# Backtest — bounded walk-forward
# -----------------------------------------------------------------------------

def walk_forward_backtest(ticker: str, horizon: int) -> pd.DataFrame:
    ind, _ = build_daily_feature_frame(ticker, "3y")
    d = make_dataset(ind, horizon)
    if len(d) < 420: return pd.DataFrame()
    features = [f for f in FEATURES if f in d.columns]
    start = int(len(d) * 0.56)
    remaining = len(d) - start
    step = max(40, remaining // 5)
    rows = []
    for end in range(start, len(d) - step + 1, step):
        tr, te = d.iloc[:end], d.iloc[end:min(end + step, len(d))]
        if len(te) < 20: continue
        Xtr, ytr = tr[features].fillna(0), tr["target"].astype(int)
        Xte, yte = te[features].fillna(0), te["target"].astype(int)
        ps = []
        for base in model_bank().values():
            try:
                m = clone(base); m.fit(Xtr, ytr); ps.append(m.predict_proba(Xte)[:, 1])
            except Exception: pass
        if not ps: continue
        p = np.mean(np.vstack(ps), axis=0); pred = (p >= 0.5).astype(int)
        try: auc = roc_auc_score(yte, p)
        except Exception: auc = np.nan
        strat = np.where(pred == 1, te["future_return"].values, -te["future_return"].values)
        equity = np.cumprod(1 + np.nan_to_num(strat))
        peak = np.maximum.accumulate(equity); dd = equity / peak - 1
        rows.append({
            "Start": str(te.index[0].date()) if hasattr(te.index[0], "date") else str(te.index[0]),
            "N": len(te), "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0), "ROC AUC": auc,
            "Avg Strategy Return": float(np.mean(strat)), "Max Drawdown": float(np.min(dd)),
        })
        if len(rows) >= 5: break
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def fmt_money(x: Any) -> str:
    try:
        x = float(x)
        if not np.isfinite(x): return "—"
        if abs(x) >= 1e12: return f"${x/1e12:.2f}T"
        if abs(x) >= 1e9: return f"${x/1e9:.2f}B"
        if abs(x) >= 1e6: return f"${x/1e6:.2f}M"
        return f"${x:,.2f}"
    except Exception: return "—"


def fmt_pct(x: Any, already_percent: bool = False) -> str:
    try:
        x = float(x)
        if not np.isfinite(x): return "—"
        return f"{x:.2f}%" if already_percent else f"{x*100:.2f}%"
    except Exception: return "—"


def render_source_badges() -> None:
    k = api_keys()
    active = ["Yahoo", "SEC"]
    if k["massive"]: active.append("Massive/Polygon")
    if k["alpha"]: active.append("Alpha Vantage")
    if k["finnhub"]: active.append("Finnhub")
    if k["marketaux"]: active.append("Marketaux")
    if k["fred"]: active.append("FRED")
    if k["openai"]: active.append("OpenAI Web")
    st.markdown(" ".join([f'<span class="pill">{x}</span>' for x in active]), unsafe_allow_html=True)


def render_live_quote_fragment(ticker: str) -> None:
    """Lightweight auto-refresh. It never retrains models or reruns the full analysis."""
    @st.fragment(run_every="30s")
    def _live() -> None:
        q = current_snapshot(ticker)
        if not q:
            st.caption("Live quote unavailable; full analysis can still use historical fallback.")
            return
        age = timestamp_age_seconds(q.get("updated"))
        age_txt = f"{age:.0f}s old" if age is not None else "provider timestamp unavailable"
        cols = st.columns(5)
        cols[0].metric("LIVE", fmt_money(q.get("price")), f"{q.get('change_pct', np.nan):+.2f}%" if _num(q.get('change_pct')) is not None else None)
        cols[1].metric("Bid", fmt_money(q.get("bid")))
        cols[2].metric("Ask", fmt_money(q.get("ask")))
        cols[3].metric("Volume", f"{int(q.get('volume')):,}" if _num(q.get('volume')) is not None else "—")
        cols[4].metric("Feed", q.get("source", "—"))
        st.caption(f"Auto refresh 30s · {age_txt} · exchange entitlement/delay depends on provider plan")
    _live()


def render_result(r: dict[str, Any]) -> None:
    t = r["ticker"]
    snap = r.get("snapshot") or {}
    price = snap.get("price")
    change = snap.get("change_pct")
    st.subheader(f"{t} · MAX ULTRA 分析")
    top = st.columns(8)
    top[0].metric("最新价格", fmt_money(price), f"{change:+.2f}%" if isinstance(change, (int, float)) and np.isfinite(change) else None)
    top[1].metric("1D 上涨概率", f"{r['fused'][1]*100:.1f}%")
    top[2].metric("3D 上涨概率", f"{r['fused'][3]*100:.1f}%")
    top[3].metric("5D 上涨概率", f"{r['fused'][5]*100:.1f}%")
    top[4].metric("Confidence", f"{r['confidence']:.0f}/100")
    top[5].metric("Risk", f"{r['risk']:.1f}/10")
    top[6].metric("Data Quality", f"{r['data_quality']:.0f}/100")
    top[7].metric("Session", market_session_status())
    st.caption(f"Quote source: {snap.get('source','Unavailable')} · Generated: {r['generated_at'][:19]} UTC · Probability is calibrated/shrunk when confidence is low.")

    c1, c2, c3 = st.columns([2.0, 1.0, 1.0])
    with c1:
        ind = r["ind"].tail(220)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ind.index, open=ind.Open, high=ind.High, low=ind.Low, close=ind.Close, name=t))
        fig.add_trace(go.Scatter(x=ind.index, y=ind.ema20, name="EMA20", line=dict(width=1.3)))
        fig.add_trace(go.Scatter(x=ind.index, y=ind.ema50, name="EMA50", line=dict(width=1.2)))
        fig.add_trace(go.Scatter(x=ind.index, y=ind.ema200, name="EMA200", line=dict(width=1.0)))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=25, b=8), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**多因子评分**")
        st.metric("Technical", f"{r['tech']:.0f}/100")
        st.metric("Intraday", f"{r['intraday'].get('score',50):.0f}/100")
        st.metric("News", f"{r['news01']:.0f}/100")
        st.metric("Macro", f"{r['macro01']:.0f}/100")
        st.metric("Fundamental", f"{r['fund']:.0f}/100")
    with c3:
        st.markdown("**价格情景 (Monte Carlo)**")
        for h in (1, 3, 5):
            s = r["ranges"].get(h, {})
            if s:
                st.write(f"**{h}D** P10 {fmt_money(s['p10'])} · P50 {fmt_money(s['p50'])} · P90 {fmt_money(s['p90'])}")
        st.markdown("**主要催化剂**")
        st.write(", ".join(r["news"].get("catalysts", [])) or "—")
        st.markdown("**主要风险**")
        st.write(", ".join(r["news"].get("risks", [])) or "—")

    tabs = st.tabs(["⏱ 多周期", "🤖 ML 模型", "📰 新闻/SEC", "🌍 宏观", "🏢 基本面", "💰 估值", "📅 财报/机构", "🧠 AI Web", "🧯 稳定性"])
    with tabs[0]:
        rows = []
        for iv, s in r["intraday"].get("signals", {}).items():
            rows.append({"Interval": iv, "Trend": s.get("trend"), "Score": s.get("score"), "RSI": s.get("rsi"), "Last bar return": s.get("ret"), "Volume ratio": s.get("volume_ratio"), "Rows": s.get("rows"), "Source": s.get("source")})
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"Score": "{:.1f}", "RSI": "{:.1f}", "Last bar return": "{:.2%}", "Volume ratio": "{:.2f}"}, na_rep="—"), use_container_width=True, hide_index=True)
        z = r["ind"].iloc[-1]
        met = st.columns(6)
        met[0].metric("RSI14", f"{z.get('rsi14',np.nan):.1f}")
        met[1].metric("ADX14", f"{z.get('adx14',np.nan):.1f}")
        met[2].metric("MACD Hist", f"{z.get('macd_hist',np.nan):.3f}")
        met[3].metric("ATR %", fmt_pct(z.get("atr_pct")))
        met[4].metric("Volume Ratio", f"{z.get('volume_ratio',np.nan):.2f}x")
        met[5].metric("20D Vol", fmt_pct(z.get("volatility20")))
    with tabs[1]:
        rows = []
        for h in (1, 3, 5):
            hr = r["hres"].get(h)
            if not hr: continue
            for name, p in hr.model_probs.items():
                m = hr.model_metrics.get(name, {})
                rows.append({"Horizon": f"{h}D", "Model": name, "Latest Up": p, "Accuracy": m.get("accuracy"), "ROC AUC": m.get("auc"), "Brier": m.get("brier"), "Weight": m.get("weight")})
            e = hr.ensemble_metrics
            rows.append({"Horizon": f"{h}D", "Model": "ENSEMBLE", "Latest Up": hr.raw_probability, "Accuracy": e.get("accuracy"), "ROC AUC": e.get("auc"), "Brier": e.get("brier"), "Weight": np.nan})
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"Latest Up": "{:.1%}", "Accuracy": "{:.1%}", "ROC AUC": "{:.3f}", "Brier": "{:.3f}", "Weight": "{:.2f}"}, na_rep="—"), use_container_width=True, hide_index=True)
        st.caption("Metrics use a chronological holdout. Latest probabilities are calibrated when validation data allow it, then the final display is shrunk toward 50% when confidence/data quality are weak.")
    with tabs[2]:
        st.write(f"News score: **{r['news'].get('score',0):+.1f}/100** · Unique items: **{r['news'].get('article_count',0)}** · Active news/filing providers: **{r['news'].get('provider_count',0)}**")
        if r["news"].get("provider_status"):
            st.dataframe(pd.DataFrame(r["news"]["provider_status"]), use_container_width=True, hide_index=True)
        for n in r["news"].get("items", [])[:35]:
            label = "Bullish" if n.get("score", 0) > 15 else ("Bearish" if n.get("score", 0) < -15 else "Neutral")
            title = n.get("title", "")
            url = n.get("url", "")
            head = f"[{title}]({url})" if url else title
            st.markdown(f"**{label} · {n.get('score',0):+.0f} · Imp {n.get('importance',0):.0f}/10** — {head}")
            st.caption(f"{n.get('provider','')} · age ~{n.get('age_hours',0):.1f}h")
    with tabs[3]:
        st.write(f"Regime: **{r['macro'].get('regime','Unknown')}** · Macro score **{r['macro01']:.0f}/100**")
        if r["macro"].get("rows"):
            st.dataframe(pd.DataFrame(r["macro"]["rows"]).style.format({"Last": "{:.2f}", "1D": "{:.2%}", "5D": "{:.2%}", "20D": "{:.2%}"}), use_container_width=True, hide_index=True)
        if r["macro"].get("fred"):
            st.markdown("**Official FRED series**")
            st.dataframe(pd.DataFrame(r["macro"]["fred"]), use_container_width=True, hide_index=True)
        else:
            st.caption("Add FRED_API_KEY in Streamlit Secrets to include official CPI/PPI/NFP/Unemployment/Fed Funds/2Y/10Y observations.")
    with tabs[4]:
        f = r["fund_fields"]
        data = []
        for k, v in f.items():
            if k in {"Revenue Growth", "Earnings Growth", "Gross Margin", "Operating Margin"}:
                show = fmt_pct(v)
            elif k in {"Market Cap", "Free Cash Flow"}:
                show = fmt_money(v)
            else:
                show = "—" if v is None else str(round(v, 3) if isinstance(v, float) else v)
            data.append({"Metric": k, "Value": show})
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        if api_keys()["finnhub"]:
            rec = finnhub_recommendations(t, api_keys()["finnhub"])
            ear = finnhub_earnings(t, api_keys()["finnhub"])
            if rec:
                st.markdown("**Analyst recommendation trends (Finnhub)**"); st.dataframe(pd.DataFrame(rec[:3]), use_container_width=True, hide_index=True)
            if ear:
                st.markdown("**Earnings calendar (Finnhub)**"); st.dataframe(pd.DataFrame(ear[:8]), use_container_width=True, hide_index=True)
    with tabs[5]:
        v = r.get("valuation") or {}
        c = st.columns(5)
        c[0].metric("Valuation Score", f"{v.get('score',50):.0f}/100")
        c[1].metric("Fair Value", fmt_money(v.get("fair_value")))
        c[2].metric("Fair Low", fmt_money(v.get("fair_low")))
        c[3].metric("Fair High", fmt_money(v.get("fair_high")))
        c[4].metric("Upside/Downside", fmt_pct(v.get("upside")))
        st.caption(f"Valuation confidence {v.get('confidence',0):.0f}/100 · WACC assumption {fmt_pct(v.get('wacc'))} · growth assumption {fmt_pct(v.get('growth_assumption'))}")
        if v.get("methods"):
            st.dataframe(pd.DataFrame(v["methods"]), use_container_width=True, hide_index=True)
        st.warning("估值是模型区间，不是目标价承诺。DCF 对增长/WACC 很敏感；亏损或负 FCF 公司会自动降低可用估值方法数量。")
    with tabs[6]:
        e = r.get("earnings_intel") or {}; a = r.get("analyst_intel") or {}; fb = r.get("fund_bundle") or {}
        c = st.columns(6)
        c[0].metric("Earnings Score", f"{e.get('score',50):.0f}/100")
        c[1].metric("Avg EPS Surprise", f"{e.get('avg_eps_surprise_pct',0):+.1f}%")
        c[2].metric("Beat Rate", f"{e.get('beat_rate',0.5):.0%}")
        c[3].metric("Days to Earnings", str(e.get("days_to_next")) if e.get("days_to_next") is not None else "—")
        c[4].metric("Analyst Score", f"{a.get('score',50):.0f}/100")
        c[5].metric("Consensus Upside", fmt_pct(a.get("target_upside")))
        if e.get("next"):
            st.markdown("**Next earnings event**"); st.json(e["next"])
        if fb.get("earnings_history"):
            st.markdown("**Recent EPS history (Alpha Vantage)**"); st.dataframe(pd.DataFrame(fb["earnings_history"][:8]), use_container_width=True, hide_index=True)
        if fb.get("recommendations"):
            st.markdown("**Analyst recommendation trends (Finnhub)**"); st.dataframe(pd.DataFrame(fb["recommendations"][:4]), use_container_width=True, hide_index=True)
        if fb.get("price_target"):
            st.markdown("**Analyst price targets (Finnhub)**"); st.json(fb["price_target"])
        if fb.get("upgrades"):
            st.markdown("**Recent upgrades/downgrades (Finnhub; plan-dependent)**"); st.dataframe(pd.DataFrame(fb["upgrades"][:10]), use_container_width=True, hide_index=True)
    with tabs[7]:
        ai = r.get("ai_web") or {}
        if ai.get("text"):
            st.markdown(ai["text"])
            st.caption(f"OpenAI web intelligence · model {ai.get('model','')} · narrative only; not directly used as a numeric trading probability.")
        elif ai.get("error"):
            st.warning(ai["error"])
        elif api_keys()["openai"]:
            st.info("本次分析未启用 AI Web。回到分析顶部打开『AI Web 最新情报』后重新运行。")
        else:
            st.info("在 Streamlit Secrets 添加 OPENAI_API_KEY 后，可按需启用最新网页情报搜索。")
    with tabs[8]:
        st.markdown("**Module health / latency**")
        st.dataframe(pd.DataFrame(r["health"]), use_container_width=True, hide_index=True)
        st.write("Protection enabled: per-provider timeouts · bounded retries · sequential heavy modules · ML n_jobs=1 · 30s quote fragment only · bounded news/MC/history sizes · cache TTL · no training on page load · failures isolated · top-level crash screen · streamlit_app.py safe wrapper.")
        opt = r.get("options") or {}
        if opt:
            st.markdown("**Options snapshot**")
            if opt.get("error"): st.warning(opt["error"])
            else:
                c = st.columns(5)
                c[0].metric("Expiry", str(opt.get("expiry", "—")))
                c[1].metric("P/C Volume", f"{opt.get('put_call_volume',np.nan):.2f}")
                c[2].metric("P/C OI", f"{opt.get('put_call_oi',np.nan):.2f}")
                c[3].metric("ATM IV", fmt_pct(opt.get("atm_iv")))
                c[4].metric("Implied Move", fmt_money(opt.get("implied_move")))


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------

def render_sidebar() -> str:
    with st.sidebar:
        st.title("📈 Stock AI MAX ULTRA X")
        st.caption(f"v{APP_VERSION} · Institutional Cloud-Safe")
        page = st.radio("功能", ["🚀 MAX 分析", "🛰️ 实时新闻雷达", "⏱ 盘中信号", "🎯 目标价概率", "🏆 Scanner", "🧪 回测", "🔌 数据源"], index=0)
        st.divider()
        st.write("**当前可用数据源**")
        render_source_badges()
        st.caption("不填任何 API Key 也可运行；填入更多 Keys 后自动扩大覆盖。")
        if st.button("🧹 清除缓存", use_container_width=True):
            st.cache_data.clear(); st.success("缓存已清除")
        st.caption("说明：实时程度取决于你的数据源套餐。概率是估计值，不保证收益。")
    return page


def page_analysis() -> None:
    a, b, c, d = st.columns([2.2, 0.9, 1.15, 1.15])
    ticker = a.text_input("股票代码", "NVDA", key="main_ticker").upper().strip()
    include_options = b.toggle("Options", value=False, help="较慢，默认关闭保护云端。")
    include_ai_web = c.toggle("AI Web 最新情报", value=False, help="需要 OPENAI_API_KEY；会额外搜索最新网页信息，较慢且有 API 成本。")
    run = d.button("🚀 开始 ULTRA X", type="primary", use_container_width=True)
    st.markdown("**实时价格脉冲（不重训模型）**")
    render_live_quote_fragment(ticker)
    if run:
        with st.status("MAX ULTRA 正在分析…", expanded=True) as status:
            try:
                st.write("① 构建 3 年日线 + SPY/QQQ/SMH/VIX/10Y 市场特征")
                r = analyze_max(ticker, include_options, include_ai_web)
                st.write("② 实时/盘中、新闻/SEC、宏观、基本面、ML 已完成")
                st.session_state["ultra_result"] = r
                st.write("③ 估值、财报/机构情报与事件风险已完成")
                status.update(label="分析完成", state="complete", expanded=False)
            except Exception as e:
                status.update(label="分析失败，但应用仍保持在线", state="error", expanded=True)
                st.error(f"{type(e).__name__}: {e}")
                with st.expander("错误详情"):
                    st.code(traceback.format_exc()[-5000:])
    r = st.session_state.get("ultra_result")
    if r and r.get("ticker") == ticker:
        render_result(r)


def page_news() -> None:
    t = st.text_input("股票代码", "NVDA", key="news_t").upper().strip()
    if st.button("刷新全部新闻源", type="primary"):
        yahoo_news.clear(); alpha_news.clear(); finnhub_news.clear(); marketaux_news.clear(); sec_filings.clear(); news_radar.clear()
    with st.spinner("汇总 Yahoo / SEC / 已配置的新闻 API…"):
        n = news_radar(t)
    c = st.columns(4)
    c[0].metric("综合新闻情绪", f"{n.get('score',0):+.1f}/100")
    c[1].metric("独立条目", n.get("article_count", 0))
    c[2].metric("有效来源", n.get("provider_count", 0))
    c[3].metric("窗口", "近 7 天")
    if n.get("provider_status"):
        st.dataframe(pd.DataFrame(n["provider_status"]), use_container_width=True, hide_index=True)
    for x in n.get("items", [])[:60]:
        label = "Bullish" if x.get("score", 0) > 15 else ("Bearish" if x.get("score", 0) < -15 else "Neutral")
        title = x.get("title", ""); url = x.get("url", "")
        st.markdown(f"**{label} {x.get('score',0):+.0f} · Imp {x.get('importance',0):.0f}/10** — " + (f"[{title}]({url})" if url else title))
        st.caption(f"{x.get('provider','')} · ~{x.get('age_hours',0):.1f}h")


def page_intraday() -> None:
    t = st.text_input("股票代码", "NVDA", key="intraday_t").upper().strip()
    if st.button("刷新盘中数据", type="primary"):
        intraday_bars.clear(); massive_bars.clear(); intraday_bundle.clear(); current_snapshot.clear()
    q = current_snapshot(t)
    b = intraday_bundle(t)
    c = st.columns(5)
    c[0].metric("Latest", fmt_money(q.get("price")), f"{q.get('change_pct',np.nan):+.2f}%" if isinstance(q.get("change_pct"), (int,float)) and np.isfinite(q.get("change_pct")) else None)
    c[1].metric("盘中综合", f"{b.get('score',50):.0f}/100")
    for col, iv in zip(c[2:], ("1h", "15m", "5m")):
        s = b.get("signals", {}).get(iv, {}); col.metric(iv, f"{s.get('score',50):.0f}/100", s.get("trend", ""))
    st.caption(f"Quote source: {q.get('source','—')} · Session: {market_session_status()}")
    rows = []
    for iv, s in b.get("signals", {}).items(): rows.append({"Interval": iv, **s})
    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_target() -> None:
    t = st.text_input("股票代码", "SNDK", key="target_t").upper().strip()
    d = yf_history(t, "3y", "1d", False)
    if d.empty:
        st.warning("无法取得行情。")
        return
    q = current_snapshot(t); last = float(q.get("price") or d.Close.iloc[-1])
    target = st.number_input("目标价格", min_value=0.01, value=float(round(last * 1.05, 2)), step=0.5)
    c = st.columns(3)
    for col, h in zip(c, (1, 3, 5)):
        p = touch_probability(d.Close, target, h)
        col.metric(f"{h}交易日内触及", f"{p*100:.1f}%" if pd.notna(p) else "—")
    st.caption("Empirical bootstrap based on recent historical return distribution; includes fat tails better than a simple normal model, but cannot predict unique future events.")


def page_scanner() -> None:
    default = "NVDA,AVGO,AMAT,AMZN,GOOGL,META,MSFT,SNDK,AMD,TSM"
    text = st.text_area("股票列表（Fast Scanner 不会对每只股票重复训练全部模型）", default, height=90)
    tickers = [x.strip().upper() for x in re.split(r"[,\s]+", text) if x.strip()][:20]
    if st.button("开始扫描", type="primary"):
        out = []; bar = st.progress(0)
        for i, t in enumerate(tickers):
            try:
                r = fast_scan_one(t)
                if r: out.append(r)
            except Exception:
                pass
            bar.progress((i + 1) / max(1, len(tickers)))
        if out:
            df = pd.DataFrame(out).sort_values(["Fast Up Score", "Technical"], ascending=False).reset_index(drop=True); df.index += 1
            st.dataframe(df.style.format({"Fast Up Score": "{:.1%}", "Technical": "{:.0f}", "20D Return": "{:.1%}", "5D Return": "{:.1%}", "Volatility": "{:.1%}", "Price": "${:.2f}"}), use_container_width=True)
            st.caption("Scanner score is a lightweight ranking signal, not the full MAX probability. Run full analysis on the top names before using the result.")


def page_backtest() -> None:
    c1, c2 = st.columns(2)
    t = c1.text_input("股票代码", "NVDA", key="bt_t").upper().strip()
    h = c2.selectbox("预测周期", [1, 3, 5], index=2)
    if st.button("运行 Walk-forward 回测", type="primary"):
        with st.spinner("最多 5 个时间顺序窗口，限制计算量保护云端实例…"):
            f = walk_forward_backtest(t, h)
        if f.empty:
            st.warning("历史数据不足或回测失败。")
        else:
            st.dataframe(f.style.format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "ROC AUC": "{:.3f}", "Avg Strategy Return": "{:.2%}", "Max Drawdown": "{:.1%}"}, na_rep="—"), use_container_width=True, hide_index=True)
            m = st.columns(6)
            m[0].metric("Accuracy", f"{f.Accuracy.mean():.1%}")
            m[1].metric("Precision", f"{f.Precision.mean():.1%}")
            m[2].metric("Recall", f"{f.Recall.mean():.1%}")
            m[3].metric("ROC AUC", f"{f['ROC AUC'].mean():.3f}")
            m[4].metric("Avg Return", f"{f['Avg Strategy Return'].mean():.2%}")
            m[5].metric("Worst DD", f"{f['Max Drawdown'].min():.1%}")


def page_sources() -> None:
    st.subheader("🔌 数据源 / Streamlit Secrets")
    st.write("默认无需 Key：Yahoo Finance + SEC EDGAR。为了提高实时性和新闻覆盖，可在 Streamlit → Manage app → Settings/Secrets 添加下面的 Keys。")
    k = api_keys()
    rows = [
        {"Source": "Yahoo Finance", "Configured": "Built-in", "Use": "Daily/intraday fallback, fundamentals"},
        {"Source": "SEC EDGAR", "Configured": "Built-in", "Use": "Official 8-K/10-Q/10-K/Form 4/etc."},
        {"Source": "Massive/Polygon", "Configured": "Yes" if k["massive"] else "No", "Use": "Current snapshot + intraday bars"},
        {"Source": "Alpha Vantage", "Configured": "Yes" if k["alpha"] else "No", "Use": "News + native ticker sentiment"},
        {"Source": "Finnhub", "Configured": "Yes" if k["finnhub"] else "No", "Use": "Quote, company news, analyst trends, earnings calendar"},
        {"Source": "Marketaux", "Configured": "Yes" if k["marketaux"] else "No", "Use": "Broad multi-source financial news + entity sentiment"},
        {"Source": "FRED", "Configured": "Yes" if k["fred"] else "No", "Use": "Official CPI/PPI/NFP/UNRATE/Fed Funds/2Y/10Y"},
        {"Source": "OpenAI Web", "Configured": "Yes" if k["openai"] else "No", "Use": "Optional fresh web intelligence / event verification"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.code(
        '''MASSIVE_API_KEY = ""
ALPHA_VANTAGE_API_KEY = ""
FINNHUB_API_KEY = ""
MARKETAUX_API_KEY = ""
FRED_API_KEY = ""
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5"
SEC_USER_AGENT = "Your Name your-email@example.com"''',
        language="toml",
    )
    st.warning("不要把真实 API Key 上传到 GitHub。只放在 Streamlit Secrets。")
    st.markdown("**稳定性保护**：所有外部 API 有 timeout/retry 上限；主分析/新闻/盘中采用 Streamlit-safe 顺序执行；ML 每个模型 n_jobs=1；Options 默认关闭；Scanner 使用轻量算法；任何模块失败都只降低 Data Quality，不会让整页黑屏。")


def main() -> None:
    st.markdown(
        '<div class="hero"><h2 style="margin:0">Stock AI MAX ULTRA X · 实时事件驱动 + 估值 + 概率校准</h2>'
        '<div class="small">实时/盘中 + 技术/ML + 多来源新闻/SEC + 财报/机构 + 宏观 + 多模型校准 + DCF/FCF/PE估值 + Monte Carlo + AI Web（可选）</div></div>',
        unsafe_allow_html=True,
    )
    page = render_sidebar()
    if page == "🚀 MAX 分析": page_analysis()
    elif page == "🛰️ 实时新闻雷达": page_news()
    elif page == "⏱ 盘中信号": page_intraday()
    elif page == "🎯 目标价概率": page_target()
    elif page == "🏆 Scanner": page_scanner()
    elif page == "🧪 回测": page_backtest()
    elif page == "🔌 数据源": page_sources()
    st.divider()
    st.caption("Research tool only. Real-time status depends on provider plan/exchange entitlements. No model or data feed can guarantee a future price or capture every market-moving event.")


# Top-level crash shield: even unexpected exceptions render a visible diagnostic instead of a blank page.
try:
    main()
except Exception as exc:
    st.error("应用遇到未预期错误，但没有崩溃/黑屏。请展开下面的错误详情。")
    st.write(f"**{type(exc).__name__}:** {exc}")
    with st.expander("错误详情（可截图发给 ChatGPT）", expanded=True):
        st.code(traceback.format_exc()[-6000:])
    st.info("可以先点击左侧『清除缓存』，或关闭较慢的 Options 后重试。")
