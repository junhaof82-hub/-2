from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Stock AI MAX Cloud", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.1rem;max-width:1500px}
.hero{padding:18px 22px;border:1px solid rgba(128,128,128,.25);border-radius:18px;margin-bottom:14px}
.small{font-size:.86rem;opacity:.72}.pill{display:inline-block;padding:3px 9px;border:1px solid rgba(128,128,128,.3);border-radius:999px;margin:2px}
[data-testid="stMetricValue"]{font-size:1.55rem}
</style>
""", unsafe_allow_html=True)

# ---------------------------- data helpers ----------------------------

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        # yfinance may return ticker as a second level
        x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
    x = x.rename(columns={c: c.title() for c in x.columns})
    needed = [c for c in ["Open","High","Low","Close","Volume"] if c in x.columns]
    x = x[needed].copy()
    for c in needed:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=["Close"]).sort_index()

@st.cache_data(ttl=600, show_spinner=False)
def get_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    try:
        return _clean_ohlcv(yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False))
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def get_multi_history(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    out = {}
    try:
        raw = yf.download(list(tickers), period=period, interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=False)
        for t in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0):
                    out[t] = _clean_ohlcv(raw[t])
                else:
                    out[t] = get_history(t, period, "1d")
            except Exception:
                out[t] = pd.DataFrame()
    except Exception:
        for t in tickers:
            out[t] = get_history(t, period, "1d")
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def get_info(ticker: str) -> dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info or {}
        return {str(k): v for k, v in info.items()}
    except Exception:
        return {}

@st.cache_data(ttl=900, show_spinner=False)
def get_news(ticker: str, limit: int = 30) -> list[dict[str, Any]]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        raw = []
    items = []
    seen = set()
    for n in raw:
        c = n.get("content") if isinstance(n, dict) else None
        if isinstance(c, dict):
            title = c.get("title") or ""
            summary = c.get("summary") or c.get("description") or ""
            provider = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else "Yahoo Finance"
            url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else ""
            pub = c.get("pubDate") or c.get("displayTime") or ""
        else:
            title = n.get("title", "") if isinstance(n, dict) else ""
            summary = n.get("summary", "") if isinstance(n, dict) else ""
            provider = n.get("publisher", "Yahoo Finance") if isinstance(n, dict) else "Yahoo Finance"
            url = n.get("link", "") if isinstance(n, dict) else ""
            pub = n.get("providerPublishTime", "") if isinstance(n, dict) else ""
        key = re.sub(r"\W+", "", title.lower())[:120]
        if not title or key in seen:
            continue
        seen.add(key)
        items.append({"title": title, "summary": summary, "provider": provider, "url": url, "published": pub})
        if len(items) >= limit:
            break
    return items

# ---------------------------- indicators ----------------------------

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c, h, l, v = x["Close"], x.get("High", x["Close"]), x.get("Low", x["Close"]), x.get("Volume", pd.Series(index=x.index, dtype=float))
    ret = c.pct_change()
    x["ret1"] = ret
    x["ret3"] = c.pct_change(3)
    x["ret5"] = c.pct_change(5)
    x["ret10"] = c.pct_change(10)
    x["ret20"] = c.pct_change(20)
    for n in (9,20,50,200): x[f"ema{n}"] = ema(c,n)
    for n in (20,50,200): x[f"sma{n}"] = sma(c,n)
    delta = c.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean().replace(0,np.nan)
    x["rsi14"] = 100 - 100/(1+rs)
    macd = ema(c,12)-ema(c,26); sig=ema(macd,9)
    x["macd"] = macd; x["macd_signal"] = sig; x["macd_hist"] = macd-sig
    mid=sma(c,20); std=c.rolling(20).std()
    x["bb_mid"]=mid; x["bb_up"]=mid+2*std; x["bb_low"]=mid-2*std; x["bb_pos"]=(c-x["bb_low"])/(x["bb_up"]-x["bb_low"]).replace(0,np.nan)
    prev=c.shift(1); tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    x["atr14"]=tr.rolling(14).mean(); x["atr_pct"]=x["atr14"]/c
    tp=(h+l+c)/3
    if v.notna().any():
        x["vwap20"]=(tp*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
        x["volume_ratio"] = v / v.rolling(20).mean().replace(0,np.nan)
        x["volume_z"]=(v-v.rolling(20).mean())/v.rolling(20).std().replace(0,np.nan)
    else:
        x["vwap20"]=np.nan; x["volume_ratio"]=np.nan; x["volume_z"]=np.nan
    x["momentum10"] = c-c.shift(10); x["roc10"] = c.pct_change(10)*100
    x["volatility20"] = ret.rolling(20).std()*np.sqrt(252)
    x["support20"] = l.rolling(20).min(); x["resistance20"] = h.rolling(20).max()
    x["dist_ema20"] = c/x["ema20"]-1; x["dist_ema50"] = c/x["ema50"]-1; x["dist_ema200"] = c/x["ema200"]-1
    x["trend20"] = x["ema20"].pct_change(10); x["trend50"] = x["ema50"].pct_change(20)
    return x.replace([np.inf,-np.inf],np.nan)

# ---------------------------- models ----------------------------
FEATURES = ["ret1","ret3","ret5","ret10","ret20","rsi14","macd","macd_hist","bb_pos","atr_pct","volume_ratio","volume_z","roc10","volatility20","dist_ema20","dist_ema50","dist_ema200","trend20","trend50"]

@dataclass
class HorizonResult:
    horizon: int
    up_probability: float
    model_probs: dict[str,float]
    metrics: dict[str,float]
    expected_return: float


def _model_bank():
    return {
        "Logistic": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=400, C=.7, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=1, class_weight="balanced_subsample"),
        "Extra Trees": ExtraTreesClassifier(n_estimators=180, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=1, class_weight="balanced"),
        "Hist Gradient": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=.055, random_state=42),
    }


def _make_dataset(ind: pd.DataFrame, h: int):
    d = ind.copy()
    d["future_return"] = d["Close"].shift(-h)/d["Close"]-1
    d["target"] = (d["future_return"]>0).astype(int)
    d = d.dropna(subset=FEATURES+["future_return"])
    return d

@st.cache_data(ttl=3600, show_spinner=False)
def train_predict(ticker: str, horizon: int) -> HorizonResult | None:
    raw=get_history(ticker,"5y","1d")
    if len(raw)<320: return None
    ind=add_indicators(raw); d=_make_dataset(ind,horizon)
    if len(d)<220: return None
    n=len(d); split=max(int(n*.78), n-140); split=min(split,n-70)
    train=d.iloc[:split]; test=d.iloc[split:]
    Xtr=train[FEATURES].fillna(0); ytr=train["target"]
    Xte=test[FEATURES].fillna(0); yte=test["target"]
    latest=ind[FEATURES].dropna().tail(1)
    if latest.empty: return None
    probs={}; weights=[]; pvals=[]; aucs=[]; accs=[]
    for name,base in _model_bank().items():
        try:
            m=clone(base); m.fit(Xtr,ytr)
            pt=m.predict_proba(Xte)[:,1]
            pl=float(m.predict_proba(latest.fillna(0))[:,1][0])
            pred=(pt>=.5).astype(int)
            acc=accuracy_score(yte,pred)
            try: auc=roc_auc_score(yte,pt)
            except Exception: auc=.5
            w=max(.05, .5*(acc-.45)+.5*(auc-.45))
            probs[name]=pl; pvals.append(pl); weights.append(w); aucs.append(auc); accs.append(acc)
        except Exception:
            continue
    if not pvals: return None
    p=float(np.average(pvals,weights=weights))
    er=float(d["future_return"].tail(500).mean())
    metrics={"accuracy":float(np.mean(accs)),"roc_auc":float(np.mean(aucs)),"test_n":int(len(test))}
    return HorizonResult(horizon,p,probs,metrics,er)

# ---------------------------- sentiment / fundamentals / macro ----------------------------
POS_WORDS={"beat","beats","growth","record","upgrade","surge","strong","profit","profits","bullish","demand","partnership","approval","buyback","raises","raised","outperform","accelerates","expands","launch","wins","win","optimistic"}
NEG_WORDS={"miss","misses","downgrade","fall","falls","weak","loss","lawsuit","probe","investigation","recall","cuts","cut","warning","bearish","decline","slump","risk","delay","delays","antitrust","tariff","fraud"}
CATALYST_WORDS={"earnings","guidance","ai","artificial intelligence","deal","acquisition","merger","approval","launch","contract","partnership","buyback","dividend","upgrade"}
RISK_WORDS={"lawsuit","probe","investigation","downgrade","tariff","antitrust","recall","warning","debt","valuation","delay"}

def score_news(items):
    scored=[]
    for it in items:
        text=(it.get("title","")+" "+it.get("summary","")).lower()
        toks=set(re.findall(r"[a-z]+",text))
        pos=len(toks&POS_WORDS); neg=len(toks&NEG_WORDS)
        score=max(-100,min(100,(pos-neg)*22))
        importance=min(10,2+2*sum(1 for k in CATALYST_WORDS if k in text)+1.5*sum(1 for k in RISK_WORDS if k in text))
        scored.append({**it,"score":score,"importance":importance})
    if scored:
        w=np.array([max(1,s["importance"]) for s in scored],float); vals=np.array([s["score"] for s in scored],float)
        overall=float(np.average(vals,weights=w))
    else: overall=0.0
    cats=[k for k in CATALYST_WORDS if any(k in (s["title"]+" "+s["summary"]).lower() for s in scored)]
    risks=[k for k in RISK_WORDS if any(k in (s["title"]+" "+s["summary"]).lower() for s in scored)]
    return overall,scored,cats[:8],risks[:8]

def fundamental_score(info):
    score=50.0; notes=[]
    rg=info.get("revenueGrowth"); eg=info.get("earningsGrowth"); gm=info.get("grossMargins"); om=info.get("operatingMargins"); fpe=info.get("forwardPE"); fcf=info.get("freeCashflow")
    if isinstance(rg,(int,float)): score+=np.clip(rg*45,-12,12); notes.append(f"Revenue growth {rg*100:.1f}%")
    if isinstance(eg,(int,float)): score+=np.clip(eg*30,-10,10); notes.append(f"Earnings growth {eg*100:.1f}%")
    if isinstance(om,(int,float)): score+=np.clip((om-.10)*20,-5,7); notes.append(f"Operating margin {om*100:.1f}%")
    if isinstance(fpe,(int,float)) and fpe>0:
        score += 4 if fpe<25 else (-5 if fpe>60 else 0); notes.append(f"Forward P/E {fpe:.1f}")
    if isinstance(fcf,(int,float)): score += 3 if fcf>0 else -5
    return float(np.clip(score,0,100)),notes

@st.cache_data(ttl=600, show_spinner=False)
def macro_snapshot():
    tickers=("SPY","QQQ","^VIX","^TNX","DX-Y.NYB","SMH")
    d=get_multi_history(tickers,"6mo")
    rows=[]; score=50
    for t in tickers:
        x=d.get(t,pd.DataFrame())
        if len(x)>21:
            last=float(x.Close.iloc[-1]); r5=float(x.Close.pct_change(5).iloc[-1]); r20=float(x.Close.pct_change(20).iloc[-1])
            rows.append({"Asset":t,"Last":last,"5D":r5,"20D":r20})
    mp={r["Asset"]:r for r in rows}
    score += np.clip(mp.get("SPY",{}).get("20D",0)*120,-8,8)
    score += np.clip(mp.get("QQQ",{}).get("20D",0)*120,-8,8)
    score += np.clip(mp.get("SMH",{}).get("20D",0)*80,-5,5)
    vix=mp.get("^VIX",{}).get("Last")
    if vix: score += 6 if vix<17 else (-8 if vix>28 else 0)
    tnx=mp.get("^TNX",{}).get("20D",0); score += np.clip(-tnx*80,-5,5)
    return float(np.clip(score,0,100)),rows

# ---------------------------- Monte Carlo ----------------------------

def bootstrap_paths(close: pd.Series, horizon: int, n_paths: int=6000, seed: int=42):
    rets=close.pct_change().dropna().tail(756).values
    if len(rets)<80: return None
    rng=np.random.default_rng(seed); sampled=rng.choice(rets,size=(n_paths,horizon),replace=True)
    start=float(close.iloc[-1]); paths=start*np.cumprod(1+sampled,axis=1); terminal=paths[:,-1]
    return paths,terminal

def range_stats(close,horizon):
    sim=bootstrap_paths(close,horizon)
    if sim is None:return {}
    _,term=sim; q=np.quantile(term,[.05,.1,.5,.9,.95])
    return {"p05":q[0],"p10":q[1],"p50":q[2],"p90":q[3],"p95":q[4],"up_prob":float(np.mean(term>close.iloc[-1]))}

def touch_probability(close,target,horizon):
    sim=bootstrap_paths(close,horizon)
    if sim is None:return np.nan
    paths,_=sim; start=float(close.iloc[-1])
    return float(np.mean(np.max(paths,axis=1)>=target)) if target>=start else float(np.mean(np.min(paths,axis=1)<=target))

# ---------------------------- combined analysis ----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def analyze(ticker: str):
    ticker=ticker.upper().strip()
    raw=get_history(ticker,"5y","1d")
    if raw.empty or len(raw)<260: raise ValueError("无法取得足够历史数据。请检查股票代码或稍后重试。")
    ind=add_indicators(raw)
    hres={h:train_predict(ticker,h) for h in (1,3,5)}
    info=get_info(ticker); fscore,fnotes=fundamental_score(info)
    news=get_news(ticker,30); nscore,scored,cats,risks=score_news(news)
    mscore,mrows=macro_snapshot()
    last=ind.iloc[-1]; tech=50.0
    tech += 8 if last.get("Close",0)>last.get("ema20",np.inf) else -8
    tech += 6 if last.get("Close",0)>last.get("ema50",np.inf) else -6
    tech += 5 if last.get("Close",0)>last.get("ema200",np.inf) else -5
    rsi=last.get("rsi14",50)
    tech += 4 if 52<=rsi<=68 else (-5 if rsi>78 else (3 if rsi<35 else 0))
    tech += 5 if last.get("macd_hist",0)>0 else -5
    tech=float(np.clip(tech,0,100))
    news01=np.clip(50+nscore*.35,0,100)
    probs={h:(hres[h].up_probability if hres[h] else .5) for h in (1,3,5)}
    fused={}
    for h in (1,3,5):
        ml=probs[h]*100
        fused[h]=float(np.clip((.62*ml+.18*tech+.08*fscore+.07*news01+.05*mscore)/100,0.05,.95))
    disagreement=np.std([hres[5].model_probs[k] for k in hres[5].model_probs]) if hres.get(5) else .15
    data_quality=100
    if len(raw)<700:data_quality-=8
    if not news:data_quality-=15
    if not info:data_quality-=12
    confidence=float(np.clip(82-disagreement*120-abs(fused[5]-.5)*-15 + (data_quality-80)*.35,30,95))
    vol=float(ind["volatility20"].iloc[-1]) if pd.notna(ind["volatility20"].iloc[-1]) else .35
    risk=float(np.clip(3.2+vol*7+(100-confidence)/25+(8 if rsi>78 else 0)/10,1,10))
    ranges={h:range_stats(raw.Close,h) for h in (1,3,5)}
    return {"ticker":ticker,"raw":raw,"ind":ind,"hres":hres,"fused":fused,"tech":tech,"fund":fscore,"news_score":nscore,"macro":mscore,"confidence":confidence,"data_quality":data_quality,"risk":risk,"info":info,"fnotes":fnotes,"news":scored,"cats":cats,"risks":risks,"macro_rows":mrows,"ranges":ranges}

# ---------------------------- UI ----------------------------
with st.sidebar:
    st.title("📈 Stock AI MAX")
    page=st.radio("功能",["🚀 深度分析","🛰️ 新闻","🎯 目标价概率","🏆 Scanner","🧪 回测"],index=0)
    st.divider(); st.caption("Cloud Single-File 版：专门适配 GitHub 网页上传 + Streamlit Community Cloud。")
    st.caption("概率是模型估计，不是收益保证。")

st.markdown('<div class="hero"><h2 style="margin:0">Stock AI MAX · Cloud</h2><div class="small">技术面 + 基本面 + 新闻情绪 + SPY/QQQ/VIX/10Y/DXY/SMH + 4模型融合 + Monte Carlo + Backtest</div></div>',unsafe_allow_html=True)

if page=="🚀 深度分析":
    c1,c2=st.columns([3,1]); ticker=c1.text_input("股票代码","NVDA").upper().strip(); run=c2.button("开始分析",type="primary",use_container_width=True)
    if run:
        try:
            with st.spinner("正在下载行情、训练模型并分析新闻…"):
                r=analyze(ticker)
            st.session_state["last_result"]=r
        except Exception as e:
            st.error(f"分析失败：{e}")
    r=st.session_state.get("last_result")
    if r and r["ticker"]==ticker:
        p1,p3,p5=r["fused"][1],r["fused"][3],r["fused"][5]
        cols=st.columns(6)
        cols[0].metric("1日上涨概率",f"{p1*100:.1f}%")
        cols[1].metric("3日上涨概率",f"{p3*100:.1f}%")
        cols[2].metric("5日上涨概率",f"{p5*100:.1f}%")
        cols[3].metric("Confidence",f"{r['confidence']:.0f}/100")
        cols[4].metric("Risk",f"{r['risk']:.1f}/10")
        cols[5].metric("Data Quality",f"{r['data_quality']:.0f}/100")
        st.subheader("价格情景")
        rc=st.columns(3)
        for col,h in zip(rc,(1,3,5)):
            s=r["ranges"][h]
            col.markdown(f"**{h}D**")
            if s: col.write(f"P10 ${s['p10']:.2f} · P50 ${s['p50']:.2f} · P90 ${s['p90']:.2f}")
        ind=r["ind"].tail(220)
        fig=go.Figure(); fig.add_trace(go.Candlestick(x=ind.index,open=ind.Open,high=ind.High,low=ind.Low,close=ind.Close,name=ticker))
        fig.add_trace(go.Scatter(x=ind.index,y=ind.ema20,name="EMA20")); fig.add_trace(go.Scatter(x=ind.index,y=ind.ema50,name="EMA50"))
        fig.update_layout(height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True)
        a,b,c,d=st.columns(4); last=ind.iloc[-1]
        a.metric("RSI 14",f"{last.rsi14:.1f}"); b.metric("MACD Hist",f"{last.macd_hist:.3f}"); c.metric("ATR %",f"{last.atr_pct*100:.2f}%"); d.metric("Volume Ratio",f"{last.volume_ratio:.2f}x" if pd.notna(last.volume_ratio) else "—")
        st.subheader("多因子评分")
        sc=st.columns(4); sc[0].metric("Technical",f"{r['tech']:.0f}/100"); sc[1].metric("Fundamental",f"{r['fund']:.0f}/100"); sc[2].metric("News",f"{np.clip(50+r['news_score']*.35,0,100):.0f}/100"); sc[3].metric("Macro",f"{r['macro']:.0f}/100")
        st.subheader("各模型概率")
        rows=[]
        for h in (1,3,5):
            hr=r["hres"].get(h)
            if hr:
                for name,p in hr.model_probs.items(): rows.append({"Horizon":f"{h}D","Model":name,"Up Probability":p,"Test Accuracy":hr.metrics["accuracy"],"ROC AUC":hr.metrics["roc_auc"]})
        if rows: st.dataframe(pd.DataFrame(rows).style.format({"Up Probability":"{:.1%}","Test Accuracy":"{:.1%}","ROC AUC":"{:.3f}"}),use_container_width=True,hide_index=True)
        if r["cats"]: st.write("**主要催化剂：** "+", ".join(r["cats"]))
        if r["risks"]: st.write("**主要风险：** "+", ".join(r["risks"]))

elif page=="🛰️ 新闻":
    t=st.text_input("股票代码","NVDA",key="news_t").upper().strip()
    if st.button("刷新新闻",type="primary"):
        get_news.clear()
    items=get_news(t,35); score,scored,cats,risks=score_news(items)
    st.metric("综合新闻情绪",f"{score:+.0f} / 100")
    if cats: st.write("**催化剂：** "+", ".join(cats))
    if risks: st.write("**风险：** "+", ".join(risks))
    for n in scored:
        label="Bullish" if n["score"]>15 else ("Bearish" if n["score"]<-15 else "Neutral")
        st.markdown(f"**{label} · {n['score']:+.0f} · Importance {n['importance']:.0f}/10** — {n['title']}")
        if n.get("provider"): st.caption(str(n["provider"]))

elif page=="🎯 目标价概率":
    t=st.text_input("股票代码","SNDK",key="target_t").upper().strip(); raw=get_history(t,"3y","1d")
    if not raw.empty:
        last=float(raw.Close.iloc[-1]); target=st.number_input("目标价格",min_value=0.01,value=float(round(last*1.05,2)),step=.5)
        c=st.columns(3)
        for col,h in zip(c,(1,3,5)):
            p=touch_probability(raw.Close,target,h); col.metric(f"{h}日内触及 ${target:.2f}",f"{p*100:.1f}%" if pd.notna(p) else "—")
    else: st.warning("无法取得行情。")

elif page=="🏆 Scanner":
    default="NVDA,AVGO,AMAT,AMZN,GOOGL,META,MSFT,SNDK,AMD"
    text=st.text_area("股票列表",default,height=90); tickers=[x.strip().upper() for x in re.split(r"[,\s]+",text) if x.strip()][:15]
    if st.button("开始扫描",type="primary"):
        out=[]; bar=st.progress(0)
        for i,t in enumerate(tickers):
            try:
                r=analyze(t); out.append({"Ticker":t,"5D Up":r["fused"][5],"Confidence":r["confidence"],"Risk":r["risk"],"Technical":r["tech"],"News":r["news_score"]})
            except Exception: pass
            bar.progress((i+1)/max(1,len(tickers)))
        if out:
            df=pd.DataFrame(out).sort_values(["5D Up","Confidence"],ascending=False).reset_index(drop=True); df.index+=1
            st.dataframe(df.style.format({"5D Up":"{:.1%}","Confidence":"{:.0f}","Risk":"{:.1f}","Technical":"{:.0f}","News":"{:+.0f}"}),use_container_width=True)

elif page=="🧪 回测":
    t=st.text_input("股票代码","NVDA",key="bt_t").upper().strip(); h=st.selectbox("预测周期",[1,3,5],index=2)
    if st.button("运行时间顺序回测",type="primary"):
        raw=get_history(t,"5y","1d"); ind=add_indicators(raw); d=_make_dataset(ind,h)
        if len(d)<300: st.warning("历史数据不足")
        else:
            # Expanding-window, limited to 6 folds for cloud stability
            folds=[]; start=int(len(d)*.55); step=max(45,(len(d)-start)//6)
            for end in range(start,len(d)-step,step):
                tr=d.iloc[:end]; te=d.iloc[end:end+step]
                Xtr=tr[FEATURES].fillna(0); ytr=tr.target; Xte=te[FEATURES].fillna(0); yte=te.target
                ps=[]
                for base in _model_bank().values():
                    try:
                        m=clone(base); m.fit(Xtr,ytr); ps.append(m.predict_proba(Xte)[:,1])
                    except Exception: pass
                if ps:
                    p=np.mean(ps,axis=0); pred=(p>=.5).astype(int)
                    try: auc=roc_auc_score(yte,p)
                    except Exception: auc=np.nan
                    folds.append({"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"ROC AUC":auc,"N":len(te)})
            if folds:
                f=pd.DataFrame(folds); st.dataframe(f.style.format({c:"{:.1%}" for c in ["Accuracy","Precision","Recall"]}).format({"ROC AUC":"{:.3f}"}),use_container_width=True,hide_index=True)
                c=st.columns(4); c[0].metric("Accuracy",f"{f.Accuracy.mean():.1%}"); c[1].metric("Precision",f"{f.Precision.mean():.1%}"); c[2].metric("Recall",f"{f.Recall.mean():.1%}"); c[3].metric("ROC AUC",f"{f['ROC AUC'].mean():.3f}")

st.divider(); st.caption("Stock AI MAX Cloud is a research tool. Model probabilities and simulations are estimates, not guarantees or personalized investment advice.")
