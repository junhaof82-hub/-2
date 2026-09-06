from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

import json
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from data.market_data import MarketDataService
from data.news import NewsService
from features.sentiment import NewsSentimentAnalyzer
from features.technical import add_technical_indicators
from prediction.pro_engine import ProAnalysisEngine
from prediction.ranking import scan
from backtest.backtest import run_backtest
from backtest.walk_forward import run_walk_forward
from storage.database import Database

st.set_page_config(page_title='Stock AI MAX', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
.block-container {padding-top:1rem; padding-bottom:3rem; max-width:1540px;}
[data-testid="stMetricValue"] {font-size:1.55rem;}
.hero {padding:18px 21px; border:1px solid rgba(128,128,128,.25); border-radius:18px; margin-bottom:14px;}
.badge {display:inline-block; padding:3px 9px; margin:2px 3px; border:1px solid rgba(128,128,128,.28); border-radius:999px; font-size:.82rem;}
.note {opacity:.72; font-size:.88rem;}
</style>
''', unsafe_allow_html=True)


def fmt_money(v):
    try:
        if v is None or not math.isfinite(float(v)): return '—'
        return f'${float(v):,.2f}'
    except Exception: return '—'


def fmt_pct(v, digits=1, already_pct=False):
    try:
        if v is None or not math.isfinite(float(v)): return '—'
        x = float(v) if already_pct else float(v)*100
        return f'{x:.{digits}f}%'
    except Exception: return '—'


def fmt_big(v):
    try:
        x = float(v)
        if not math.isfinite(x): return '—'
        if abs(x)>=1e12: return f'${x/1e12:.2f}T'
        if abs(x)>=1e9: return f'${x/1e9:.2f}B'
        if abs(x)>=1e6: return f'${x/1e6:.2f}M'
        return f'${x:,.0f}'
    except Exception: return '—'


def prob_label(p):
    p=float(p)
    if p>=.65: return '明显偏多'
    if p>=.56: return '轻度偏多'
    if p<=.35: return '明显偏空'
    if p<=.44: return '轻度偏空'
    return '中性'


@st.cache_resource
def get_engine(): return ProAnalysisEngine()

@st.cache_resource
def get_market(): return MarketDataService()

@st.cache_resource
def get_db(): return Database()

@st.cache_resource
def get_news_service(): return NewsService()

@st.cache_resource
def get_sentiment(): return NewsSentimentAnalyzer()

@st.cache_data(ttl=900, show_spinner=False)
def cached_max_analysis(ticker: str, force_train: bool = False):
    return get_engine().analyze(ticker, force_train=force_train)

@st.cache_data(ttl=600, show_spinner=False)
def chart_data(ticker: str):
    prices, provider = get_market().get_history(ticker, period='1y', interval='1d')
    return add_technical_indicators(prices).tail(220), provider


with st.sidebar:
    st.title('📈 Stock AI MAX')
    st.caption('Cloud Safe · 概率校准 · 多来源新闻 · SEC · Walk-forward')
    mode = st.radio('功能', [
        '🚀 一键深度分析', '🛰️ 新闻雷达', '🎯 目标价概率', '🏆 股票排行榜',
        '🧪 回测实验室', '📊 模型准确率', '⚙️ 数据源状态',
    ], index=0)
    st.divider()
    st.caption('云端稳定模式默认开启：限制CPU/内存，避免点击分析后实例崩溃。')
    st.caption('系统会在数据不足或模型分歧大时主动降低置信度，而不是硬给高概率。')
    st.caption('研究工具，不构成投资建议。')

st.markdown('<div class="hero"><h2 style="margin:0">Stock AI MAX · 美股多因子概率分析</h2><div class="note">技术面 + 市场上下文 + 基本面 + 多来源新闻/SEC + 宏观 + Options + 6模型校准融合 + Monte Carlo + 真实时间顺序回测</div></div>', unsafe_allow_html=True)

engine = get_engine()

if mode == '🚀 一键深度分析':
    c1,c2,c3 = st.columns([2.2,1,1])
    ticker = c1.text_input('股票代码', value='NVDA', placeholder='NVDA / AVGO / SNDK / AMZN').strip().upper()
    force_train = c2.toggle('重新训练模型', value=False, help='模型版本升级或你想强制刷新时打开。')
    run = c3.button('开始 MAX 分析', type='primary', use_container_width=True)
    if run:
        try:
            with st.spinner(f'正在分析 {ticker}：历史行情 → 6个ML模型 → 新闻/SEC → 宏观 → Options → 多周期… 第一次会慢一些。'):
                result = cached_max_analysis(ticker, force_train=force_train)
            st.session_state['max_result'] = result
        except Exception as e:
            st.session_state.pop('max_result', None)
            st.error(f'分析失败：{type(e).__name__}: {e}')
            st.info('最常见原因是网络数据源暂时不可用。系统会自动切换数据源，但所有源都失败时仍会停止。')

    result = st.session_state.get('max_result')
    if result and result.get('ticker') == ticker:
        pro=result['pro']; probs=result['probabilities']; conf=pro['confidence']; news=result.get('news',{}); opt=result.get('options',{})
        st.subheader(f"{ticker} · {pro['stance']} · 信号质量：{pro['signal_quality']}")
        cols=st.columns(8)
        vals=[
            ('最新价',fmt_money(result['last_price'])),('1日上涨',fmt_pct(probs[1])),('3日上涨',fmt_pct(probs[3])),('5日上涨',fmt_pct(probs[5])),
            ('5日置信度',f"{conf[5]:.0f}/100"),('风险',f"{result['risk_score']:.1f}/10"),('数据质量',f"{pro['data_quality']:.0f}/100"),('新闻覆盖',f"{news.get('coverage_score',0):.0f}/100"),
        ]
        for col,(lab,val) in zip(cols,vals): col.metric(lab,val)

        if conf[5] < 50:
            st.warning('模型分歧或历史外推质量不足：当前概率应视为低置信度，系统不会把它包装成强信号。')
        if news.get('breaking_news_count',0):
            st.info(f"过去约12小时检测到 {news.get('breaking_news_count')} 条高重要度事件/新闻，短线波动风险更高。")

        left,right=st.columns([2.15,1])
        with left:
            try:
                frame,provider=chart_data(ticker); fig=go.Figure()
                fig.add_trace(go.Candlestick(x=frame.index,open=frame.open,high=frame.high,low=frame.low,close=frame.close,name='K线'))
                for col,name in [('ema_20','EMA20'),('ema_50','EMA50'),('ema_200','EMA200')]:
                    if col in frame: fig.add_trace(go.Scatter(x=frame.index,y=frame[col],mode='lines',name=name,line={'width':1.2}))
                fig.update_layout(height=455,margin=dict(l=5,r=5,t=10,b=5),xaxis_rangeslider_visible=False,legend_orientation='h')
                st.plotly_chart(fig,use_container_width=True); st.caption(f'行情数据源：{provider}')
            except Exception as e: st.warning(f'图表不可用：{e}')
        with right:
            st.markdown('#### MAX 决策面板')
            st.metric('综合评分',f"{pro['pro_score']:.0f}/100")
            st.metric('多周期技术',f"{pro['multi_timeframe']['aggregate_score']:.0f}/100")
            st.metric('市场环境',pro['market_regime']['label'])
            st.metric('相对强度',f"{result.get('relative_strength',{}).get('score',50):.0f}/100")
            if opt.get('available'):
                st.metric('Options 情绪',f"{opt.get('options_score',50):.0f}/100")
                st.caption(f"最近到期 {opt.get('expiry')} · 隐含波动 {fmt_pct(opt.get('atm_iv'))} · 隐含波幅 {fmt_pct(opt.get('implied_move_pct'))}")
            else: st.caption('Options 数据当前不可用，不会强行计入。')

        tabs=st.tabs(['AI概率/情景','新闻与SEC','多周期技术','基本面/Options','宏观/相对强度','模型明细'])
        with tabs[0]:
            rows=[]
            for h in (1,3,5):
                sc=pro['scenarios'][h]; dist=result.get('scenario_distributions',{}).get(h,{})
                rows.append({'周期':f'{h}个交易日','上涨概率':fmt_pct(probs[h]),'判断':prob_label(probs[h]),'置信度':f"{conf[h]:.0f}/100",
                             'Bear(P10)':fmt_money(sc['bear']),'Base(P50)':fmt_money(sc['base']),'Bull(P90)':fmt_money(sc['bull']),
                             '极端下沿(P05)':fmt_money(dist.get('p05')),'极端上沿(P95)':fmt_money(dist.get('p95'))})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.caption('P10/P50/P90 来自经验分布 Monte Carlo，并用模型方向做小幅漂移调整；不是保证价格。')
            st.markdown('##### 模型融合权重')
            weight_rows=[]
            for h in (1,3,5):
                rr={'周期':f'{h}日'}; rr.update({k:f'{v*100:.1f}%' for k,v in result['component_weights'][h].items()}); weight_rows.append(rr)
            st.dataframe(pd.DataFrame(weight_rows).fillna('—'),hide_index=True,use_container_width=True)

        with tabs[1]:
            n1,n2,n3,n4=st.columns(4)
            n1.metric('新闻情绪',f"{news.get('sentiment_score',0):+.0f}/100")
            n2.metric('活跃数据源',news.get('source_count',0))
            n3.metric('覆盖评分',f"{news.get('coverage_score',0):.0f}/100")
            n4.metric('高重要度突发',news.get('breaking_news_count',0))
            srcs=news.get('active_sources',[])
            if srcs: st.markdown(''.join([f'<span class="badge">{s}</span>' for s in srcs]),unsafe_allow_html=True)
            a,b=st.columns(2)
            with a:
                st.markdown('##### 主要催化剂')
                for x in result.get('catalysts') or ['暂无明确催化剂']: st.write(f'• {x}')
            with b:
                st.markdown('##### 主要风险')
                for x in result.get('risks') or ['暂无明确风险']: st.write(f'• {x}')
            st.divider()
            for item in news.get('items',[])[:18]:
                src=item.get('source',''); label=item.get('label','neutral'); imp=item.get('importance',0); age=item.get('age_hours')
                age_text=f'{age:.0f}h前' if isinstance(age,(int,float)) else ''
                st.markdown(f"**{item.get('title','')}**  · `{label}` · 重要度 {imp:.1f}/10 · `{src}` · {age_text}")
                if item.get('summary'): st.caption(str(item['summary'])[:450])

        with tabs[2]:
            tf=[]
            for interval in ('1d','1h','15m','5m'):
                x=pro['multi_timeframe']['items'][interval]
                tf.append({'周期':x['label'],'趋势':x['trend'],'评分':'—' if x['score'] is None else f"{x['score']:.0f}/100",'RSI':'—' if x.get('rsi') is None else f"{x['rsi']:.1f}",'ADX':'—' if x.get('adx') is None else f"{x['adx']:.1f}",'K线数':x.get('bars',0),'数据源':x.get('provider') or '—'})
            st.dataframe(pd.DataFrame(tf),hide_index=True,use_container_width=True)
            tech=result.get('latest_technical',{})
            keytech=['rsi_14','macd_hist','adx_14','atr_pct','volume_ratio','return_5d','volatility_20','ema_20','ema_50','ema_200']
            st.dataframe(pd.DataFrame([{'指标':k,'数值':tech.get(k)} for k in keytech]),hide_index=True,use_container_width=True)

        with tabs[3]:
            f=result.get('fundamentals',{})
            fund=[
                ('Revenue',fmt_big(f.get('revenue'))),('EPS',f"{f.get('eps'):.2f}" if isinstance(f.get('eps'),(int,float)) and math.isfinite(f.get('eps')) else '—'),
                ('EPS surprise',fmt_pct(f.get('eps_surprise'),already_pct=True)),('Forward PE',f"{f.get('forward_pe'):.1f}" if isinstance(f.get('forward_pe'),(int,float)) and math.isfinite(f.get('forward_pe')) else '—'),
                ('Market Cap',fmt_big(f.get('market_cap'))),('Free Cash Flow',fmt_big(f.get('free_cash_flow'))),('Gross Margin',fmt_pct(f.get('gross_margin'))),('Operating Margin',fmt_pct(f.get('operating_margin'))),
                ('Revenue Growth',fmt_pct(f.get('revenue_growth'))),('Earnings Growth',fmt_pct(f.get('earnings_growth'))),('Short % Float',fmt_pct(f.get('short_percent_float'))),('Days to Earnings',f"{f.get('days_to_earnings'):.0f}" if isinstance(f.get('days_to_earnings'),(int,float)) and math.isfinite(f.get('days_to_earnings')) else '—'),
                ('Analyst Target',fmt_money(f.get('analyst_target_mean'))),('Analyst Count',f"{f.get('analyst_count'):.0f}" if isinstance(f.get('analyst_count'),(int,float)) and math.isfinite(f.get('analyst_count')) else '—'),
            ]
            st.dataframe(pd.DataFrame(fund,columns=['指标','数值']),hide_index=True,use_container_width=True)
            if opt.get('available'):
                st.markdown('##### Options')
                st.dataframe(pd.DataFrame([{
                    '到期日':opt.get('expiry'),'DTE':opt.get('days_to_expiry'),'Put/Call Volume':opt.get('put_call_volume'),'Put/Call OI':opt.get('put_call_open_interest'),
                    'ATM IV':fmt_pct(opt.get('atm_iv')),'隐含波幅':fmt_pct(opt.get('implied_move_pct')),'Options评分':f"{opt.get('options_score',50):.0f}/100"
                }]),hide_index=True,use_container_width=True)

        with tabs[4]:
            m=result.get('macro',{}); rs=result.get('relative_strength',{})
            macro_rows=[('SPY 5日',fmt_pct(m.get('spy_5d_return'))),('QQQ 5日',fmt_pct(m.get('qqq_5d_return'))),('SMH 5日',fmt_pct(m.get('semiconductors_5d_return'))),('VIX',m.get('vix')),('10Y',m.get('treasury_10y')),('Fed Funds',m.get('fed_funds_rate')),('CPI YoY',fmt_pct(m.get('cpi_yoy'),already_pct=True)),('Core CPI YoY',fmt_pct(m.get('core_cpi_yoy'),already_pct=True)),('失业率',fmt_pct(m.get('unemployment_rate'),already_pct=True)),('10Y-2Y',m.get('yield_curve_10y2y')),('Fed状态',m.get('fed_policy_bias'))]
            st.dataframe(pd.DataFrame(macro_rows,columns=['指标','数值']),hide_index=True,use_container_width=True)
            rel=rs.get('relative',{})
            st.markdown('##### 相对强弱')
            st.dataframe(pd.DataFrame([{'比较':k,'超额收益':fmt_pct(v)} for k,v in rel.items()]),hide_index=True,use_container_width=True)
            st.write('市场环境因素：',' · '.join(pro['market_regime']['reasons']) or '暂无')

        with tabs[5]:
            rows=[]
            for h in (1,3,5):
                for name,p in result['model_probabilities'][h].items():
                    met=result['model_metrics'][h].get(name,{})
                    rows.append({'周期':f'{h}日','模型':name,'上涨概率':fmt_pct(p),'Accuracy':met.get('accuracy'),'Balanced Acc':met.get('balanced_accuracy'),'ROC AUC':met.get('roc_auc'),'Brier(越低越好)':met.get('brier_score'),'LogLoss':met.get('log_loss')})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.caption('模型使用时间顺序 Train → Calibration → Test，概率经过校准；测试集不是随机打乱。')

elif mode == '🛰️ 新闻雷达':
    ticker=st.text_input('股票代码',value='NVDA').strip().upper()
    if st.button('扫描全部可用新闻源',type='primary'):
        try:
            with st.spinner('并行检查 yfinance / SEC / 已配置新闻 API…'):
                cov=get_news_service().get_with_coverage(ticker,limit=settings.news_max_items); n=get_sentiment().analyze(cov['items'],coverage=cov)
            st.session_state['news_radar']=(ticker,n)
        except Exception as e: st.error(str(e))
    saved=st.session_state.get('news_radar')
    if saved and saved[0]==ticker:
        n=saved[1]; c1,c2,c3,c4=st.columns(4)
        c1.metric('综合情绪',f"{n['sentiment_score']:+.0f}/100"); c2.metric('文章/事件',len(n['items'])); c3.metric('活跃源',n['source_count']); c4.metric('覆盖',f"{n['coverage_score']:.0f}/100")
        st.markdown(''.join([f'<span class="badge">{s}</span>' for s in n.get('active_sources',[])]),unsafe_allow_html=True)
        for x in n['items']:
            st.markdown(f"**{x.get('title','')}** · `{x.get('source','')}` · `{x.get('label','neutral')}` · 重要度 {x.get('importance',0):.1f}/10")
            if x.get('summary'): st.caption(str(x['summary'])[:500])

elif mode == '🎯 目标价概率':
    c1,c2=st.columns(2); ticker=c1.text_input('股票代码',value='SNDK').strip().upper(); target=c2.number_input('目标价格',min_value=.01,value=170.0,step=1.0)
    condition=st.toggle('使用当前AI方向概率调整模拟',value=True,help='会先运行/复用AI分析，再对经验分布漂移做小幅调整，不会强行改变波动率。')
    if st.button('计算触及概率',type='primary'):
        try:
            p=None
            if condition:
                last=st.session_state.get('max_result')
                if last and last.get('ticker')==ticker: p=last['probabilities'][5]
                else:
                    with st.spinner('先计算AI方向概率…'): last=engine.analyze(ticker); p=last['probabilities'][5]; st.session_state['max_result']=last
            with st.spinner('运行经验分布 Monte Carlo…'): r=engine.predictor.target_touch(ticker,target,direction_prob=p)
            st.metric('当前价',fmt_money(r['last_price'])); cols=st.columns(3)
            for col,h in zip(cols,(1,3,5)): col.metric(f'{h}日内触及 {fmt_money(target)}',fmt_pct(r['probabilities'][h]))
        except Exception as e: st.error(f'计算失败：{e}')

elif mode == '🏆 股票排行榜':
    text=st.text_area('股票列表（逗号分隔）',value=','.join(settings.default_tickers),height=90); horizon=st.selectbox('周期',[1,3,5],index=2,format_func=lambda x:f'{x}个交易日')
    if st.button('开始 MAX 扫描',type='primary'):
        tickers=[x.strip().upper() for x in text.replace('\n',',').split(',') if x.strip()][:25]
        try:
            with st.spinner('逐只分析；首次会训练模型，股票越多耗时越长…'): df=scan(tickers,predictor=engine.predictor,horizon=horizon)
            st.dataframe(df,hide_index=True,use_container_width=True)
        except Exception as e: st.error(str(e))

elif mode == '🧪 回测实验室':
    c1,c2,c3=st.columns(3); ticker=c1.text_input('Ticker',value='NVDA').strip().upper(); horizon=c2.selectbox('预测周期',[1,3,5]); method=c3.selectbox('方法',['快速时间顺序 Holdout','Walk-forward（更严格）'])
    if st.button('运行回测',type='primary'):
        try:
            with st.spinner('获取5年历史并按时间顺序回测…'):
                prices,_=get_market().get_history(ticker,period='5y',interval='1d')
                r=run_walk_forward(prices,horizon=horizon,max_windows=12) if method.startswith('Walk') else run_backtest(prices,horizon=horizon)
            rows=[]
            for name,met in r.items():
                if name.startswith('_'): continue
                rows.append({'模型':name,'Accuracy':met.get('prediction_accuracy'),'Balanced Acc':met.get('balanced_accuracy'),'ROC AUC':met.get('roc_auc'),'Brier':met.get('brier_score'),'Win Rate':met.get('win_rate'),'Sharpe':met.get('sharpe_ratio'),'Max Drawdown':met.get('max_drawdown'),'Trades':met.get('trades')})
                get_db().save_backtest(ticker,horizon,name,met,method=r['_meta'].get('method'))
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True); st.json(r['_meta'])
            st.caption('Walk-forward 更接近真实使用：每个测试窗口只能使用当时之前的数据训练。历史表现不保证未来。')
        except Exception as e: st.error(f'回测失败：{e}')

elif mode == '📊 模型准确率':
    df=get_db().latest_accuracy()
    if df.empty: st.info('还没有准确率记录。先运行分析或每日学习，之后这里会积累真实结果。')
    else:
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.caption('这里既可能包含训练时的时间顺序测试，也可能包含每日预测兑现后的真实准确率。Sample Size 很小时不要过度解读。')

else:
    st.subheader('数据源状态')
    rows=[
        ('yfinance','免费基础行情/新闻','无需Key','✅'),('SEC EDGAR','官方公司申报/重大文件','建议设置 SEC_USER_AGENT','✅' if settings.sec_user_agent else '⚠️ 建议配置'),
        ('Alpha Vantage','行情 + 新闻情绪','ALPHA_VANTAGE_API_KEY','✅' if settings.alpha_vantage_api_key else '未配置'),('FMP','行情/财报/新闻','FMP_API_KEY','✅' if settings.fmp_api_key else '未配置'),
        ('Massive/Polygon','行情 + 新闻','POLYGON_API_KEY','✅' if settings.polygon_api_key else '未配置'),('Finnhub','公司新闻','FINNHUB_API_KEY','✅' if settings.finnhub_api_key else '未配置'),
        ('Marketaux','全球财经新闻聚合','MARKETAUX_API_KEY','✅' if settings.marketaux_api_key else '未配置'),('FRED','官方宏观历史','FRED_API_KEY（无Key也有CSV fallback）','✅' if settings.fred_api_key else 'Fallback'),
        ('OpenAI','可选LLM新闻分类','OPENAI_API_KEY','✅' if settings.openai_api_key else '本地情绪模型'),
    ]
    st.dataframe(pd.DataFrame(rows,columns=['数据源','用途','配置','状态']),hide_index=True,use_container_width=True)
    st.markdown('#### 在线网页部署')
    st.write('本项目已经是 Streamlit Cloud / Render / Railway 可部署结构。最简单的方式是把整个文件夹上传到 GitHub，再在 Streamlit Community Cloud 选择 `app.py`。API Key 放到云端 Secrets，不要写进代码。')
    st.code('streamlit run app.py',language='bash')
    st.caption('你不需要在 Windows 上运行 BAT；云端部署成功后，以后只打开网址。')

st.divider(); st.caption('⚠️ “最高准确率”不能被保证。MAX 版重点是减少数据泄漏、校准概率、增加新闻/SEC/Options/宏观覆盖，并在低质量数据时降低置信度。')
