import streamlit as st
import plotly.graph_objects as go
from data.market_data import MarketDataService
from features.technical import add_technical_indicators, technical_score


def render():
    st.title('Technical Analysis')
    ticker = st.text_input('Ticker', 'AMAT', key='tech_ticker').upper().strip()
    if st.button('Run technical analysis', type='primary'):
        df, provider = MarketDataService().get_history(ticker, period='1y', interval='1d')
        f = add_technical_indicators(df)
        score, bulls, bears = technical_score(f.iloc[-1])
        st.metric('Technical score', f'{score:.1f}/100')
        st.write('Bullish signals:', ', '.join(bulls) or 'None')
        st.write('Bearish signals:', ', '.join(bears) or 'None')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f.index, y=f['close'], name='Close'))
        fig.add_trace(go.Scatter(x=f.index, y=f['ema_20'], name='EMA20'))
        fig.add_trace(go.Scatter(x=f.index, y=f['ema_50'], name='EMA50'))
        fig.add_trace(go.Scatter(x=f.index, y=f['ema_200'], name='EMA200'))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(f.tail(30), use_container_width=True)
