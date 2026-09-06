import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.market_data import MarketDataService
from features.technical import add_technical_indicators


def render():
    st.title('Stock Analysis')
    ticker = st.text_input('Ticker', 'AVGO').upper().strip()
    interval = st.selectbox('Interval', ['1d','1h','15m','5m'])
    period = st.selectbox('Period', ['1mo','3mo','6mo','1y','2y','5y'], index=4)
    if st.button('Load market data', type='primary'):
        with st.spinner('Fetching market data...'):
            df, provider = MarketDataService().get_history(ticker, period=period, interval=interval)
            feat = add_technical_indicators(df)
        st.success(f'Data provider: {provider} | rows: {len(df)}')
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close)])
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(feat.tail(50), use_container_width=True)
