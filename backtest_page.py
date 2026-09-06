import streamlit as st
import pandas as pd
from data.market_data import MarketDataService
from backtest.backtest import run_backtest
from storage.database import Database


def render():
    st.title('Backtest')
    ticker = st.text_input('Ticker', 'AVGO', key='bt_ticker').upper().strip()
    horizon = st.selectbox('Prediction horizon', [1,3,5])
    if st.button('Run backtest', type='primary'):
        with st.spinner('Running backtest...'):
            df, _ = MarketDataService().get_history(ticker, period='5y', interval='1d')
            r = run_backtest(df, horizon=horizon)
            db = Database()
            for name, metrics in r.items():
                if not name.startswith('_'):
                    db.save_backtest(ticker, horizon, name, metrics)
        table = pd.DataFrame({k:v for k,v in r.items() if not k.startswith('_')}).T
        st.dataframe(table, use_container_width=True)
        st.json(r['_meta'])
