import streamlit as st
from storage.database import Database


def render():
    st.title('US Stock AI Dashboard')
    st.caption('Technical + Fundamental + News + Macro + ML + Monte Carlo')
    db = Database()
    hist = db.prediction_history()
    c1, c2, c3 = st.columns(3)
    c1.metric('Saved Predictions', len(hist))
    c2.metric('Tracked Tickers', int(hist['ticker'].nunique()) if not hist.empty else 0)
    c3.metric('Evaluated Predictions', int(hist['actual_up'].notna().sum()) if not hist.empty else 0)
    st.info('Open **Stock Analysis** to fetch data and produce a live prediction. API keys are optional; yfinance is the default fallback.')
    if not hist.empty:
        st.subheader('Recent predictions')
        st.dataframe(hist[['ticker','prediction_date','horizon','probability','bullish_score','risk_score']].head(30), use_container_width=True)
