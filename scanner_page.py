import streamlit as st
from config import settings
from prediction.ranking import scan


def render():
    st.title('Stock Scanner')
    raw = st.text_area('Tickers (comma separated)', ','.join(settings.default_tickers))
    horizon = st.selectbox('Horizon', [1,3,5], index=2, key='scan_h')
    if st.button('Run scanner', type='primary'):
        tickers = [x.strip().upper() for x in raw.split(',') if x.strip()]
        with st.spinner('Scanning stocks...'):
            df = scan(tickers, horizon=horizon)
        st.dataframe(df.style.format({'Up Probability':'{:.1f}%','Expected Return %':'{:.2f}%','Risk':'{:.1f}','Confidence':'{:.1f}%'}), use_container_width=True)
