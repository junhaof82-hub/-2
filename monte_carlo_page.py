import streamlit as st
from prediction.predictor import StockPredictor


def render():
    st.title('Target Touch Probability')
    ticker = st.text_input('Ticker', 'SNDK', key='mc_ticker').upper().strip()
    target = st.number_input('Target price', min_value=0.01, value=170.0, step=1.0)
    if st.button('Run Monte Carlo', type='primary'):
        r = StockPredictor().target_touch(ticker, target)
        st.write(f"Last price: ${r['last_price']:.2f}")
        for h in (1,3,5):
            st.metric(f'Touch within {h} day(s)', f"{r['probabilities'][h]*100:.1f}%")
