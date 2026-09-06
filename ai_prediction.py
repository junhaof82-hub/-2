import streamlit as st
from prediction.predictor import StockPredictor


def render():
    st.title('AI Prediction')
    ticker = st.text_input('Ticker', 'AVGO', key='pred_ticker').upper().strip()
    force = st.checkbox('Retrain models now')
    if st.button('Run AI prediction', type='primary'):
        with st.spinner('Training/loading models and analyzing data...'):
            r = StockPredictor().analyze(ticker, period='5y' if force else '2y', force_train=force)
        st.subheader(f"{r['ticker']} — Last ${r['last_price']:.2f}")
        cols = st.columns(5)
        for i, h in enumerate((1,3,5)):
            cols[i].metric(f'{h}-Day Up', f"{r['probabilities'][h]*100:.1f}%")
        cols[3].metric('Bullish Score', f"{r['bullish_score']:.1f}/10")
        cols[4].metric('Risk Score', f"{r['risk_score']:.1f}/10")
        st.write('**Expected ranges**')
        for h in (1,3,5):
            lo, hi = r['expected_ranges'][h]
            st.write(f'{h}D: ${lo:.2f} – ${hi:.2f}')
        st.write('**Catalysts:**', ', '.join(r['catalysts']) or 'None detected')
        st.write('**Risks:**', ', '.join(r['risks']) or 'None detected')
        with st.expander('Model probabilities'):
            st.json(r['model_probabilities'])
        with st.expander('Fundamentals'):
            st.json(r['fundamentals'])
        with st.expander('Macro'):
            st.json(r['macro'])
