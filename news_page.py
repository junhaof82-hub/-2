import streamlit as st
from data.news import NewsService
from features.sentiment import NewsSentimentAnalyzer


def render():
    st.title('News')
    ticker = st.text_input('Ticker', 'NVDA', key='news_ticker').upper().strip()
    if st.button('Analyze news', type='primary'):
        items = NewsService().get(ticker, 20)
        r = NewsSentimentAnalyzer().analyze(items)
        st.metric('Aggregate sentiment', f"{r['sentiment_score']:.1f}/100", r['label'])
        for x in r['items']:
            with st.expander(f"[{x.get('label','neutral').upper()}] {x.get('title','')}"):
                st.write(x.get('summary',''))
                st.write(f"Score: {x.get('sentiment_score',0):.1f} | Importance: {x.get('importance',0):.1f}/10 | Method: {x.get('method','')}")
                if x.get('url'): st.link_button('Open article', x['url'])
