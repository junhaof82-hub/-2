import streamlit as st
import pandas as pd
from storage.database import Database


def render():
    st.title('Model Accuracy')
    db = Database()
    acc = db.latest_accuracy()
    pred = db.prediction_history()
    if acc.empty:
        st.info('No model accuracy records yet. Run AI Prediction first.')
    else:
        latest = acc.sort_values('id').groupby(['ticker','horizon','model_name'], as_index=False).tail(1)
        st.dataframe(latest[['ticker','horizon','model_name','accuracy','precision','recall','roc_auc','sample_size','as_of']], use_container_width=True)
    if not pred.empty and pred['actual_up'].notna().any():
        evald = pred.dropna(subset=['actual_up']).copy()
        evald['correct'] = ((evald['probability'] >= .5).astype(int) == evald['actual_up'].astype(int)).astype(int)
        horizon_acc = evald.groupby('horizon')['correct'].mean().mul(100).reset_index(name='Accuracy %')
        st.subheader('Ensemble live prediction accuracy')
        st.dataframe(horizon_acc, use_container_width=True)
