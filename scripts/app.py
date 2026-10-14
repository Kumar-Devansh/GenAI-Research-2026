"""Streamlit dashboard.  Run: streamlit run scripts/app.py"""
import streamlit as st
from core import predict_next_day, search_knowledge

st.set_page_config(page_title="Delhi Monsoon AI", page_icon="🌦️")
st.title("🌦️ Delhi Weather Intelligence")
t1, t2, t3 = st.tabs(["Forecast", "Knowledge Search", "Ask the AI"])

with t1:
    if st.button("Predict next-day temperature"):
        r = predict_next_day()
        st.metric(f"Mean temp on {r['forecast_date']}", f"{r['predicted_mean_temp_c']} °C")
        st.caption(f"Model: multivariate {r['model']}")
with t2:
    q = st.text_input("Search monsoon knowledge base")
    if q: st.text(search_knowledge(q))
with t3:
    q2 = st.text_input("Ask anything about Delhi weather / monsoon")
    if q2:
        from agent import run_agent
        with st.spinner("Thinking..."):
            st.write(run_agent(q2))
