import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(layout="wide")
st.title("Advanced RAG + Multimodal + SQL QA")

tab1, tab2, tab3 = st.tabs(["Text QA", "Image QA", "SQL QA"])

with tab1:
    q = st.text_input("Ask a question")
    if st.button("Ask"):
        r = requests.post(f"{API_URL}/ask", json={"question": q}).json()
        st.write(r["answer"])
        st.metric("Confidence", r["confidence"])
        st.metric("Hallucination", r["hallucination_score"])
with tab2:
    iq = st.text_input("Ask a question(image)")
    img = st.file_uploader("Upload an image (optional)", type=["png","jpg","jpeg"])
    mode = st.selectbox("Retrieval mode", ["image", "text", "image_to_text"])

    if st.button("Ask Image"):
        if img:
            files = {"file": img.getvalue()}
        else:
            files = None

        payload = {
            "question": iq,
            "mode": mode
        }

        r = requests.post(f"{API_URL}/ask-image", data=payload, files={"file": img} if img else None)
        r = r.json()

        st.write(r["answer"])
        st.metric("Confidence", r["confidence"])
        st.json(r["matches"])
with tab3:
    sq = st.text_input("Ask SQL question")
    if st.button("Run SQL QA"):
        r = requests.post(f"{API_URL}/ask-sql", json={"question": sq}).json()
        st.write(r["answer"])
        st.metric("Confidence", r["confidence"])