"""Streamlit UI: upload PDFs, ask a question, see the answer and the sources it came from.

Run: streamlit run app.py   (Ollama must be running: ollama serve)
"""
import streamlit as st

from ask import bm25_index, citation, generate, is_refusal, retrieve
from audit import log_question
from ingest import get_collection, ingest_file

st.title("Private Research Assistant")
st.caption("Runs fully on this machine. Your documents never leave it.")

# --- Documents: upload new PDFs ---
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Index uploaded files"):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}..."):
                count = ingest_file(f.name, f.getvalue())
            st.success(f"{f.name}: {count} chunks")
        bm25_index.cache_clear()  # so keyword search sees the new chunks
    st.caption(f"{get_collection().count()} chunks indexed")

# --- Question and answer ---
with st.form("ask"):
    question = st.text_input("Ask a question about your documents")
    asked = st.form_submit_button("Ask")

if asked and question:
    try:
        with st.spinner("Searching the documents..."):
            chunks = retrieve(question)
            answer = generate(question, chunks)
        st.subheader("Answer")
        st.write(answer)
        refused = is_refusal(answer)
        log_question(question, chunks, answer, refused)
        if not refused:  # nothing to cite when the model found no answer
            st.subheader("Sources")
            for text, m in chunks:
                with st.expander(citation(m)):
                    st.write(text)
    except ConnectionError:
        st.error("Can't reach Ollama. Start it with `ollama serve` and try again.")
