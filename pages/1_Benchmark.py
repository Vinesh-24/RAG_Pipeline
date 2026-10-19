"""Benchmark page: how the Q4, Q8 and FP16 versions of the model compare.

Reads the files written by benchmark.py. Opens from the sidebar of `streamlit run app.py`.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).parent.parent / "results"
MODELS = {"Q4": "llama3.1:8b", "Q8": "llama3.1:8b-instruct-q8_0", "FP16": "llama3.1:8b-instruct-fp16"}

st.title("Model benchmark")
st.caption("Same questions, same retrieved chunks, temperature 0. Only the model's precision changes.")

results = {}
for label, model in MODELS.items():
    path = RESULTS / f"benchmark_{model.replace(':', '_')}.json"
    if path.exists():
        results[label] = json.loads(path.read_text())
if not results:
    st.info("No results yet. Run `python benchmark.py` first.")
    st.stop()

table = pd.DataFrame(
    [{
        "Decode speed (tokens/s)": round(r["decode_tokens_per_second"], 1),
        "Prefill speed (tokens/s)": round(r["prefill_tokens_per_second"]),
        "Memory (GB)": round(r["memory_gb"], 1),
        "Correct answers": r["correct"],
        "Refusals when no answer exists": r["refused"],
    } for r in results.values()],
    index=list(results),
)
first = next(iter(results.values()))
st.dataframe(table, width="stretch")
st.caption(f"Accuracy is out of {first['answerable']} answerable questions; refusals out of {first['unanswerable']} unanswerable ones.")

for column, (title, name) in zip(st.columns(3), [
    ("Decode speed (tokens/s, higher is better)", "Decode speed (tokens/s)"),
    ("Memory (GB, lower is better)", "Memory (GB)"),
    ("Correct answers (higher is better)", "Correct answers"),
]):
    with column:
        st.subheader(title)
        st.bar_chart(table[name], sort=False)  # sort=False keeps Q4, Q8, FP16 in order

st.header("Compare the answers")
questions = [a["question"] for a in first["answers"]]
question = st.selectbox("Question", questions)
index = questions.index(question)
for column, (label, r) in zip(st.columns(len(results)), results.items()):
    with column:
        st.subheader(label)
        st.write(r["answers"][index]["answer"])
