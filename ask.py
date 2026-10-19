"""Ask a question about the ingested documents.

Usage: python ask.py "Can an ACE inhibitor be combined with an ARB?"
"""
import os
import re
import sys
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # load the reranker from the local cache, never the internet

import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from audit import log_question
from ingest import EMBED_MODEL, get_collection

LLM_MODEL = "llama3.1:8b"
RERANK_MODEL = "BAAI/bge-reranker-base"
TOP_K = 5  # chunks sent to the model
CANDIDATES = 20  # chunks each search method proposes before they are merged
RRF_K = 60  # standard constant for Reciprocal Rank Fusion
# temperature 0: factual answers, no creativity. num_ctx: 5 chunks need more than a small default window.
CHAT_OPTIONS = {"temperature": 0, "num_ctx": 8192}

REFUSAL = "I can't find this in the documents."

PROMPT = """Answer ONLY using the context below. End every claim with the number of the
source it came from, like (Source 2). There are {count} sources, numbered 1 to {count};
other numbers in the text, such as recommendation 1.9.8, are not source numbers.
If the answer is not in the context, say "{refusal}"

Context:
{context}

Question: {question}"""


def embed_query(question):
    # "search_query: " is the prefix nomic-embed-text expects for questions
    return ollama.embed(model=EMBED_MODEL, input=[f"search_query: {question}"])["embeddings"][0]


def dense_search(vector):
    """Meaning-based search: the chunks closest to the question's vector. Returns [(id, text, metadata)]."""
    result = get_collection().query(query_embeddings=[vector], n_results=CANDIDATES)
    return list(zip(result["ids"][0], result["documents"][0], result["metadatas"][0]))


def tokenize(text):
    return re.findall(r"\w+", text.lower())


@lru_cache
def bm25_index():
    """Keyword index over every stored chunk. Built once; call bm25_index.cache_clear() after ingesting."""
    data = get_collection().get()
    chunks = list(zip(data["ids"], data["documents"], data["metadatas"]))
    return BM25Okapi([tokenize(text) for _, text, _ in chunks]), chunks


def bm25_search(question):
    """Keyword search: the chunks sharing the most (rare) words with the question. Returns [(id, text, metadata)]."""
    bm25, chunks = bm25_index()
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
    return [chunks[i] for i in best if scores[i] > 0]  # drop chunks with no word in common


def fuse(*rankings):
    """Reciprocal Rank Fusion: a chunk ranked high in any list scores high. Returns all chunks, best first."""
    scores, chunks = {}, {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk[0]] = scores.get(chunk[0], 0) + 1 / (RRF_K + rank + 1)
            chunks[chunk[0]] = chunk
    return [chunks[chunk_id] for chunk_id in sorted(scores, key=scores.get, reverse=True)]


@lru_cache
def reranker():
    return CrossEncoder(RERANK_MODEL)


def rerank_chunks(question, chunks):
    """Re-order chunks by reading the question and each chunk together. Returns the chunks, best first."""
    scores = reranker().predict([(question, text) for _, text, _ in chunks])
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in best]


def retrieve(question, hybrid=True, rerank=True):
    """Find the best TOP_K chunks for the question. Returns [(text, metadata)]."""
    found = dense_search(embed_query(question))
    if hybrid:
        found = fuse(found, bm25_search(question))
    if rerank:
        found = rerank_chunks(question, found)
    return [(text, metadata) for _, text, metadata in found[:TOP_K]]


def citation(metadata):
    """File name and page, plus the document title when the PDF has one."""
    label = f"{metadata['source']}, page {metadata['page']}"
    return f"{label}, \"{metadata['title']}\"" if metadata.get("title") else label


def is_refusal(answer):
    """True when the model said the answer isn't in the documents."""
    return "can't find this in the documents" in answer.replace("\u2019", "'").lower()


def finalize(answer, chunks):
    """Either an answer with full citations, or the refusal. Nothing in between.
    The model's 'Source 2' becomes the full citation (file, page, title) of the 2nd chunk. If the answer
    cites no source, or cites one that doesn't exist, we can't verify it, so it counts as not found."""
    numbers = [int(n) for n in re.findall(r"Source (\d+)", answer)]
    if is_refusal(answer) or not numbers or not all(1 <= n <= len(chunks) for n in numbers):
        return REFUSAL
    return re.sub(r"Source (\d+)", lambda m: citation(chunks[int(m.group(1)) - 1][1]), answer)


def build_prompt(question, chunks):
    """The full prompt: instructions, the numbered sources, and the question."""
    context = "\n\n".join(f"Source {i}: {citation(m)}\n{text}" for i, (text, m) in enumerate(chunks, 1))
    return PROMPT.format(count=len(chunks), refusal=REFUSAL, context=context, question=question)


def generate(question, chunks):
    """Ask the local model to answer from the chunks. Returns the answer with full citations, or the refusal."""
    reply = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        options=CHAT_OPTIONS,
    )
    return finalize(reply["message"]["content"], chunks)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or sys.exit(__doc__)
    chunks = retrieve(question)
    answer = generate(question, chunks)
    print(answer)
    refused = is_refusal(answer)
    log_question(question, chunks, answer, refused)
    if not refused:
        print("\n\nSources:")
        for _, m in chunks:
            print(f"  {citation(m)}")
