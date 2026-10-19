"""Measure retrieval quality (recall@5) and refusal behaviour.

Usage: python evaluate.py [dense|hybrid|rerank] [--set eval_set_hard.json]
       saves results/<question set>_<mode>.json, e.g. results/eval_set_rerank.json
       dense  = vector search only
       hybrid = vector search + BM25, merged with RRF
       rerank = hybrid, then re-ordered by the reranker (default)
       --set  = question file to use (default: eval_set.json)
"""
import argparse
import json
import time
from collections import defaultdict

from ask import (TOP_K, bm25_index, bm25_search, dense_search, embed_query, fuse,
                 generate, is_refusal, rerank_chunks, reranker, retrieve)
from ingest import ROOT

times = defaultdict(list)  # stage name -> seconds for each call


def timed(stage, fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    times[stage].append(time.perf_counter() - start)
    return result


def timed_retrieve(question, mode):
    """Same steps as ask.retrieve(), with each stage timed. Returns [(id, text, metadata)]."""
    found = timed("dense", dense_search, timed("embed", embed_query, question))
    if mode != "dense":
        found = timed("fuse", fuse, found, timed("bm25", bm25_search, question))
    if mode == "rerank":
        found = timed("rerank", rerank_chunks, question, found)
    return found[:TOP_K]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["dense", "hybrid", "rerank"], default="rerank")
    parser.add_argument("--set", default="eval_set.json", help="question file, relative to the project folder")
    args = parser.parse_args()
    mode = args.mode
    question_file = ROOT / args.set

    questions = json.loads(question_file.read_text())
    answerable = [q for q in questions if q.get("answerable", True)]
    unanswerable = [q for q in questions if not q.get("answerable", True)]

    embed_query("warm up")  # load the models and build the BM25 index so the first question isn't slower
    bm25_index()
    if mode == "rerank":
        reranker()

    # 1. Retrieval: is the labelled file + page among the top-5 chunks? (no LLM)
    hits, misses, per_file = 0, [], defaultdict(list)
    for q in answerable:
        chunks = timed_retrieve(q["question"], mode)
        valid_pages = {a["page"] for a in q["answers"]}  # any of these pages answers the question
        hit = any(m["source"].startswith(q["source"]) and m["page"] in valid_pages for _, _, m in chunks)
        per_file[q["source"]].append(hit)
        if hit:
            hits += 1
        else:
            misses.append(q)

    # 2. Refusal: with no answer in the documents, does the model say so? (needs the LLM)
    refused = 0
    for q in unanswerable:
        chunks = retrieve(q["question"], hybrid=mode != "dense", rerank=mode == "rerank")
        answer = timed("generate", generate, q["question"], chunks)
        if is_refusal(answer):
            refused += 1
        else:
            print(f"  did not refuse: {q['question']}\n    -> {answer[:150]!r}")

    results = {
        "mode": mode,
        "recall_at_5": hits / len(answerable),
        "hits": hits,
        "answerable": len(answerable),
        "refused": refused,
        "unanswerable": len(unanswerable),
        "avg_seconds": {stage: sum(t) / len(t) for stage, t in times.items()},
        "missed": [{"question": q["question"], "source": q["source"], "pages": [a["page"] for a in q["answers"]]} for q in misses],
    }

    print(f"\nQuestions: {question_file.name}   Mode: {mode}")
    print(f"Recall@{TOP_K}: {hits}/{len(answerable)} = {results['recall_at_5']:.0%}")
    for source, flags in per_file.items():
        print(f"  {source:<22} {sum(flags)}/{len(flags)}")
    print("\nMissed:")
    for q in misses:
        print(f"  [{q['source']} p.{[a['page'] for a in q['answers']]}] {q['question']}")
    print(f"\nRefusals: {refused}/{len(unanswerable)}")
    print("\nAvg latency per stage:")
    for stage, seconds in results["avg_seconds"].items():
        print(f"  {stage:<9} {seconds * 1000:.1f} ms")

    out = ROOT / "results" / f"{question_file.stem}_{mode}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
