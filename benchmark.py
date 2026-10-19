"""Compare model sizes (Q4 vs Q8 vs FP16) on speed, memory and accuracy.

Usage: python benchmark.py [model ...]     (default: the three models below)
       saves results/benchmark_<model>.json, one file per model

Every model gets the same retrieved chunks, so only the model itself differs.
"""
import json
import re
import sys

import ollama

from ask import CHAT_OPTIONS, build_prompt, is_refusal, retrieve
from ingest import ROOT

MODELS = ["llama3.1:8b", "llama3.1:8b-instruct-q8_0", "llama3.1:8b-instruct-fp16"]  # Q4, Q8, FP16
QUESTION_FILES = ["eval_set.json", "eval_set_hard.json"]


def is_correct(answer, key_fact):
    """True if the answer contains any of the key phrases as whole words."""
    return any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", answer, re.IGNORECASE) for k in key_fact)


def benchmark(model, questions):
    def chat(prompt):
        return ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], options=CHAT_OPTIONS)

    chat("Say hello.")  # warm-up: loads the model, so loading time isn't counted

    answers, tokens, seconds, prompt_tokens, prompt_seconds = [], 0, 0, 0, 0
    for q in questions:
        reply = chat(q["prompt"])
        tokens += reply["eval_count"]
        seconds += reply["eval_duration"] / 1e9
        prompt_tokens += reply["prompt_eval_count"]
        prompt_seconds += reply["prompt_eval_duration"] / 1e9
        answers.append(reply["message"]["content"])
        print(".", end="", flush=True)

    loaded = next(m for m in ollama.ps().models if m.model == model)  # memory while the model is in use
    ollama.generate(model=model, keep_alive=0)  # unload, so the next model starts with free memory

    scored = [(q, a) for q, a in zip(questions, answers) if q.get("answerable", True)]
    unanswerable = [(q, a) for q, a in zip(questions, answers) if not q.get("answerable", True)]
    return {
        "model": model,
        "correct": sum(is_correct(a, q["key_fact"]) for q, a in scored),
        "answerable": len(scored),
        "refused": sum(is_refusal(a) for _, a in unanswerable),
        "unanswerable": len(unanswerable),
        "decode_tokens_per_second": tokens / seconds,
        "prefill_tokens_per_second": prompt_tokens / prompt_seconds,
        "memory_gb": loaded.size / 1e9,
        "answers": [{"question": q["question"], "answer": a} for q, a in zip(questions, answers)],
    }


def main():
    questions = []
    for file in QUESTION_FILES:
        for q in json.loads((ROOT / file).read_text()):
            q["prompt"] = build_prompt(q["question"], retrieve(q["question"]))  # retrieved once, shared by all models
            questions.append(q)

    for model in sys.argv[1:] or MODELS:
        print(f"\n{model} ", end="")
        r = benchmark(model, questions)
        print(f"\n  decode {r['decode_tokens_per_second']:.1f} tok/s | prefill {r['prefill_tokens_per_second']:.0f} tok/s"
              f" | memory {r['memory_gb']:.1f} GB")
        print(f"  accuracy {r['correct']}/{r['answerable']} = {r['correct'] / r['answerable']:.0%}"
              f" | refusals {r['refused']}/{r['unanswerable']}")

        out = ROOT / "results" / f"benchmark_{model.replace(':', '_')}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(r, indent=2))
        print(f"  saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
