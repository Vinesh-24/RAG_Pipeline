"""Audit log: one line per question asked, kept for 24 hours.

Each line records when it was asked, the question, which documents were used and a short
excerpt of the answer (never the full text). Lines older than KEEP are deleted on every write.
"""
import json
from datetime import datetime, timedelta, timezone

from ingest import ROOT

LOG_FILE = ROOT / "logs" / "audit.jsonl"
KEEP = timedelta(hours=24)
EXCERPT_CHARS = 60  # how much of the start and of the end of the answer to keep


def excerpt(answer):
    """The start and end of the answer, joined by '...'. Short answers are kept whole."""
    answer = " ".join(answer.split())
    if len(answer) <= 2 * EXCERPT_CHARS:
        return answer
    return f"{answer[:EXCERPT_CHARS]} ... {answer[-EXCERPT_CHARS:]}"


def log_question(question, chunks, answer, refused):
    """Append one entry, and drop entries older than KEEP. chunks is [(text, metadata)]."""
    now = datetime.now(timezone.utc)
    entry = {
        "time": now.isoformat(timespec="seconds"),
        "question": question,
        "sources": [f"{m['source']}, page {m['page']}" for _, m in chunks],
        "outcome": "refused" if refused else "answered",
        "answer_excerpt": excerpt(answer),
    }
    old_lines = LOG_FILE.read_text().splitlines() if LOG_FILE.exists() else []
    recent = [line for line in old_lines if datetime.fromisoformat(json.loads(line)["time"]) > now - KEEP]
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text("\n".join(recent + [json.dumps(entry)]) + "\n")
