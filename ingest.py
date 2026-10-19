"""Ingestion: read the PDFs page by page, split into fixed-size overlapping chunks,
embed each chunk with Ollama and store it in a local Chroma database."""
import re
import time
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import chromadb
import ollama
import pymupdf

pymupdf.no_recommend_layout()  # silence PyMuPDF's "install pymupdf_layout" hint

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "docs"
CHROMA_DIR = ROOT / "data" / "chroma"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 32      # chunks embedded per Ollama call
MAX_CHARS = 2000     # ~500 tokens (1 token is about 4 characters)
OVERLAP_CHARS = 200  # ~50 tokens
MIN_CHARS = 100      # drop tiny leftovers (e.g. a page that only had a footer)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict  # {"source": file name, "page": page number, "title": document title (only if the PDF has one)}


def read_lines(data):
    """Return [(page_number, line)] for a PDF (given as bytes), with the page footer removed."""
    result = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, start=1):
            lines = [line.strip() for line in page.get_text().splitlines()]
            # Footer: "<guideline title> (NG136)", "© NICE 2026 ...", "Page 20 of", "52"
            for i, line in enumerate(lines):
                if line.startswith("© NICE"):
                    title_lines = 0
                    if i > 0 and re.search(r"\(NG\d+\)$", lines[i - 1]):
                        title_lines = 2 if lines[i - 1].startswith("(") else 1  # long titles wrap onto 2 lines
                    lines = lines[: i - title_lines]
                    break
            result += [(number, line) for line in lines if line]
    return result


def read_title(data):
    """The document title stored in the PDF's metadata, or '' if it has none."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return (doc.metadata.get("title") or "").strip()


def join_lines(lines):
    """Undo the PDF's hard line wrapping, but keep each bullet on its own line."""
    text = ""
    for line in lines:
        text += ("\n" if line.startswith("•") else " ") + line
    return text.strip()


def split_by_size(text):
    """Cut text into ~MAX_CHARS pieces at a line break or sentence end, with OVERLAP_CHARS overlap."""
    pieces, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + MAX_CHARS // 2:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return pieces


def chunk_pdf(name, data):
    """Split each page separately, so every chunk has an exact page number."""
    chunks = []
    title = read_title(data)
    for page, group in groupby(read_lines(data), key=lambda t: t[0]):
        text = join_lines([line for _, line in group])
        for piece in split_by_size(text):
            if len(piece) >= MIN_CHARS:
                metadata = {"source": name, "page": page}
                if title:
                    metadata["title"] = title
                chunks.append(Chunk(id=f"{Path(name).stem}-{len(chunks):04d}", text=piece, metadata=metadata))
    return chunks


def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})


def embed_documents(texts):
    # nomic-embed-text is trained with task prefixes: "search_document: " for stored text
    # and "search_query: " for questions. Using them improves retrieval quality.
    response = ollama.embed(model=EMBED_MODEL, input=[f"search_document: {t}" for t in texts])
    return response["embeddings"]


def ingest_file(name, data):
    """Chunk, embed and store one PDF (given as bytes). Returns the number of chunks."""
    collection = get_collection()
    chunks = chunk_pdf(name, data)
    collection.delete(where={"source": name})  # re-ingesting a file replaces its old chunks, no duplicates
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embed_documents([c.text for c in batch]),
        )
    return len(chunks)


def ingest():
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        start = time.perf_counter()
        count = ingest_file(pdf.name, pdf.read_bytes())
        print(f"{pdf.name[:50]:<52} {count:>4} chunks  {time.perf_counter() - start:5.1f}s")
    print(f"\nChroma now holds {get_collection().count()} chunks ({CHROMA_DIR})")


if __name__ == "__main__":
    ingest()