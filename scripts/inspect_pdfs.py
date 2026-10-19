"""Check how well PyMuPDF can extract text from the PDFs in a folder (read-only).

Usage: python scripts/inspect_pdfs.py [folder]    (default: <project>/data/docs, from any directory)
"""
import sys
from pathlib import Path

import pymupdf  # PyMuPDF

MIN_CHARS = 50  # a page with fewer characters than this is probably a scan or blank

PROJECT_ROOT = Path(__file__).resolve().parent.parent
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "docs"
pdfs = sorted(folder.glob("*.pdf"))
if not pdfs:
    sys.exit(f"No PDFs found in {folder}")

for pdf in pdfs:
    with pymupdf.open(pdf) as doc:
        texts = [page.get_text() for page in doc]

    lengths = [len(t.strip()) for t in texts]
    empty = [i + 1 for i, n in enumerate(lengths) if n < MIN_CHARS]
    print(f"\n{pdf.name}")
    print(f"  pages: {len(texts)}   total chars: {sum(lengths)}   avg chars/page: {sum(lengths) // len(texts)}")
    if empty:
        print(f"  pages with (almost) no text: {empty}  -> likely scanned, would need OCR")
    else:
        print("  every page has text")

    sample = next((t.strip() for t in texts if len(t.strip()) >= MIN_CHARS), None)
    if sample:
        print("  sample:", " ".join(sample.split())[:300])
