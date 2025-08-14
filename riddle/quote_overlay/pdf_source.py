#!/usr/bin/env python3
"""PDF-based quote sourcing for The Infinite Riddle."""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import List

try:  # Prefer PyPDF2 if available
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional dependency missing
    PdfReader = None  # type: ignore

try:  # Fallback to pdfminer.six
    from pdfminer.high_level import extract_text
except Exception:  # pragma: no cover - optional dependency missing
    extract_text = None  # type: ignore


def _extract_page_text(path: Path, page_index: int) -> str:
    """Extract text from the given page of ``path``.

    This tries PyPDF2 first and falls back to pdfminer.six.  Returns an empty
    string on any failure.
    """
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            if page_index < len(reader.pages):
                text = reader.pages[page_index].extract_text() or ""
                return text
        except Exception:
            return ""
    if extract_text is not None:
        try:
            text = extract_text(str(path), page_numbers=[page_index])
            return text or ""
        except Exception:
            return ""
    return ""


def fetch_random_quote(pdf_dir: Path, rng: Random) -> str:
    """Return a short random quote from PDFs under ``pdf_dir``.

    The function chooses a random PDF, selects a random page index >= 2, extracts
    text and returns a snippet comprising between 3 and 50 words.  An empty
    string is returned if no suitable text is found or on error.
    """
    if not pdf_dir.is_dir():
        return ""

    pdfs: List[Path] = [p for p in pdf_dir.glob("*.pdf") if p.is_file()]
    if not pdfs:
        return ""

    pdf_path = rng.choice(pdfs)
    try:
        if PdfReader is not None:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
        else:
            # pdfminer requires reading once to know page count; fall back to 0
            total_pages = 0
    except Exception:
        return ""

    if total_pages <= 2:
        return ""

    page_index = rng.randrange(2, total_pages)
    text = _extract_page_text(pdf_path, page_index)
    words = [w for w in text.split() if w]
    if not words:
        return ""

    n = rng.randint(3, 50)
    m = min(n, len(words))
    return " ".join(words[:m])
