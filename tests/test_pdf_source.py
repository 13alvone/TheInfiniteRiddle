#!/usr/bin/env python3
"""Tests for PDF quote sourcing."""
from pathlib import Path
from random import Random

from riddle.quote_overlay.pdf_source import fetch_random_quote


def _make_pdf(texts: list[str], path: Path) -> None:
    """Create a simple multi-page PDF containing ``texts``."""
    pdf_lines = ["%PDF-1.4\n"]
    offsets = []

    def add(obj: str) -> None:
        offsets.append(sum(len(x) for x in pdf_lines))
        pdf_lines.append(obj)

    add("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    kids = " ".join(f"{i+3} 0 R" for i in range(len(texts)))
    add(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(texts)} >>\nendobj\n")
    for i in range(len(texts)):
        content_num = 7 + i
        add(
            f"{3+i} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 6 0 R >> >> /Contents {content_num} 0 R >>\nendobj\n"
        )
    add("6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    for i, text in enumerate(texts):
        stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET"
        add(
            f"{7+i} 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream}\nendstream\nendobj\n"
        )
    xref_start = sum(len(x) for x in pdf_lines)
    pdf_lines.append(f"xref\n0 {len(offsets)+1}\n0000000000 65535 f \n")
    for off in offsets:
        pdf_lines.append(f"{off:010d} 00000 n \n")
    pdf_lines.append(
        f"trailer\n<< /Size {len(offsets)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
    )
    path.write_bytes("".join(pdf_lines).encode("latin1"))


def test_fetch_random_quote_basic(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_pdf(
        ["First page", "Second page", "Third page with words"], pdf_dir / "sample.pdf"
    )
    rng = Random(0)
    quote = fetch_random_quote(pdf_dir, rng)
    assert quote == "Third page with words"


def test_missing_dir_returns_empty(tmp_path: Path) -> None:
    rng = Random(0)
    quote = fetch_random_quote(tmp_path / "missing", rng)
    assert quote == ""


def test_empty_pdf_returns_empty(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_pdf(["Page1", "Page2", ""], pdf_dir / "empty.pdf")
    rng = Random(0)
    quote = fetch_random_quote(pdf_dir, rng)
    assert quote == ""
