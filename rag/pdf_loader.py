from __future__ import annotations

import io
from typing import List, Dict, Any

from PyPDF2 import PdfReader


def extract_text_by_page(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Return list of dicts with page number (1-indexed) and text."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to read PDF: {exc}")

    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        cleaned = text.replace("\u0000", " ").strip()
        pages.append({"page": idx, "text": cleaned})
    if not pages:
        raise ValueError("No readable pages found in the PDF.")
    return pages
