from __future__ import annotations

from typing import List, Dict, Any

from config import settings


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int | None = None, overlap: int | None = None) -> List[Dict[str, Any]]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    # Validate settings
    try:
        chunk_size = int(chunk_size)
    except Exception:
        raise ValueError("chunk_size must be an integer > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    try:
        overlap = int(overlap)
    except Exception:
        overlap = 0
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # ensure progress: keep some overlap but less than chunk_size
        overlap = max(0, chunk_size // 4)

    chunks: List[Dict[str, Any]] = []

    if not isinstance(pages, list):
        raise TypeError("pages must be a list of {'page': int, 'text': str} dicts")

    for entry in pages:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", "") or ""
        # Coerce non-str values to str safely
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                text = ""
        page = entry.get("page")

        start = 0
        last_start = -1
        text_len = len(text)
        if text_len == 0:
            continue

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]
            chunks.append({"page": page, "text": chunk_text})
            last_start = start
            start = end - overlap
            if start < 0:
                start = 0
            # Prevent infinite loops: if we didn't make progress, break
            if start <= last_start:
                break
            if start >= text_len:
                break

    return chunks
