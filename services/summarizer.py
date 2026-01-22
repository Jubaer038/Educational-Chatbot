from __future__ import annotations

from typing import Literal

from config import prompts
from rag import chunker, pdf_loader
from services import llm_client

SummaryMode = Literal["short", "bullets"]


def _limit_text(text: str, max_chars: int = 4000) -> str:
    return text[:max_chars]


def summarize_text(content: str, mode: SummaryMode = "short") -> str:
    client = llm_client.get_client()
    user_prompt = prompts.SUMMARY_USER_TEMPLATE.format(mode=mode, content=_limit_text(content))
    raw = client.generate(prompts.SUMMARY_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    if mode == "bullets":
        return _postprocess_bullets(raw)
    return raw


def _postprocess_bullets(text: str, max_points: int = 8, max_len: int = 120) -> str:
    """Normalize LLM bullet output to one bullet per line and shorten each point.

    - Splits by common bullet markers or newlines
    - Keeps up to `max_points` items
    - Shortens each item to `max_len` characters at a sentence or space boundary
    - Returns a markdown-style list (one item per line starting with '- ')
    """
    if not text:
        return ""

    # Normalize separators to newlines
    sep_candidates = ['\n', '\r', '•', '\u2022', '-', '*']
    normalized = text
    for s in sep_candidates:
        normalized = normalized.replace(s, '\n')

    # Split and clean
    parts = [p.strip() for p in normalized.split('\n') if p.strip()]

    # If the model returned a single long line with bullets separated by '•' or commas,
    # trying a comma split as a fallback when only one part found.
    if len(parts) <= 1 and ',' in text:
        parts = [p.strip() for p in text.split(',') if p.strip()]

    # Keep only top N points
    parts = parts[:max_points]

    def shorten(p: str) -> str:
        if len(p) <= max_len:
            return p
        # try to cut at sentence end
        for sep in ['. ', '? ', '! ']:
            idx = p.find(sep, 80)
            if idx != -1 and idx + 1 < max_len:
                return p[: idx + 1].strip()
        # otherwise cut at last space before max_len
        cut = p[:max_len]
        if ' ' in cut:
            cut = cut.rsplit(' ', 1)[0]
        return cut.strip() + '...'

    shortened = [shorten(p) for p in parts]

    # Return as markdown list so Streamlit renders one item per line
    return "\n".join(f"- {s}" for s in shortened)


def summarize_pdf(file_bytes: bytes, mode: SummaryMode = "short") -> str:
    pages = pdf_loader.extract_text_by_page(file_bytes)
    combined = "\n".join(p.get("text", "") for p in pages)
    chunks = chunker.chunk_pages([{"page": 0, "text": combined}], chunk_size=2000, overlap=200)
    merged = "\n".join(c["text"] for c in chunks[:3])
    return summarize_text(merged, mode=mode)
