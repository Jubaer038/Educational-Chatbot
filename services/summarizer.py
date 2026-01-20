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
    return client.generate(prompts.SUMMARY_SYSTEM_PROMPT, user_prompt, max_tokens=300)


def summarize_pdf(file_bytes: bytes, mode: SummaryMode = "short") -> str:
    pages = pdf_loader.extract_text_by_page(file_bytes)
    combined = "\n".join(p.get("text", "") for p in pages)
    chunks = chunker.chunk_pages([{"page": 0, "text": combined}], chunk_size=2000, overlap=200)
    merged = "\n".join(c["text"] for c in chunks[:3])
    return summarize_text(merged, mode=mode)
