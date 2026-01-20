from __future__ import annotations

import hashlib
from typing import Dict, Any, List, Tuple

import numpy as np
import streamlit as st

from config import prompts, settings
from rag import chunker, pdf_loader, retriever
from services import llm_client


@st.cache_resource(show_spinner=False)
def _cached_index(file_hash: str, file_bytes: bytes) -> Tuple[Any, List[Dict[str, Any]]]:
    pages = pdf_loader.extract_text_by_page(file_bytes)
    chunks = chunker.chunk_pages(pages)
    index, _ = retriever.build_index(chunks)
    return index, chunks


def _hash_pdf(file_bytes: bytes, file_name: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    hasher.update(file_name.encode())
    return hasher.hexdigest()


def get_index(file_bytes: bytes, file_name: str) -> Tuple[Any, List[Dict[str, Any]]]:
    file_hash = _hash_pdf(file_bytes, file_name)
    return _cached_index(file_hash, file_bytes)


def build_context_snippet(results: List[Dict[str, Any]]) -> str:
    snippets = []
    for res in results:
        page = res.get("page")
        text = res.get("text", "").strip().replace("\n", " ")
        snippets.append(f"[Page {page}] {text}")
    return "\n".join(snippets)


def answer_question(question: str, file_bytes: bytes, file_name: str) -> Dict[str, Any]:
    index, chunks = get_index(file_bytes, file_name)
    matches = retriever.retrieve(question, index, chunks, top_k=settings.TOP_K)
    if not matches:
        return {"answer": "No content found in the PDF to answer this question.", "citations": []}

    context = build_context_snippet(matches)
    client = llm_client.get_client()
    user_prompt = prompts.QA_USER_TEMPLATE.format(context=context, question=question)
    answer = client.generate(prompts.QA_SYSTEM_PROMPT, user_prompt, max_tokens=400)
    citations = [
        {
            "page": m.get("page"),
            "snippet": m.get("text", "")[:220] + ("..." if len(m.get("text", "")) > 220 else ""),
            "score": round(m.get("score", 0), 3),
        }
        for m in matches
    ]
    return {"answer": answer, "citations": citations}


def reference_answer(question: str, file_bytes: bytes, file_name: str) -> str:
    index, chunks = get_index(file_bytes, file_name)
    matches = retriever.retrieve(question, index, chunks, top_k=settings.TOP_K)
    if not matches:
        return ""
    context = build_context_snippet(matches)
    client = llm_client.get_client()
    user_prompt = prompts.REFERENCE_ANSWER_TEMPLATE.format(context=context, question=question)
    return client.generate(prompts.REFERENCE_ANSWER_SYSTEM, user_prompt, max_tokens=200)
