from __future__ import annotations

from typing import List, Dict, Any

from config import settings
from rag import embeddings, vector_store


def build_index(chunks: List[Dict[str, Any]]):
    texts = [c["text"] for c in chunks]
    embeds = embeddings.embed_texts(texts)
    index = vector_store.create_index(embeds)
    return index, embeds


def retrieve(query: str, index, chunks: List[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.TOP_K
    query_vec = embeddings.embed_texts([query])
    scores, ids = vector_store.search(index, query_vec, top_k=top_k)
    return vector_store.pack_results(ids, scores, chunks)
