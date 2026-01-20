from __future__ import annotations

from typing import List, Dict, Any, Tuple

import faiss
import numpy as np


def create_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index: faiss.IndexFlatIP, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if query_vector.ndim == 1:
        query_vector = np.expand_dims(query_vector, axis=0)
    scores, ids = index.search(query_vector.astype("float32"), top_k)
    return scores, ids


def pack_results(ids: np.ndarray, scores: np.ndarray, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        meta = chunks[int(idx)]
        results.append({"page": meta.get("page"), "text": meta.get("text"), "score": float(score)})
    return results
