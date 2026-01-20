from __future__ import annotations

from typing import Dict, Any, List

import numpy as np

from config import prompts
from rag import embeddings


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) or 1e-8
    return float(np.dot(vec_a, vec_b) / denom)


def _length_ratio(student: str, reference: str) -> float:
    if not student or not reference:
        return 0.0
    ratio = len(student.split()) / max(len(reference.split()), 1)
    return float(min(ratio, 1.5))  # cap to avoid overweight


def _clarity_score(text: str) -> float:
    if not text:
        return 0.0
    sentences = [s for s in text.replace("\n", " ").split(".") if s.strip()]
    avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if avg_len <= 12:
        return 1.0
    if avg_len <= 22:
        return 0.8
    if avg_len <= 32:
        return 0.6
    return 0.4


def _language_score(text: str) -> float:
    if not text:
        return 0.0
    has_punctuation = any(p in text for p in [".", ",", ";", ":", "?"])
    capped_len = min(len(text), 200)
    capital_ratio = sum(1 for c in text[:capped_len] if c.isupper()) / max(capped_len, 1)
    penalty = 0.2 if capital_ratio > 0.3 else 0.0
    return max(0.5 if has_punctuation else 0.3, 1.0 - penalty)


def grade_response(question: str, student_answer: str, reference_answer: str) -> Dict[str, Any]:
    weights = prompts.RUBRIC_WEIGHTS
    if not reference_answer:
        return {
            "score": 0,
            "details": {},
            "strengths": [],
            "improvements": ["Reference answer missing; cannot grade."],
        }

    embed_student = embeddings.embed_texts([student_answer])[0]
    embed_reference = embeddings.embed_texts([reference_answer])[0]

    concept = _cosine_similarity(embed_student, embed_reference)
    coverage = _length_ratio(student_answer, reference_answer)
    clarity = _clarity_score(student_answer)
    language = _language_score(student_answer)

    details = {
        "concept_correctness": round(concept * 100, 2),
        "coverage": round(coverage * 100, 2),
        "clarity": round(clarity * 100, 2),
        "language": round(language * 100, 2),
    }

    weighted = (
        concept * weights["concept_correctness"]
        + coverage * weights["coverage"]
        + clarity * weights["clarity"]
        + language * weights["language"]
    )
    score = round(weighted * 100, 1)

    strengths: List[str] = []
    improvements: List[str] = []
    if concept > 0.6:
        strengths.append("Concepts align with reference.")
    else:
        improvements.append("Improve conceptual alignment with the source material.")
    if coverage > 0.7:
        strengths.append("Good coverage of the expected points.")
    else:
        improvements.append("Add more key points from the reference answer.")
    if clarity > 0.7:
        strengths.append("Sentences are concise and readable.")
    else:
        improvements.append("Write shorter, clearer sentences.")
    if language > 0.6:
        strengths.append("Language and punctuation are adequate.")
    else:
        improvements.append("Check punctuation and casing for readability.")

    return {"score": score, "details": details, "strengths": strengths, "improvements": improvements}
