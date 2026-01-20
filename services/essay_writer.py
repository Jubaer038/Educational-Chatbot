from __future__ import annotations

from services import llm_client
from config import prompts


def write_essay(topic: str, word_limit: int, tone: str, outline: str | None = None) -> str:
    client = llm_client.get_client()
    user_prompt = prompts.ESSAY_USER_TEMPLATE.format(
        topic=topic,
        word_limit=word_limit,
        tone=tone,
        outline=outline or "(none provided)",
    )
    return client.generate(prompts.ESSAY_SYSTEM_PROMPT, user_prompt, max_tokens=min(word_limit * 2, 800))
