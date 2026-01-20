from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

from config import settings

load_dotenv()


class LLMClient:
    """Groq-based LLM client for AI Learning Assistant."""
    
    def __init__(self):
        self.groq_model = settings.GROQ_MODEL
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        if not self.groq_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY environment variable. "
                "Please set it in .env file or environment. "
                "Get your free API key at https://console.groq.com"
            )
        
        self.client = Groq(api_key=self.groq_key)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Groq API.
        
        Args:
            system_prompt: System instruction for the model
            user_prompt: User query or instruction
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            chat = self.client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return chat.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(
                f"Groq API request failed: {exc}. "
                f"Check your GROQ_API_KEY and internet connection. "
                f"Current model: {self.groq_model}"
            )


def get_client() -> LLMClient:
    """Get singleton LLM client instance."""
    return LLMClient()
