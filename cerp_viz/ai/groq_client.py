"""
Shared Groq client utilities.
Read GROQ_API_KEY from the environment — never from code.
"""
from __future__ import annotations

import os


_MODEL = "llama-3.3-70b-versatile"


def is_available() -> bool:
    try:
        import groq  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("GROQ_API_KEY"))


def get_client():
    """Return a ready Groq client. Raises if not available."""
    if not is_available():
        raise RuntimeError("groq package not installed or GROQ_API_KEY not set.")
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def chat(prompt: str, max_tokens: int = 1024) -> str:
    """Send a single user message and return the assistant text."""
    client = get_client()
    resp = client.chat.completions.create(
        model=_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""
