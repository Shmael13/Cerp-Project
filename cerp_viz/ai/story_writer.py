"""
AI data story writer — extends the chart interpreter to produce a full narrative.
Requires anthropic SDK + ANTHROPIC_API_KEY (same gate as interpreter.py).
"""
from __future__ import annotations
import os
from typing import Any

_MODEL = "claude-sonnet-4-6"


def is_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def write_story(
    viz_name: str,
    columns: dict[str, str | None],
    params: dict[str, Any],
    df_stats: dict[str, dict],
    stat_bullets: list[str],
    warnings: list[str],
) -> str:
    """
    Call Claude to produce a 3-paragraph data story:
      1. Overview — what the chart shows
      2. Key insight — the single most important finding
      3. Recommendation — one concrete action the business should take

    Returns the story as a markdown string.
    Raises RuntimeError if not available.
    """
    if not is_available():
        raise RuntimeError("anthropic SDK not installed or ANTHROPIC_API_KEY not set.")

    import anthropic

    col_lines    = "\n".join(f"  {role}: {col}" for role, col in columns.items() if col)
    stat_lines   = "\n".join(
        f"  {col}: min={s.get('min', 0):.2f}, max={s.get('max', 0):.2f}, mean={s.get('mean', 0):.2f}"
        for col, s in df_stats.items()
    )
    bullet_text  = "\n".join(f"  - {b}" for b in stat_bullets) if stat_bullets else "  (none)"
    warning_text = "\n".join(f"  - {w}" for w in warnings)     if warnings     else "  None"

    prompt = f"""You are a senior data analyst writing a concise data story for a business audience.

Chart type: {viz_name}

Columns shown:
{col_lines}

Key statistics:
{stat_lines}

Statistical observations already computed:
{bullet_text}

Data caveats:
{warning_text}

Write a data story in exactly 3 paragraphs using Markdown:

**Paragraph 1 — Overview**: What this chart shows and the overall pattern in one or two sentences.

**Paragraph 2 — Key Insight**: The single most important, specific finding (mention actual numbers). \
What does this mean for the business?

**Paragraph 3 — Recommendation**: One concrete, actionable recommendation based on the data.

Be direct, specific, and quantitative. Address a non-technical executive audience. \
Do not mention that you are an AI. Do not use bullet points — prose only."""

    client  = anthropic.Anthropic()
    message = client.messages.create(
        model=_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
