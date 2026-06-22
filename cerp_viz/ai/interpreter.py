"""
AI chart interpretation — depends on the anthropic SDK and ANTHROPIC_API_KEY.
Designed to degrade gracefully: callers check is_available() first.
No Streamlit or chart-layer imports here.
"""
from __future__ import annotations
import os
from typing import Any

_MODEL = "claude-sonnet-4-6"


def is_available() -> bool:
    """True only if the anthropic package is installed and an API key is set."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def interpret_chart(
    viz_name: str,
    viz_description: str,
    columns: dict[str, str | None],
    params: dict[str, Any],
    df_stats: dict[str, dict],
    warnings: list[str],
) -> str:
    """
    Call Claude to produce a plain-English interpretation of the chart.
    Raises RuntimeError if anthropic is not available.
    """
    if not is_available():
        raise RuntimeError("anthropic SDK not installed or ANTHROPIC_API_KEY not set.")

    import anthropic

    col_lines   = "\n".join(f"  {role}: {col}" for role, col in columns.items() if col)
    param_lines = "\n".join(f"  {k}: {v}" for k, v in params.items())
    stat_lines  = "\n".join(
        f"  {col}: min={s.get('min'):.2f}, max={s.get('max'):.2f}, mean={s.get('mean'):.2f}"
        for col, s in df_stats.items()
    )
    adj_lines   = "\n".join(f"  - {w}" for w in warnings) if warnings else "  None"

    prompt = f"""You are a senior data analyst reviewing a chart produced from an Excel file.

Chart type: {viz_name}
Description: {viz_description}

Columns mapped:
{col_lines}

Assumptions applied:
{param_lines}

Key statistics of plotted columns:
{stat_lines}

Data adjustments made before rendering:
{adj_lines}

Write a concise 2–3 sentence plain-English interpretation for a non-technical business user:
1. What the chart shows at a high level.
2. The most important pattern, trend, or insight visible.
3. Any caveat the user should know given the adjustments made.

Be direct and specific. Do not mention that you are an AI."""

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=_MODEL,
        max_tokens=350,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
