"""
Plain-English chart interpretation via Groq.
Degrades gracefully: callers check is_available() before calling.
"""
from __future__ import annotations

from typing import Any

from cerp_viz.ai.groq_client import chat, is_available


def interpret_chart(
    viz_name: str,
    viz_description: str,
    columns: dict[str, str | None],
    params: dict[str, Any],
    df_stats: dict[str, dict],
    warnings: list[str],
) -> str:
    """Return a 2-3 sentence plain-English interpretation of the chart."""
    if not is_available():
        raise RuntimeError("GROQ_API_KEY not set or groq package missing.")

    col_lines   = "\n".join(f"  {role}: {col}" for role, col in columns.items() if col)
    param_lines = "\n".join(f"  {k}: {v}" for k, v in params.items())
    stat_lines  = "\n".join(
        f"  {col}: min={s.get('min', 0):.2f}, max={s.get('max', 0):.2f}, mean={s.get('mean', 0):.2f}"
        for col, s in df_stats.items()
    )
    adj_lines = "\n".join(f"  - {w}" for w in warnings) if warnings else "  None"

    prompt = f"""You are a senior data analyst reviewing a business chart.

Chart type: {viz_name}
Description: {viz_description}

Columns mapped:
{col_lines}

Key statistics of plotted columns:
{stat_lines}

Data adjustments made before rendering:
{adj_lines}

Write a concise 2-3 sentence plain-English interpretation for a non-technical business user:
1. What the chart shows at a high level.
2. The most important pattern, trend, or insight visible.
3. Any caveat the user should know.

Be direct and specific. Do not mention that you are an AI."""

    return chat(prompt, max_tokens=350)
