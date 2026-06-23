"""
AI-powered chart suggester using Groq (llama-3.3-70b-versatile).
Same interface as AISuggester — returns [] on any failure.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from cerp_viz.ai.groq_client import chat, is_available
from cerp_viz.core.registry import registry
from cerp_viz.core.suggestions import BaseSuggester, SuggestionResult

_MAX_SAMPLE_ROWS  = 4
_MAX_SUGGESTIONS  = 5


def _describe_df(df: pd.DataFrame) -> str:
    lines = [f"Rows: {len(df)},  Columns: {len(df.columns)}", ""]
    lines.append("Column overview:")
    for col in df.columns:
        n_unique    = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:4].tolist()
        lines.append(f"  {col!r:30s} dtype={str(df[col].dtype):12s} unique={n_unique:4d}  sample={sample_vals}")

    lines.append("")
    lines.append("Numeric statistics:")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        stats = df[num_cols].describe().T[["mean", "std", "min", "max"]].round(2)
        for col, row in stats.iterrows():
            lines.append(f"  {col!r:30s}  mean={row['mean']:>10.2f}  std={row['std']:>10.2f}  "
                         f"min={row['min']:>10.2f}  max={row['max']:>10.2f}")

    lines.append("")
    lines.append(f"First {_MAX_SAMPLE_ROWS} rows:")
    lines.append(df.head(_MAX_SAMPLE_ROWS).to_string(index=False))
    return "\n".join(lines)


def _available_charts() -> str:
    lines = []
    for name in registry.names():
        viz   = registry.get(name)()
        roles = [f"{c.role}({'required' if c.required else 'optional'}, {c.dtype})"
                 for c in viz.required_columns()]
        lines.append(f"  {name}: roles=[{', '.join(roles)}]")
    return "\n".join(lines)


def _build_prompt(df: pd.DataFrame) -> str:
    return f"""You are a senior data analyst advising a business user who just uploaded a spreadsheet.

DATAFRAME DESCRIPTION:
{_describe_df(df)}

AVAILABLE CHART TYPES:
{_available_charts()}

TASK:
Suggest the {_MAX_SUGGESTIONS} most insightful visualizations for this data.
Only suggest a chart if its required columns can be filled from the DataFrame above.

RESPOND WITH ONLY a JSON array (no markdown, no text before or after):
[
  {{
    "chart_name": "<exact chart name from the list above>",
    "title": "<short specific headline e.g. 'Revenue by Region'>",
    "rationale": "<one sentence: what the chart reveals and why it matters>",
    "columns": {{"<role>": "<column name or null>"}},
    "params": {{}},
    "score": <float 0.0-1.0>
  }}
]

Rules:
- chart_name must exactly match one of the names above.
- Column names must exist verbatim in the DataFrame.
- Return the array sorted by descending score."""


def _parse_response(raw: str, df: pd.DataFrame) -> list[SuggestionResult]:
    # Strip markdown fences if the model adds them despite instructions
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    data = json.loads(raw)
    if not isinstance(data, list):
        return []

    valid_names = set(registry.names())
    df_cols     = set(df.columns)
    results: list[SuggestionResult] = []

    for item in data:
        try:
            chart_name = item["chart_name"]
            if chart_name not in valid_names:
                continue

            columns: dict[str, str | None] = {}
            for role, col in item.get("columns", {}).items():
                if col is not None and col not in df_cols:
                    col = None
                columns[role] = col

            cls      = registry.get(chart_name)
            defaults: dict[str, Any] = {s.key: s.default for s in cls().assumptions()}
            params   = {**defaults, **item.get("params", {})}

            results.append(SuggestionResult(
                chart_name=chart_name,
                columns=columns,
                params=params,
                title=str(item.get("title", chart_name)),
                rationale=str(item.get("rationale", "")),
                score=float(item.get("score", 0.5)),
            ))
        except Exception:
            continue

    return results


class GroqSuggester(BaseSuggester):
    """Calls Groq to generate chart suggestions. Returns [] on any error."""

    def suggest(self, df: pd.DataFrame) -> list[SuggestionResult]:
        if not is_available():
            return []
        try:
            raw = chat(_build_prompt(df), max_tokens=1400)
            return sorted(_parse_response(raw, df), key=lambda r: r.score, reverse=True)
        except Exception:
            return []
