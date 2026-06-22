"""
AI-powered chart suggester using the Anthropic API.

Sends a compact DataFrame description to Claude and parses the JSON response
into SuggestionResult objects.  Falls back to an empty list on any failure —
callers should always pair this with RuleBasedSuggester as a safety net.

Imports: cerp_viz.core only (+ anthropic SDK).
         Never imports cerp_viz.ui, cerp_viz.charts, cerp_viz.loaders.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from cerp_viz.core.registry import registry
from cerp_viz.core.suggestions import BaseSuggester, SuggestionResult

_MODEL = "claude-sonnet-4-6"
_MAX_SAMPLE_ROWS = 4
_MAX_SUGGESTIONS = 5


def _is_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _describe_df(df: pd.DataFrame) -> str:
    """Build a compact text description of the DataFrame for the prompt."""
    lines = [f"Rows: {len(df)},  Columns: {len(df.columns)}"]
    lines.append("")
    lines.append("Column overview:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:4].tolist()
        lines.append(f"  {col!r:30s} dtype={dtype:12s} unique={n_unique:4d}  sample={sample_vals}")

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
    """Describe available chart types and their required roles."""
    lines = []
    for name in registry.names():
        cls = registry.get(name)
        if cls is None:
            continue
        viz = cls()
        roles = [
            f"{c.role}({'required' if c.required else 'optional'}, {c.dtype})"
            for c in viz.required_columns()
        ]
        param_keys = [s.key for s in viz.assumptions() if s.category == "Data"]
        lines.append(f"  {name}: roles=[{', '.join(roles)}]  data_params=[{', '.join(param_keys)}]")
    return "\n".join(lines)


def _build_prompt(df: pd.DataFrame) -> str:
    return f"""You are a senior data analyst advising a business user who has just uploaded a spreadsheet.

DATAFRAME DESCRIPTION:
{_describe_df(df)}

AVAILABLE CHART TYPES:
{_available_charts()}

TASK:
Suggest the {_MAX_SUGGESTIONS} most insightful visualizations for this data.
Choose chart types that reveal genuinely interesting patterns — trends, comparisons, flows, distributions, or outliers.
Only suggest a chart if its required columns can be filled from the columns listed above.

RESPOND WITH ONLY a JSON array (no markdown, no prose before or after):
[
  {{
    "chart_name": "<exact chart name from the list above>",
    "title": "<short, specific headline e.g. 'Revenue by Region'>",
    "rationale": "<one sentence: what the chart reveals and why it matters>",
    "columns": {{"<role>": "<column name or null>"}},
    "params": {{"<param_key>": <value>}},
    "score": <float 0.0-1.0>
  }}
]

Rules:
- chart_name must be one of the exact names in the list above.
- column names must exist verbatim in the DataFrame.
- params should only override defaults where a non-default value makes the chart more insightful.
- score reflects how interesting and clear the visualization will be for this data.
- Return the array sorted by descending score.
"""


def _parse_response(raw: str, df: pd.DataFrame) -> list[SuggestionResult]:
    """Parse JSON, validate each entry, and return clean SuggestionResult objects."""
    data = json.loads(raw.strip())
    if not isinstance(data, list):
        return []

    valid_chart_names = set(registry.names())
    df_cols = set(df.columns)
    results: list[SuggestionResult] = []

    for item in data:
        try:
            chart_name = item["chart_name"]
            if chart_name not in valid_chart_names:
                continue

            columns: dict[str, str | None] = {}
            for role, col in item.get("columns", {}).items():
                if col is not None and col not in df_cols:
                    col = None  # drop invalid column references
                columns[role] = col

            # Merge AI params on top of defaults
            cls = registry.get(chart_name)
            defaults: dict[str, Any] = {s.key: s.default for s in cls().assumptions()}
            params = {**defaults, **item.get("params", {})}

            results.append(SuggestionResult(
                chart_name=chart_name,
                columns=columns,
                params=params,
                title=str(item.get("title", chart_name)),
                rationale=str(item.get("rationale", "")),
                score=float(item.get("score", 0.5)),
            ))
        except Exception:
            continue  # skip malformed entries silently

    return results


class AISuggester(BaseSuggester):
    """
    Calls Claude to generate chart suggestions tailored to the specific dataset.
    Returns an empty list on any error (API failure, quota, parse error, etc.)
    so the caller can fall back to RuleBasedSuggester.

    Depends only on cerp_viz.core and the anthropic SDK.
    """

    def suggest(self, df: pd.DataFrame) -> list[SuggestionResult]:
        if not _is_available():
            return []
        try:
            import anthropic
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=_MODEL,
                max_tokens=1200,
                messages=[{"role": "user", "content": _build_prompt(df)}],
            )
            raw = message.content[0].text
            return sorted(
                _parse_response(raw, df),
                key=lambda r: r.score,
                reverse=True,
            )
        except Exception:
            return []
