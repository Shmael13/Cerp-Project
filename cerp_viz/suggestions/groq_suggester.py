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
    import math
    lines = [f"Shape: {len(df)} rows × {len(df.columns)} columns", ""]

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # ── Column overview ────────────────────────────────────────────────────────
    lines.append("Columns:")
    for col in df.columns:
        s         = df[col]
        n_unique  = int(s.nunique())
        null_pct  = float(s.isna().mean() * 100)
        null_tag  = f"  {null_pct:.0f}% null" if null_pct >= 1 else ""
        dtype_str = str(s.dtype)

        if col in num_cols:
            valid = s.dropna()
            rng   = f"[{valid.min():.3g} … {valid.max():.3g}]" if len(valid) else "[]"
            skew  = float(valid.skew()) if len(valid) > 3 else 0.0
            skew_tag = f"  skew={skew:.2f}" if abs(skew) > 0.5 else ""
            lines.append(f"  {col!r:28s} numeric   unique={n_unique:4d}  range={rng}{skew_tag}{null_tag}")
        elif col in dt_cols:
            valid = s.dropna()
            rng   = f"{valid.min()} → {valid.max()}" if len(valid) else "empty"
            lines.append(f"  {col!r:28s} datetime  {rng}{null_tag}")
        else:
            # Detect hidden datetime columns
            try:
                parsed = pd.to_datetime(s.dropna().head(20), errors="raise", infer_datetime_format=True)
                rng    = f"{parsed.min()} → {parsed.max()}"
                lines.append(f"  {col!r:28s} date-str  {rng}{null_tag}")
                dt_cols.append(col)
            except Exception:
                sample = s.dropna().unique()[:5].tolist()
                lines.append(f"  {col!r:28s} categ     unique={n_unique:4d}  top={sample}{null_tag}")

    # ── Numeric statistics ─────────────────────────────────────────────────────
    if num_cols:
        lines.append("")
        lines.append("Numeric statistics (mean / std / min / median / max / skew):")
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            lines.append(
                f"  {col!r:28s}  "
                f"mean={s.mean():>10.3g}  std={s.std():>9.3g}  "
                f"min={s.min():>10.3g}  median={s.median():>10.3g}  max={s.max():>10.3g}  "
                f"skew={s.skew():>5.2f}"
            )

    # ── Categorical top values ─────────────────────────────────────────────────
    cat_only = [c for c in cat_cols if c not in dt_cols]
    if cat_only:
        lines.append("")
        lines.append("Top values (categorical columns):")
        for col in cat_only[:10]:   # cap at 10 cols
            vc = df[col].value_counts().head(6)
            pairs = "  ".join(f"{str(v)!r}:{cnt}" for v, cnt in vc.items())
            lines.append(f"  {col!r:28s}  {pairs}")

    # ── Correlations ──────────────────────────────────────────────────────────
    if len(num_cols) >= 2:
        lines.append("")
        lines.append("Top numeric correlations (|r| > 0.4):")
        corr = df[num_cols].corr().abs()
        pairs_seen: set[frozenset] = set()
        corr_pairs: list[tuple[float, str, str]] = []
        for a in num_cols:
            for b in num_cols:
                if a == b:
                    continue
                key = frozenset([a, b])
                if key in pairs_seen:
                    continue
                pairs_seen.add(key)
                r = corr.loc[a, b]
                if not math.isnan(r) and r > 0.4:
                    corr_pairs.append((r, a, b))
        for r, a, b in sorted(corr_pairs, reverse=True)[:8]:
            raw_r = df[[a, b]].corr().iloc[0, 1]
            direction = "+" if raw_r > 0 else "−"
            lines.append(f"  {a!r:20s} ↔ {b!r:20s}  r={direction}{r:.3f}")
        if not corr_pairs:
            lines.append("  (no strong correlations found)")

    # ── Date ranges ───────────────────────────────────────────────────────────
    true_dt = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if true_dt:
        lines.append("")
        lines.append("Date ranges:")
        for col in true_dt:
            valid = df[col].dropna()
            span  = (valid.max() - valid.min()).days if len(valid) >= 2 else 0
            lines.append(f"  {col!r:28s}  {valid.min()} → {valid.max()}  ({span} days)")

    # ── Sample rows ───────────────────────────────────────────────────────────
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
For each suggestion, also recommend any data transforms that would make the chart clearer or more meaningful.

RESPOND WITH ONLY a JSON array (no markdown, no text before or after):
[
  {{
    "chart_name": "<exact chart name from the list above>",
    "title": "<short specific headline e.g. 'Revenue by Region'>",
    "rationale": "<one sentence: what the chart reveals and why it matters>",
    "columns": {{"<role>": "<column name or null>"}},
    "params": {{}},
    "score": <float 0.0-1.0>,
    "transforms": [
      {{
        "type": "filter",
        "column": "<column name>",
        "operator": "<one of: =, ≠, >, <, ≥, ≤, contains, not contains, is blank, is not blank, starts with, ends with, in list>",
        "value": "<string value (comma-separated list for 'in list')>",
        "label": "<short description>"
      }},
      {{
        "type": "date_part",
        "source_column": "<datetime column name>",
        "part": "<one of: Year, Quarter, Month, Month Num, Week Num, Day of Month, Day of Week, Hour>",
        "label": "<short description>"
      }},
      {{
        "type": "top_n",
        "cat_column": "<categorical column name>",
        "num_column": "<numeric column name to rank by>",
        "n": <integer, typically 10>,
        "agg": "<one of: sum, mean, max, count>",
        "label": "<short description>"
      }},
      {{
        "type": "derived",
        "name": "<new column name, no spaces>",
        "expression": "<pandas eval expression using existing column names e.g. `Revenue` / `Cost`>",
        "label": "<short description>"
      }},
      {{
        "type": "col_expr",
        "column": "<existing column name to transform>",
        "expression": "<math expression using 'x' as the column value — e.g. log(x), sqrt(x), x/1000, x**2, clip(x,0,100)>",
        "new_name": "<optional new column name; leave empty to overwrite in-place>",
        "label": "<short description e.g. 'Log-scale the revenue axis'>"
      }},
      {{
        "type": "bin",
        "column": "<numeric column to discretise>",
        "n_bins": <integer, e.g. 10>,
        "strategy": "<equal_width or quantile>",
        "new_name": "<optional new column name>",
        "label": "<short description e.g. 'Bin age into 10 equal-width groups'>"
      }},
      {{
        "type": "normalize",
        "column": "<numeric column to scale>",
        "method": "<min_max | z_score | pct_of_total>",
        "new_name": "<optional new column name>",
        "label": "<short description>"
      }},
      {{
        "type": "fill_nulls",
        "column": "<column with missing values>",
        "method": "<mean | median | zero | ffill | bfill | value>",
        "value": "<literal fill value — only used when method is 'value'>",
        "label": "<short description>"
      }},
      {{
        "type": "rolling",
        "column": "<numeric column>",
        "window": <integer window size, e.g. 3 or 7>,
        "agg": "<mean | sum | max | min | std>",
        "sort_col": "<column to sort by before rolling, e.g. a date column — leave empty if already ordered>",
        "new_name": "<optional new column name>",
        "label": "<short description e.g. '7-period rolling average of sales'>"
      }}
    ]
  }}
]

Rules:
- chart_name must exactly match one of the names above.
- Column names in columns and transforms must exist verbatim in the DataFrame.
- Only include transforms that genuinely improve the chart.
- Use col_expr when a log/sqrt/power axis transform clarifies skewed distributions.
- Use rolling only for time-series data with a clear sort column.
- Use bin to convert continuous numeric columns into categorical buckets for bar/heatmap.
- Use normalize for radar charts so different metrics are comparable.
- transforms can be an empty array if no transforms are needed.
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

            # Validate transforms — drop entries with missing required fields
            raw_transforms = item.get("transforms", []) or []
            clean_transforms = []
            for t in raw_transforms:
                if not isinstance(t, dict):
                    continue
                kind = t.get("type")
                if kind == "filter" and t.get("column") and t.get("operator"):
                    clean_transforms.append(t)
                elif kind == "date_part" and t.get("source_column") and t.get("part"):
                    clean_transforms.append(t)
                elif kind == "top_n" and t.get("cat_column") and t.get("num_column"):
                    clean_transforms.append(t)
                elif kind == "derived" and t.get("name") and t.get("expression"):
                    clean_transforms.append(t)
                elif kind == "col_expr" and t.get("column") and t.get("expression"):
                    clean_transforms.append(t)
                elif kind == "bin" and t.get("column"):
                    clean_transforms.append(t)
                elif kind == "normalize" and t.get("column"):
                    clean_transforms.append(t)
                elif kind == "fill_nulls" and t.get("column"):
                    clean_transforms.append(t)
                elif kind == "rolling" and t.get("column"):
                    clean_transforms.append(t)

            results.append(SuggestionResult(
                chart_name=chart_name,
                columns=columns,
                params=params,
                title=str(item.get("title", chart_name)),
                rationale=str(item.get("rationale", "")),
                score=float(item.get("score", 0.5)),
                transforms=clean_transforms,
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
