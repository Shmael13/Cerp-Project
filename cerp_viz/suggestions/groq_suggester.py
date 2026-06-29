"""
AI-powered chart suggester using Groq (llama-3.3-70b-versatile).
Same interface as AISuggester — returns [] on any failure.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from cerp_viz.ai.groq_client import chat, is_available
from cerp_viz.core.registry import registry
from cerp_viz.core.suggestions import BaseSuggester, SuggestionResult

_MAX_SAMPLE_ROWS = 5
_MAX_SUGGESTIONS = 8


# ── Deep data description ──────────────────────────────────────────────────────

def _describe_df(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    lines = [f"Shape: {len(df):,} rows × {len(df.columns)} columns", ""]

    # ── Column overview ────────────────────────────────────────────────────────
    lines.append("── Columns ──")
    for col in df.columns:
        s        = df[col]
        n_unique = int(s.nunique())
        null_pct = float(s.isna().mean() * 100)
        null_tag = f"  {null_pct:.0f}% null" if null_pct >= 1 else ""

        if col in num_cols:
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v) == 0:
                lines.append(f"  {col!r:30s} numeric  (all null)")
                continue
            skew = float(v.skew()) if len(v) > 3 else 0.0
            kurt = float(v.kurtosis()) if len(v) > 3 else 0.0
            n_neg  = int((v < 0).sum())
            n_zero = int((v == 0).sum())
            q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
            iqr    = q3 - q1
            n_out  = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
            flags  = []
            if abs(skew) > 1.5:  flags.append(f"skew={skew:+.1f}")
            if abs(kurt)  > 3:   flags.append(f"kurt={kurt:.1f}")
            if n_neg  > 0:       flags.append(f"{n_neg} neg")
            if n_zero > 0:       flags.append(f"{n_zero} zeros")
            if n_out  > 0:       flags.append(f"{n_out} outliers")
            flag_str = "  [" + ", ".join(flags) + "]" if flags else ""
            lines.append(
                f"  {col!r:30s} numeric  "
                f"range=[{v.min():.4g}…{v.max():.4g}]  "
                f"mean={v.mean():.4g}  median={v.median():.4g}  "
                f"std={v.std():.4g}{flag_str}{null_tag}"
            )
        elif col in dt_cols:
            v   = s.dropna()
            rng = f"{v.min()} → {v.max()}" if len(v) else "empty"
            span = f"({(v.max()-v.min()).days}d span)" if len(v) >= 2 else ""
            lines.append(f"  {col!r:30s} datetime  {rng}  {span}{null_tag}")
        else:
            # Detect hidden datetime strings
            try:
                parsed = pd.to_datetime(s.dropna().head(30), errors="raise", infer_datetime_format=True)
                rng = f"{parsed.min()} → {parsed.max()}"
                lines.append(f"  {col!r:30s} date-str  {rng}{null_tag}")
                dt_cols.append(col)
                continue
            except Exception:
                pass
            vc     = s.value_counts()
            top5   = list(vc.head(5).index)
            dom_pct = float(vc.iloc[0] / len(s) * 100) if len(vc) else 0
            lines.append(
                f"  {col!r:30s} categ    unique={n_unique}  "
                f"top={top5}  dom={dom_pct:.0f}%{null_tag}"
            )

    # ── Full numeric statistics ────────────────────────────────────────────────
    if num_cols:
        lines += ["", "── Numeric statistics ──"]
        for col in num_cols:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) == 0:
                continue
            cv = float(v.std() / v.mean() * 100) if v.mean() != 0 else float("inf")
            lines.append(
                f"  {col!r:30s}  "
                f"min={v.min():.4g}  p25={v.quantile(.25):.4g}  "
                f"median={v.median():.4g}  p75={v.quantile(.75):.4g}  max={v.max():.4g}  "
                f"mean={v.mean():.4g}  std={v.std():.4g}  CV={cv:.0f}%"
            )

    # ── Categorical top-value breakdown ────────────────────────────────────────
    cat_only = [c for c in cat_cols if c not in dt_cols]
    if cat_only:
        lines += ["", "── Categorical top values ──"]
        for col in cat_only[:12]:
            vc    = df[col].value_counts()
            pairs = "  ".join(
                f"{str(v)!r}:{cnt}({cnt/len(df)*100:.0f}%)"
                for v, cnt in vc.head(7).items()
            )
            lines.append(f"  {col!r:30s}  {pairs}")

    # ── Correlations ──────────────────────────────────────────────────────────
    if len(num_cols) >= 2:
        lines += ["", "── Numeric correlations (|r| ≥ 0.35) ──"]
        corr     = df[num_cols].corr()
        seen: set[frozenset] = set()
        pairs: list[tuple[float, str, str]] = []
        for a in num_cols:
            for b in num_cols:
                if a == b:
                    continue
                key = frozenset([a, b])
                if key in seen:
                    continue
                seen.add(key)
                r = float(corr.loc[a, b])
                if not math.isnan(r) and abs(r) >= 0.35:
                    pairs.append((abs(r), a, b, r))
        for _, a, b, r in sorted(pairs, reverse=True)[:10]:
            lines.append(f"  {a!r:22s} ↔ {b!r:22s}  r={r:+.3f}")
        if not pairs:
            lines.append("  (no strong correlations found)")

    # ── Cross-category averages ────────────────────────────────────────────────
    if cat_only and num_cols:
        lines += ["", "── Per-category averages (top 5 cats × top 4 numeric cols) ──"]
        for cat in cat_only[:2]:
            top_cats = df[cat].value_counts().head(5).index
            sub      = df[df[cat].isin(top_cats)]
            show_num = num_cols[:4]
            agg      = sub.groupby(cat)[show_num].mean()
            lines.append(f"  {cat}:")
            for cv in top_cats:
                if cv not in agg.index:
                    continue
                row  = agg.loc[cv]
                vals = "  ".join(f"{c}={v:.3g}" for c, v in zip(show_num, row))
                lines.append(f"    {str(cv)!r:18s} → {vals}")

    # ── Time-series trend hints ────────────────────────────────────────────────
    true_dt = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if true_dt and num_cols:
        lines += ["", "── Time-series trends ──"]
        for dt_col in true_dt[:2]:
            tmp = df[[dt_col] + num_cols[:3]].dropna()
            if len(tmp) < 4:
                continue
            tmp = tmp.sort_values(dt_col)
            for nc in num_cols[:3]:
                vals = pd.to_numeric(tmp[nc], errors="coerce").dropna()
                if len(vals) < 4:
                    continue
                # Simple linear trend via numpy polyfit
                x_idx = np.arange(len(vals))
                slope = float(np.polyfit(x_idx, vals.values, 1)[0])
                pct   = slope / float(vals.mean()) * 100 if vals.mean() != 0 else 0
                direction = "↑ upward" if pct > 1 else "↓ downward" if pct < -1 else "→ flat"
                lines.append(f"  {nc!r} over {dt_col!r}: {direction} trend ({pct:+.1f}% per period)")

    # ── Data quality ──────────────────────────────────────────────────────────
    quality: list[str] = []
    n_dup = int(df.duplicated().sum())
    if n_dup:
        quality.append(f"{n_dup} duplicate rows ({n_dup/len(df)*100:.1f}%)")
    const_cols = [c for c in num_cols if pd.to_numeric(df[c], errors="coerce").nunique() <= 1]
    if const_cols:
        quality.append(f"constant/empty numeric columns: {const_cols}")
    if quality:
        lines += ["", "── Data quality ──"]
        for q in quality:
            lines.append(f"  {q}")

    # ── Sample rows ───────────────────────────────────────────────────────────
    lines += ["", f"── First {_MAX_SAMPLE_ROWS} rows ──"]
    lines.append(df.head(_MAX_SAMPLE_ROWS).to_string(index=False))

    return "\n".join(lines)


# ── Data-driven formula hints ─────────────────────────────────────────────────

def _formula_hints(df: pd.DataFrame) -> str:
    """
    Compute concrete, ready-to-run formula recommendations from data statistics.
    Gives the AI specific expressions to use rather than generic placeholders.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    lines: list[str] = []

    for col in num_cols:
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(v) < 5:
            continue

        skew   = float(v.skew())
        n_neg  = int((v < 0).sum())
        n_zero = int((v == 0).sum())
        vmin, vmax = float(v.min()), float(v.max())
        q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
        iqr = q3 - q1

        col_hints: list[str] = []

        # ── Log transform ──────────────────────────────────────────────────────
        if skew > 1.5:
            if n_neg == 0 and n_zero == 0:
                col_hints.append(f"right-skewed (skew={skew:.2f}), all positive → col_expr: log(x)")
            elif n_neg == 0 and n_zero > 0:
                col_hints.append(f"right-skewed (skew={skew:.2f}), {n_zero} zeros → col_expr: log(x + 1)")
            elif n_neg > 0:
                col_hints.append(f"right-skewed with negatives → col_expr: log(abs(x) + 1)")
        elif skew < -1.5:
            col_hints.append(f"left-skewed (skew={skew:.2f}) → col_expr: x**2  or  exp(x/max) if bounded")

        # ── Magnitude scaling ──────────────────────────────────────────────────
        if vmax >= 1_000_000_000:
            col_hints.append(f"values in billions (max={vmax:,.0f}) → col_expr: x / 1000000000")
        elif vmax >= 1_000_000:
            col_hints.append(f"values in millions (max={vmax:,.0f}) → col_expr: x / 1000000")
        elif vmax >= 10_000:
            col_hints.append(f"values in thousands (max={vmax:,.0f}) → col_expr: x / 1000")

        # ── Outlier clipping ───────────────────────────────────────────────────
        if iqr > 0:
            lo_bound = round(max(vmin, q1 - 1.5 * iqr), 4)
            hi_bound = round(q3 + 1.5 * iqr, 4)
            n_out = int(((v < lo_bound) | (v > hi_bound)).sum())
            out_pct = n_out / len(v) * 100
            if out_pct >= 3:
                col_hints.append(
                    f"{n_out} outliers ({out_pct:.0f}%) beyond IQR fence "
                    f"[{lo_bound:.4g}, {hi_bound:.4g}] → col_expr: clip(x, {lo_bound:.4g}, {hi_bound:.4g})"
                )

        # ── Square root for counts ─────────────────────────────────────────────
        if v.dtype.kind in "iu" and vmax > 100 and 0.5 < skew < 2.0:
            col_hints.append(f"integer count column, moderate skew → col_expr: sqrt(x)")

        # ── Percentage / fraction normalisation ───────────────────────────────
        if 0 < vmax <= 1 and vmin >= 0:
            col_hints.append(f"looks like a 0-1 fraction → col_expr: x * 100  (convert to %)")

        if col_hints:
            lines.append(f"  '{col}':")
            for h in col_hints:
                lines.append(f"    • {h}")

    # ── Ratio / derived column hints ──────────────────────────────────────────
    if len(num_cols) >= 2:
        lines.append("  Potential derived columns (check if meaningful for this domain):")
        seen: set[frozenset] = set()
        for a in num_cols:
            for b in num_cols:
                if a == b:
                    continue
                key = frozenset([a, b])
                if key in seen:
                    continue
                seen.add(key)
                va = pd.to_numeric(df[a], errors="coerce").dropna()
                vb = pd.to_numeric(df[b], errors="coerce").dropna()
                if len(va) < 5 or len(vb) < 5:
                    continue
                if (vb == 0).mean() > 0.05:
                    continue   # too many zeros in denominator
                try:
                    ratio = (va / vb).replace([np.inf, -np.inf], np.nan).dropna()
                    if ratio.std() < ratio.abs().mean() * 2:  # ratio is stable
                        safe_a = a.replace(" ", "_").replace("/", "_per_")
                        safe_b = b.replace(" ", "_").replace("/", "_per_")
                        lines.append(
                            f"    • '{a}' / '{b}' is stable "
                            f"(mean={ratio.mean():.3g}) → derived: "
                            f"name={safe_a}_per_{safe_b}  expr=`{a}` / `{b}`"
                        )
                except Exception:
                    pass

    return "\n".join(lines) if lines else "  (no specific formula hints for this dataset)"


# ── Chart catalogue with descriptions ─────────────────────────────────────────

def _available_charts() -> str:
    lines = []
    for name in registry.names():
        viz   = registry.get(name)()
        desc  = getattr(viz, "description", "")
        roles = []
        for c in viz.required_columns():
            req = "required" if c.required else "optional"
            roles.append(f"{c.role}[{req},{c.dtype}]={c.label!r}")
        lines.append(f"  {name}:")
        lines.append(f"    description: {desc}")
        lines.append(f"    columns: {', '.join(roles)}")
    return "\n".join(lines)


_CHART_SELECTION_GUIDE = """\
CHART SELECTION GUIDE — use this to pick the RIGHT chart from similar options:

SCATTER family (2 numeric columns):
  • Scatter Plot        — general correlation exploration, <5 k rows, want trendline/R²
  • Marginal Scatter    — use when you ALSO want to see each variable's distribution shape
                          on the margins (histogram/violin/box on X and Y axes simultaneously).
                          Best choice when skewness or bimodality may explain the relationship.
  • Density Heatmap     — use when rows > 1 000 and scatter would overplot; reveals WHERE
                          the mass of data lies rather than individual points.
  • Scatter Matrix      — use when there are 4+ numeric columns and you want ALL pairwise
                          views at once (SPLOM). Scale back max_cols if > 8 columns.

DISTRIBUTION family (1 numeric + optional category):
  • Distribution        — simple histogram; best for a single column with no grouping.
  • Box Plot            — compare medians, IQR, outliers across groups; n per group can be small.
  • Violin Plot         — use over Box Plot when skewness, bimodality, or heavy tails exist
                          (skew > 1.0 in the data description is a strong signal).
  • Strip Plot          — show ALL individual points; only useful when total rows < 1 000
                          (beyond that points overlap and Violin/Box is cleaner).

FLOW family (source + target columns):
  • Sankey Diagram      — directed one-way flows where nodes appear at different "levels".
  • Chord Diagram       — bidirectional flows where the SAME entities appear on both sides
                          (e.g. trade between countries, migration between cities).
  • Network Graph       — many-to-many relationships without a clear flow direction;
                          use when entities form a graph, not a ranked hierarchy.

TEMPORAL family (date column required):
  • Line Chart          — standard time-series; good for 1–5 series.
  • Area Chart          — time-series when volume/magnitude matters as much as trend.
  • Calendar Heatmap    — daily granularity data; reveals weekday/weekend patterns.
  • Forecast            — when data has a detectable trend (R² > 0.2) and forward projection adds value.

MULTI-NUMERIC correlation:
  • Correlation Matrix  — overview of ALL pairwise correlations as a colour heatmap; pick
                          when you want a quick "which columns are related" answer.
  • Scatter Matrix      — detail-level exploration of the same; prefer when confirming
                          or disproving specific pair relationships matters more than overview.

RANKING / COMPARISON:
  • Bar Chart           — absolute values per category; standard choice.
  • Lollipop Chart      — same as Bar Chart but cleaner; prefer when n > 10 categories or
                          when visual clutter is a concern. Use horizontal orientation for long labels.
  • Waterfall           — use when values have MIXED signs (positive + negative contributions).
  • Tornado Chart       — sensitivity analysis; use when different factors push in opposite directions.
  • Bump Chart          — rank changes over time (requires date + category + value).

PART-OF-WHOLE:
  • Pie / Donut         — ≤ 8 categories, shares sum to 100%.
  • Treemap             — hierarchical part-of-whole with 2+ categorical levels.
  • Sunburst            — same as Treemap but in a radial layout; better when depth > 2 levels.

Use these rules to AVOID common mistakes:
- Never suggest Strip Plot for > 1 000 rows.
- Never suggest Marginal Scatter without at least 2 numeric columns.
- Never suggest Chord Diagram when source and target values have NO overlap (use Sankey instead).
- Never suggest Forecast unless there is a datetime column.
- Prefer Violin over Box Plot when the numeric column description mentions skew > 1.5 or kurt > 3.
- Prefer Marginal Scatter over plain Scatter when you want to highlight distributional asymmetry.\
"""


# ── Chart-type detection for query ────────────────────────────────────────────

from cerp_viz.suggestions._utils import detect_chart_from_query  # noqa: E402


def _task_section(df: pd.DataFrame, query: str) -> str:
    """Build the TASK section of the prompt, adapting to the user's query."""
    chart_target = detect_chart_from_query(query) if query else None

    formula_rules = """\
CRITICAL FORMULA RULES — you must follow these exactly:
1. col_expr: write the EXACT ready-to-run expression using real numbers from FORMULA HINTS \
   (e.g. clip(x, 0, 847.5) not clip(x, lo, hi); x / 1000000 not x / scale).
2. derived: complete pandas eval with column names in backticks (`Revenue` / `Cost` * 100). \
   Use actual DataFrame column names.
3. No placeholder text like '<value>', 'threshold', 'scale', or 'N' — only real numbers.
4. For log transforms, check FORMULA HINTS to decide log(x) vs log(x+1) based on zeros."""

    if not query:
        return f"""\
Suggest {_MAX_SUGGESTIONS} visualizations. Aim for variety — use different chart types and \
explore different analytical angles (distribution, trend, ranking, correlation, composition, comparison).

{formula_rules}"""

    if chart_target:
        # User asked for a specific chart type — give N variations of it
        return f"""\
USER REQUEST: "{query}"

The user specifically wants {chart_target} charts. Generate {_MAX_SUGGESTIONS} DIFFERENT \
{chart_target} configurations — vary the column choices, aggregations, and transforms so each \
card reveals a genuinely distinct insight. Avoid repeating the same columns.

If the data doesn't support {_MAX_SUGGESTIONS} truly distinct {chart_target} charts, fill \
remaining slots with the most closely related chart types.

{formula_rules}"""

    # General analytical question
    return f"""\
USER REQUEST: "{query}"

Answer the user's question with {_MAX_SUGGESTIONS} visualizations that directly address it. \
Choose whatever chart types best reveal the answer. Each chart should show a different angle \
on the same question.

{formula_rules}"""


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(df: pd.DataFrame, query: str = "") -> str:
    return f"""You are a world-class data analyst. A business user uploaded a spreadsheet. \
Your job is to suggest insightful, distinct, and immediately actionable visualizations, \
complete with any data transforms needed to make each chart as clear as possible.

══════════════════════════════════════════════════════════════
DATAFRAME DESCRIPTION
══════════════════════════════════════════════════════════════
{_describe_df(df)}

══════════════════════════════════════════════════════════════
FORMULA HINTS  (data-driven — use these to write EXACT expressions)
══════════════════════════════════════════════════════════════
{_formula_hints(df)}

══════════════════════════════════════════════════════════════
AVAILABLE CHART TYPES
══════════════════════════════════════════════════════════════
{_available_charts()}

══════════════════════════════════════════════════════════════
{_CHART_SELECTION_GUIDE}
══════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════
TASK
══════════════════════════════════════════════════════════════
{_task_section(df, query)}

RESPOND WITH ONLY a valid JSON array (no markdown, no explanation, no text outside the array):
[
  {{
    "chart_name": "<exact name from AVAILABLE CHART TYPES>",
    "title": "<specific headline, e.g. 'Monthly Revenue Trend 2022–2024'>",
    "rationale": "<one sentence: what insight this chart reveals>",
    "columns": {{"<role>": "<exact column name from DataFrame, or null if optional and unused>"}},
    "params": {{}},
    "score": <float 0.0–1.0, higher = more insightful for this specific dataset>,
    "transforms": [
      {{
        "type": "filter",
        "column": "<exact column name>",
        "operator": "<one of: =, ≠, >, <, ≥, ≤, contains, not contains, is blank, is not blank, starts with, ends with, in list>",
        "value": "<literal value; comma-separated list for 'in list'>",
        "label": "<what this filter does>"
      }},
      {{
        "type": "date_part",
        "source_column": "<exact datetime column name>",
        "part": "<one of: Year, Quarter, Month, Month Num, Week Num, Day of Month, Day of Week, Hour>",
        "label": "<e.g. 'Extract month for seasonal analysis'>"
      }},
      {{
        "type": "top_n",
        "cat_column": "<exact categorical column>",
        "num_column": "<exact numeric column to rank by>",
        "n": <integer>,
        "agg": "<sum | mean | max | count>",
        "label": "<e.g. 'Keep top 10 products by total sales'>"
      }},
      {{
        "type": "derived",
        "name": "<new_column_name_no_spaces>",
        "expression": "<pandas eval: `Col A` / `Col B` * 100>",
        "label": "<what this derived column represents>"
      }},
      {{
        "type": "col_expr",
        "column": "<exact existing column to transform>",
        "expression": "<EXACT math expression — use real numbers from FORMULA HINTS, e.g. log(x+1), clip(x, 0, 9999.5), x/1000000>",
        "new_name": "<new column name, or empty to overwrite in-place>",
        "label": "<why this transform helps, e.g. 'Log-transform Price to correct right skew (skew=3.2)'>"
      }},
      {{
        "type": "bin",
        "column": "<exact numeric column>",
        "n_bins": <integer, e.g. 10>,
        "strategy": "<equal_width | quantile>",
        "new_name": "<new column name>",
        "label": "<e.g. 'Bin Age into 10 equal-width groups'>"
      }},
      {{
        "type": "normalize",
        "column": "<exact numeric column>",
        "method": "<min_max | z_score | pct_of_total>",
        "new_name": "<new column name>",
        "label": "<e.g. 'Normalize to 0–1 for radar comparability'>"
      }},
      {{
        "type": "fill_nulls",
        "column": "<exact column with missing values>",
        "method": "<mean | median | zero | ffill | bfill | value>",
        "value": "<literal fill value if method=value, else empty>",
        "label": "<e.g. 'Fill 12% missing values with median'>"
      }},
      {{
        "type": "rolling",
        "column": "<exact numeric column>",
        "window": <integer, e.g. 7>,
        "agg": "<mean | sum | max | min | std>",
        "sort_col": "<column to sort by, e.g. date column — empty if already ordered>",
        "new_name": "<new column name>",
        "label": "<e.g. '7-day rolling average to smooth noise'>"
      }}
    ]
  }}
]

Additional rules:
- chart_name must exactly match one of the names in AVAILABLE CHART TYPES.
- All column names in columns{{}} and in any transform must exist verbatim in the DataFrame.
- Only include transforms that genuinely improve the chart. Empty transforms array is fine.
- Score each suggestion 0.0–1.0 for how insightful it is given THIS specific dataset.
- Sort the array by descending score.
- Return exactly {_MAX_SUGGESTIONS} suggestions (fewer only if the data genuinely cannot support more)."""


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_response(raw: str, df: pd.DataFrame) -> list[SuggestionResult]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("`").strip()

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

            raw_transforms = item.get("transforms", []) or []
            clean_transforms: list[dict] = []
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


# ── Suggester class ────────────────────────────────────────────────────────────

class GroqSuggester(BaseSuggester):
    """Calls Groq to generate chart suggestions. Returns [] on any error."""

    def suggest(self, df: pd.DataFrame, query: str = "") -> list[SuggestionResult]:
        if not is_available():
            return []
        try:
            raw = chat(_build_prompt(df, query=query), max_tokens=4000)
            return sorted(_parse_response(raw, df), key=lambda r: r.score, reverse=True)
        except Exception:
            return []
