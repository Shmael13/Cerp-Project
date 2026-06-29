"""
Shared column-inspection helpers used by all suggester implementations.
Private to the suggestions package — no imports from outside cerp_viz.core.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from cerp_viz.core.registry import registry

# ── Column-name pattern hints ─────────────────────────────────────────────────

DATE_HINTS     = re.compile(r"date|time|year|month|quarter|week|period|day", re.I)
NUMERIC_HINTS  = re.compile(r"revenue|sales|amount|cost|price|profit|value|total|count|qty|quantity|volume|margin|spend|budget|forecast", re.I)
STAGE_HINTS    = re.compile(r"stage|step|phase|funnel|level|tier|status", re.I)
FLOW_SRC_HINTS = re.compile(r"source|from|origin|sender|supplier", re.I)
FLOW_TGT_HINTS = re.compile(r"target|to|dest|destination|receiver|customer", re.I)


# ── Column classifiers ────────────────────────────────────────────────────────

def numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def datetime_cols(df: pd.DataFrame) -> list[str]:
    """Datetime columns plus object columns whose values parse as dates."""
    dt = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    for col in df.select_dtypes(include="object").columns:
        if DATE_HINTS.search(col):
            sample = df[col].dropna().head(10)
            try:
                pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
                dt.append(col)
            except Exception:
                pass
    return dt


def categorical_cols(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    dt_names = set(datetime_cols(df))
    return [
        c for c in df.select_dtypes(exclude="number").columns
        if c not in exclude and c not in dt_names
    ]


# ── Column selectors ──────────────────────────────────────────────────────────

def best_numeric(df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    """Numeric column with highest variance; prefers finance/volume-sounding names."""
    exclude = exclude or set()
    candidates = [c for c in numeric_cols(df) if c not in exclude]
    if not candidates:
        return None
    hinted = [c for c in candidates if NUMERIC_HINTS.search(c)]
    pool = hinted if hinted else candidates
    return max(pool, key=lambda c: df[c].std())


def best_categorical(df: pd.DataFrame, target_n: int = 8,
                     exclude: set[str] | None = None) -> str | None:
    """Categorical column whose cardinality is closest to target_n."""
    exclude = exclude or set()
    candidates = [c for c in categorical_cols(df) if c not in exclude]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(df[c].nunique() - target_n))


# ── Scoring helpers ───────────────────────────────────────────────────────────

def cardinality_score(n: int, lo: int = 3, hi: int = 15) -> float:
    """1.0 when lo ≤ n ≤ hi, decays outside that range."""
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return max(0.1, n / lo)
    return max(0.1, hi / n)


def has_mixed_sign(df: pd.DataFrame, col: str) -> bool:
    return bool((df[col] > 0).any() and (df[col] < 0).any())


# ── Statistics ────────────────────────────────────────────────────────────────

def ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS R² — trend strength. Returns 0 on degenerate input."""
    try:
        x = x.astype(float)
        y = y.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 3:
            return 0.0
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        return 0.0


def pearson_r(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Pearson correlation between two columns; returns 0 on failure."""
    try:
        pair = df[[col_a, col_b]].dropna()
        if len(pair) < 3:
            return 0.0
        return float(pair[col_a].corr(pair[col_b]))
    except Exception:
        return 0.0


# ── Chart defaults & column helpers ──────────────────────────────────────────

def default_params(chart_name: str) -> dict[str, Any]:
    cls = registry.get(chart_name)
    return {} if cls is None else {s.key: s.default for s in cls().assumptions()}


def complete_columns(chart_name: str, **specified: str | None) -> dict[str, str | None]:
    """
    Return a full column mapping for chart_name.
    Roles listed in specified are used as-is; all other roles default to None.
    Guarantees the returned dict contains every role the chart declares.
    """
    cls = registry.get(chart_name)
    if cls is None:
        return dict(specified)
    return {spec.role: specified.get(spec.role) for spec in cls().required_columns()}


def validate_and_complete(
    s: "SuggestionResult", df: pd.DataFrame
) -> "SuggestionResult | None":
    """
    Validate that every required column role is mapped to a column that exists
    in df, and that every optional role is present (None if unused).
    Returns None when any required role is unresolvable.
    """
    from cerp_viz.core.suggestions import SuggestionResult as SR

    cls = registry.get(s.chart_name)
    if cls is None:
        return None

    completed: dict[str, str | None] = {}
    for spec in cls().required_columns():
        col = s.columns.get(spec.role)
        if spec.required:
            if col is None or col not in df.columns:
                return None        # can't satisfy this suggestion
        else:
            if col not in df.columns:
                col = None         # silently drop invalid optional mapping
        completed[spec.role] = col

    return SR(
        chart_name=s.chart_name,
        columns=completed,
        params=s.params,
        title=s.title,
        rationale=s.rationale,
        score=s.score,
        transforms=s.transforms,
    )


# ── Transform hint builders ───────────────────────────────────────────────────

def suggest_date_parts(df: pd.DataFrame, col: str) -> list[dict]:
    """Return the most relevant date-part hints for col based on its time granularity."""
    if col not in df.columns:
        return []
    try:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid  = parsed.dropna()
        if len(valid) < 3:
            return []

        n_years   = valid.dt.year.nunique()
        n_months  = (valid.dt.year * 100 + valid.dt.month).nunique()
        n_days    = valid.dt.date.nunique()
        parts: list[dict] = []

        # Year — only when data spans multiple years
        if n_years > 1:
            parts.append({"type": "date_part", "source_column": col, "part": "Year",
                          "label": f"Group by year  ({n_years} years)"})
        # Quarter / Month — multi-month data
        if n_months > 2:
            parts.append({"type": "date_part", "source_column": col, "part": "Quarter",
                          "label": f"Group by quarter"})
            parts.append({"type": "date_part", "source_column": col, "part": "Month",
                          "label": f"Group by month name"})
        # Day of week — sub-weekly granularity or behavioural patterns
        if n_days > 7:
            parts.append({"type": "date_part", "source_column": col, "part": "Day of Week",
                          "label": f"Group by day of week"})
        # Hour — only for intra-day data
        hour_spread = int(valid.dt.hour.nunique())
        if hour_spread > 3:
            parts.append({"type": "date_part", "source_column": col, "part": "Hour",
                          "label": f"Group by hour of day"})
        return parts
    except Exception:
        return []


def suggest_outlier_filter(df: pd.DataFrame, col: str, iqr_k: float = 1.5) -> list[dict]:
    """Return filter hints to remove extreme outliers from a numeric column."""
    if col not in df.columns:
        return []
    try:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:
            return []
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr     = q3 - q1
        if iqr == 0:
            return []
        lo_mask  = s < q1 - iqr_k * iqr
        hi_mask  = s > q3 + iqr_k * iqr
        n_out    = int(lo_mask.sum() + hi_mask.sum())
        rate     = n_out / len(s)
        if rate < 0.04:
            return []
        hints: list[dict] = []
        if lo_mask.any():
            lower = round(float(q1 - iqr_k * iqr), 2)
            hints.append({"type": "filter", "column": col, "operator": "≥",
                          "value": str(lower),
                          "label": f"Remove low outliers in {col}  (< {lower:,.1f})"})
        if hi_mask.any():
            upper = round(float(q3 + iqr_k * iqr), 2)
            hints.append({"type": "filter", "column": col, "operator": "≤",
                          "value": str(upper),
                          "label": f"Remove high outliers in {col}  (> {upper:,.1f})"})
        return hints
    except Exception:
        return []


def suggest_top_n(
    df: pd.DataFrame, cat_col: str, num_col: str, n: int = 10, agg: str = "sum"
) -> list[dict]:
    """
    Suggest a TopN transform to reduce a high-cardinality categorical column to
    the N most significant categories.  Uses the proper top_n transform type
    (not a row filter) so every row for a top-N category is kept intact.
    """
    if cat_col not in df.columns or num_col not in df.columns:
        return []
    try:
        n_cats = int(df[cat_col].nunique())
        if n_cats <= n:
            return []
        return [
            {
                "type":       "top_n",
                "cat_column": cat_col,
                "num_column": num_col,
                "n":          n,
                "agg":        agg,
                "label":      f"Keep top {n} {cat_col} by {agg}({num_col})  ({n_cats} → {n})",
            }
        ]
    except Exception:
        return []


def suggest_ratio_derived(
    df: pd.DataFrame, numerator: str, denominator: str
) -> list[dict]:
    """Suggest a derived ratio column when meaningful and denominator is non-zero."""
    if numerator not in df.columns or denominator not in df.columns:
        return []
    try:
        denom = pd.to_numeric(df[denominator], errors="coerce")
        numer = pd.to_numeric(df[numerator],   errors="coerce")
        if (denom == 0).any() or denom.isna().all() or numer.isna().all():
            return []
        ratio_mean = float((numer / denom).mean())
        if not (0 < abs(ratio_mean) < 200):
            return []
        safe_num = numerator.replace(" ", "_").replace("/", "_per_")
        safe_den = denominator.replace(" ", "_").replace("/", "_per_")
        col_name   = f"{safe_num}_per_{safe_den}"
        expression = f"`{numerator}` / `{denominator}`"
        return [
            {
                "type":       "derived",
                "name":       col_name,
                "expression": expression,
                "label":      f"{numerator} ÷ {denominator}",
            }
        ]
    except Exception:
        return []


def suggest_null_filter(df: pd.DataFrame, col: str) -> list[dict]:
    """Suggest filtering nulls when >5% of values are missing in a key column."""
    if col not in df.columns:
        return []
    try:
        null_rate = float(df[col].isna().mean())
        if null_rate < 0.05:
            return []
        return [
            {
                "type":     "filter",
                "column":   col,
                "operator": "is not blank",
                "value":    "",
                "label":    f"Remove {null_rate*100:.0f}% blank rows in {col}",
            }
        ]
    except Exception:
        return []


def suggest_derived(name: str, expression: str, label: str) -> dict:
    return {"type": "derived", "name": name, "expression": expression, "label": label}


# ── Query-based filtering for non-AI engines ──────────────────────────────────

# Maps user-typed terms to registered chart names.
_QUERY_CHART_ALIASES: dict[str, str] = {
    "heatmap":                "Heatmap",
    "heat map":               "Heatmap",
    "heat":                   "Heatmap",
    "scatter":                "Scatter Plot",
    "scatter plot":           "Scatter Plot",
    "bubble":                 "Scatter Plot",
    "bubble chart":           "Scatter Plot",
    "bar":                    "Bar Chart",
    "bar chart":              "Bar Chart",
    "column chart":           "Bar Chart",
    "line":                   "Line Chart",
    "line chart":             "Line Chart",
    "trend":                  "Line Chart",
    "pie":                    "Pie / Donut",
    "donut":                  "Pie / Donut",
    "doughnut":               "Pie / Donut",
    "radar":                  "Radar Chart",
    "spider":                 "Radar Chart",
    "radar chart":            "Radar Chart",
    "spider chart":           "Radar Chart",
    "bump":                   "Bump Chart",
    "bump chart":             "Bump Chart",
    "ranking":                "Bump Chart",
    "rank":                   "Bump Chart",
    "parallel":               "Parallel Coordinates",
    "parallel coordinates":   "Parallel Coordinates",
    "parcoords":              "Parallel Coordinates",
    "slope":                  "Slope Chart",
    "slope chart":            "Slope Chart",
    "before after":           "Slope Chart",
    "sunburst":               "Sunburst",
    "sun burst":              "Sunburst",
    "treemap":                "Treemap",
    "tree map":               "Treemap",
    "tree":                   "Treemap",
    "funnel":                 "Funnel Chart",
    "funnel chart":           "Funnel Chart",
    "waterfall":              "Waterfall",
    "waterfall chart":        "Waterfall",
    "area":                   "Area Chart",
    "area chart":             "Area Chart",
    "sankey":                 "Sankey Diagram",
    "sankey diagram":         "Sankey Diagram",
    "flow":                   "Sankey Diagram",
    "alluvial":               "Sankey Diagram",
    "distribution":           "Distribution",
    "histogram":              "Distribution",
    "tornado":                "Tornado Chart",
    "butterfly":              "Tornado Chart",
    "tornado chart":          "Tornado Chart",
    "kpi":                    "KPI Tiles",
    "kpi tiles":              "KPI Tiles",
    "metrics":                "KPI Tiles",
    "bullet":                 "Bullet Chart",
    "bullet chart":           "Bullet Chart",
    "gauge":                  "Bullet Chart",
    # New charts
    "box plot":               "Box Plot",
    "boxplot":                "Box Plot",
    "box":                    "Box Plot",
    "whisker":                "Box Plot",
    "box whisker":            "Box Plot",
    "violin":                 "Violin Plot",
    "violin plot":            "Violin Plot",
    "violin chart":           "Violin Plot",
    "strip":                  "Strip Plot",
    "strip plot":             "Strip Plot",
    "jitter":                 "Strip Plot",
    "jitter plot":            "Strip Plot",
    "dot plot":               "Strip Plot",
    "calendar":               "Calendar Heatmap",
    "calendar heatmap":       "Calendar Heatmap",
    "github chart":           "Calendar Heatmap",
    "activity":               "Calendar Heatmap",
    "forecast":               "Forecast",
    "predict":                "Forecast",
    "prediction":             "Forecast",
    "project":                "Forecast",
    "extrapolate":            "Forecast",
    "network":                "Network Graph",
    "network graph":          "Network Graph",
    "graph":                  "Network Graph",
    "node":                   "Network Graph",
    "force directed":         "Network Graph",
    "chord":                  "Chord Diagram",
    "chord diagram":          "Chord Diagram",
    "circular flow":          "Chord Diagram",
    "correlation":            "Correlation Matrix",
    "correlation matrix":     "Correlation Matrix",
    "corr":                   "Correlation Matrix",
    "correlogram":            "Correlation Matrix",
    "splom":                  "Scatter Matrix",
    "scatter matrix":         "Scatter Matrix",
    "pairplot":               "Scatter Matrix",
    "pair plot":              "Scatter Matrix",
    "pairs plot":             "Scatter Matrix",
    "density":                "Density Heatmap",
    "density heatmap":        "Density Heatmap",
    "hexbin":                 "Density Heatmap",
    "2d histogram":           "Density Heatmap",
    "2d heatmap":             "Density Heatmap",
}


def detect_chart_from_query(query: str) -> str | None:
    """Return the registered chart name most mentioned in the query, or None."""
    q = query.lower()
    # Prefer longer matches (more specific) over shorter ones
    for alias in sorted(_QUERY_CHART_ALIASES, key=len, reverse=True):
        if alias in q:
            return _QUERY_CHART_ALIASES[alias]
    return None


def filter_by_query(
    results: list,   # list[SuggestionResult]
    query: str,
) -> list:
    """
    If query specifies a chart type, keep only suggestions of that type.
    If nothing matches, return the original list unchanged so the user
    still gets *some* suggestions.
    """
    if not query.strip():
        return results
    target = detect_chart_from_query(query)
    if target is None:
        return results
    filtered = [r for r in results if r.chart_name == target]
    return filtered if filtered else results
