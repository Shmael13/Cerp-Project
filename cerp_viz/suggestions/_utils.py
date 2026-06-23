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
    """Return date-part hints when col looks like a high-cardinality datetime."""
    if col not in df.columns:
        return []
    try:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() < 3:
            return []
        n_unique = parsed.dt.year.nunique() if hasattr(parsed.dt, "year") else 0
        parts = []
        if n_unique > 1:
            parts.append({"type": "date_part", "source_column": col, "part": "Year",
                          "label": f"Extract Year from {col}"})
        parts.append({"type": "date_part", "source_column": col, "part": "Month",
                      "label": f"Extract Month from {col}"})
        parts.append({"type": "date_part", "source_column": col, "part": "Quarter",
                      "label": f"Extract Quarter from {col}"})
        return parts
    except Exception:
        return []


def suggest_outlier_filter(df: pd.DataFrame, col: str, iqr_k: float = 1.5) -> list[dict]:
    """Return a filter hint to remove extreme outliers from col."""
    if col not in df.columns:
        return []
    try:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - iqr_k * iqr) | (s > q3 + iqr_k * iqr)).sum()
        rate = outliers / len(s)
        if rate > 0.04:   # >4% outliers — worth flagging
            lower = round(float(q1 - iqr_k * iqr), 2)
            upper = round(float(q3 + iqr_k * iqr), 2)
            return [
                {"type": "filter", "column": col, "operator": "≥", "value": str(lower),
                 "label": f"Remove low outliers in {col} (below {lower:,.1f})"},
                {"type": "filter", "column": col, "operator": "≤", "value": str(upper),
                 "label": f"Remove high outliers in {col} (above {upper:,.1f})"},
            ]
        return []
    except Exception:
        return []


def suggest_derived(name: str, expression: str, label: str) -> dict:
    return {"type": "derived", "name": name, "expression": expression, "label": label}


def suggest_top_n_filter(
    df: pd.DataFrame, cat_col: str, num_col: str, n: int = 10
) -> list[dict]:
    """
    When a categorical column has many distinct values, suggest keeping only the
    top-N by sum of num_col.  Returns [] when cardinality is already manageable.
    """
    if cat_col not in df.columns or num_col not in df.columns:
        return []
    try:
        n_cats = df[cat_col].nunique()
        if n_cats <= n:
            return []
        top = (
            df.groupby(cat_col)[num_col]
            .sum()
            .nlargest(n)
            .index
            .tolist()
        )
        threshold = (
            df.groupby(cat_col)[num_col].sum().nlargest(n).min()
        )
        return [
            {
                "type": "filter",
                "column": num_col,
                "operator": "≥",
                "value": str(round(float(threshold), 2)),
                "label": f"Keep top {n} {cat_col} by {num_col} ({n_cats} → {n} categories)",
            }
        ]
    except Exception:
        return []


def suggest_ratio_derived(
    df: pd.DataFrame, numerator: str, denominator: str
) -> list[dict]:
    """
    Suggest a derived ratio column when both columns are numeric and the
    denominator is never zero.  Returns [] when the ratio is not meaningful.
    """
    if numerator not in df.columns or denominator not in df.columns:
        return []
    try:
        denom = pd.to_numeric(df[denominator], errors="coerce")
        if (denom == 0).any() or denom.isna().all():
            return []
        numer = pd.to_numeric(df[numerator], errors="coerce")
        ratio_mean = float((numer / denom).mean())
        # Only useful when the ratio is in a plausible 0–1 or 0–100 range
        if not (0 < abs(ratio_mean) < 200):
            return []
        col_name   = f"{numerator}_per_{denominator}"
        expression = f"{numerator} / {denominator}"
        return [
            {
                "type": "derived",
                "name": col_name,
                "expression": expression,
                "label": f"Compute {numerator} ÷ {denominator} ratio",
            }
        ]
    except Exception:
        return []


def suggest_null_filter(df: pd.DataFrame, col: str) -> list[dict]:
    """Suggest filtering nulls when >5% of values are missing in key column."""
    if col not in df.columns:
        return []
    try:
        null_rate = df[col].isna().mean()
        if null_rate < 0.05:
            return []
        return [
            {
                "type": "filter",
                "column": col,
                "operator": "is not blank",
                "value": "",
                "label": f"Remove {null_rate*100:.0f}% blank rows in {col}",
            }
        ]
    except Exception:
        return []
