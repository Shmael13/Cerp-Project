"""
Data transformation layer — pure Python / pandas, no HTTP concerns.

Apply order:
  1. derived_cols  — create new columns from multi-column pandas expressions
  2. col_exprs     — per-column math transforms (log, sqrt, x^2 …)
  3. fill_nulls    — impute missing values
  4. rolling       — rolling-window aggregations
  5. bins          — discretise numeric columns
  6. normalize     — scale numeric columns
  7. top_n         — keep only top-N categories
  8. date_parts    — extract date components
  9. filters       — row-level predicates
"""
from __future__ import annotations

import math as _math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Operator table ─────────────────────────────────────────────────────────────

def _coerce(col: pd.Series, val: str) -> Any:
    if pd.api.types.is_numeric_dtype(col):
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return val


_OPERATORS: dict[str, Any] = {
    "=":            lambda col, val: col == _coerce(col, val),
    "≠":            lambda col, val: col != _coerce(col, val),
    ">":            lambda col, val: pd.to_numeric(col, errors="coerce") > float(val),
    "<":            lambda col, val: pd.to_numeric(col, errors="coerce") < float(val),
    "≥":            lambda col, val: pd.to_numeric(col, errors="coerce") >= float(val),
    "≤":            lambda col, val: pd.to_numeric(col, errors="coerce") <= float(val),
    "contains":     lambda col, val: col.astype(str).str.contains(str(val), case=False, na=False),
    "not contains": lambda col, val: ~col.astype(str).str.contains(str(val), case=False, na=False),
    "is blank":     lambda col, _:   col.isna() | (col.astype(str).str.strip() == ""),
    "is not blank": lambda col, _:   col.notna() & (col.astype(str).str.strip() != ""),
    "starts with":  lambda col, val: col.astype(str).str.startswith(str(val), na=False),
    "ends with":    lambda col, val: col.astype(str).str.endswith(str(val), na=False),
    "in list":      lambda col, val: col.astype(str).isin([v.strip() for v in str(val).split(",")]),
}

# ── Date-part table ────────────────────────────────────────────────────────────

_DATE_PARTS: dict[str, Any] = {
    "Year":         lambda s: s.dt.year,
    "Quarter":      lambda s: s.dt.quarter.map(lambda q: f"Q{q}"),
    "Month":        lambda s: s.dt.month_name(),
    "Month Num":    lambda s: s.dt.month,
    "Week Num":     lambda s: s.dt.isocalendar().week.astype(int),
    "Day of Month": lambda s: s.dt.day,
    "Day of Week":  lambda s: s.dt.day_name(),
    "Hour":         lambda s: s.dt.hour,
}


def _safe_col_name(base: str, part: str) -> str:
    part_clean = part.lower().replace(" ", "_").replace("#", "num")
    return f"{base}_{part_clean}"


# ── Safe math namespace for col_expr ──────────────────────────────────────────

_SAFE_NS: dict[str, Any] = {
    "__builtins__": {},
    # numpy ufuncs — work on arrays and scalars
    "log":   np.log,   "log2":  np.log2,  "log10": np.log10,
    "sqrt":  np.sqrt,  "abs":   np.abs,   "exp":   np.exp,
    "sin":   np.sin,   "cos":   np.cos,   "tan":   np.tan,
    "floor": np.floor, "ceil":  np.ceil,  "round": np.round,
    "clip":  np.clip,  "sign":  np.sign,
    # constants
    "pi":    _math.pi, "e":     _math.e,
    "nan":   float("nan"), "inf": float("inf"),
}


def _eval_col_expr(series: pd.Series, expr: str) -> pd.Series:
    """Evaluate a math expression using 'x' as the column placeholder.
    Returns a float Series; NaN on any per-element error."""
    x = pd.to_numeric(series, errors="coerce").values  # numpy array
    ns = {**_SAFE_NS, "x": x}
    result = eval(compile(expr, "<col_expr>", "eval"), ns)  # noqa: S307 — restricted namespace
    return pd.Series(result, index=series.index, dtype=float)


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class FilterRule:
    column:   str
    operator: str
    value:    str  = ""
    enabled:  bool = True


@dataclass
class TopNRule:
    """Keep only rows belonging to the top-N categories of cat_column by agg(num_column)."""
    cat_column: str
    num_column: str
    n:          int
    agg:        str  = "sum"
    enabled:    bool = True


@dataclass
class DerivedColumn:
    """Create a new column from a multi-column pandas expression (df.eval)."""
    name:       str
    expression: str
    enabled:    bool = True


@dataclass
class DatePart:
    source_column: str
    part:          str
    enabled:       bool = True


@dataclass
class ColTransformRule:
    """Apply a math expression to an existing column using 'x' as the value.
    Examples: log(x), sqrt(abs(x)), x/1000, x**2, clip(x, 0, 100)
    If new_name is empty, the source column is overwritten in-place."""
    column:     str
    expression: str
    new_name:   str  = ""
    enabled:    bool = True


@dataclass
class BinRule:
    """Discretise a numeric column into N labelled buckets."""
    column:   str
    n_bins:   int  = 10
    strategy: str  = "equal_width"   # "equal_width" | "quantile"
    new_name: str  = ""              # defaults to column + "_bin"
    enabled:  bool = True


@dataclass
class NormalizeRule:
    """Scale a numeric column to a standard range."""
    column:   str
    method:   str  = "min_max"   # "min_max" | "z_score" | "pct_of_total"
    new_name: str  = ""          # defaults to column + "_norm"
    enabled:  bool = True


@dataclass
class FillNullRule:
    """Impute missing values in a column."""
    column:  str
    method:  str  = "mean"   # "mean" | "median" | "zero" | "ffill" | "bfill" | "value"
    value:   str  = ""       # used when method == "value"
    enabled: bool = True


@dataclass
class RollingRule:
    """Apply a rolling-window aggregation (sorts by sort_col first if provided)."""
    column:   str
    window:   int  = 3
    agg:      str  = "mean"   # "mean" | "sum" | "max" | "min" | "std"
    sort_col: str  = ""       # column to sort by before rolling
    new_name: str  = ""       # defaults to column + "_rolling_{window}"
    enabled:  bool = True


@dataclass
class TransformConfig:
    filters:      list[FilterRule]       = field(default_factory=list)
    top_n:        list[TopNRule]         = field(default_factory=list)
    derived_cols: list[DerivedColumn]    = field(default_factory=list)
    date_parts:   list[DatePart]         = field(default_factory=list)
    col_exprs:    list[ColTransformRule] = field(default_factory=list)
    bins:         list[BinRule]          = field(default_factory=list)
    normalize:    list[NormalizeRule]    = field(default_factory=list)
    fill_nulls:   list[FillNullRule]     = field(default_factory=list)
    rolling:      list[RollingRule]      = field(default_factory=list)


# ── Core apply ─────────────────────────────────────────────────────────────────

def apply_transforms(df: pd.DataFrame, cfg: TransformConfig) -> tuple[pd.DataFrame, list[str]]:
    work     = df.copy()
    warnings: list[str] = []

    # 1. Derived columns (multi-column pandas eval)
    for dc in cfg.derived_cols:
        if not dc.enabled or not dc.name.strip() or not dc.expression.strip():
            continue
        try:
            work[dc.name] = work.eval(dc.expression)
        except Exception as exc:
            warnings.append(f"Derived column '{dc.name}': {exc}")

    # 2. Column expression transforms (single-column math, Desmos-style)
    for ct in cfg.col_exprs:
        if not ct.enabled or not ct.column.strip() or not ct.expression.strip():
            continue
        if ct.column not in work.columns:
            warnings.append(f"Col expr: column '{ct.column}' not found.")
            continue
        try:
            result = _eval_col_expr(work[ct.column], ct.expression)
            dest   = ct.new_name.strip() if ct.new_name.strip() else ct.column
            work[dest] = result
        except Exception as exc:
            warnings.append(f"Col expr on '{ct.column}' ({ct.expression!r}): {exc}")

    # 3. Fill nulls
    for fn in cfg.fill_nulls:
        if not fn.enabled or fn.column not in work.columns:
            continue
        try:
            col    = work[fn.column]
            n_null = int(col.isna().sum())
            if n_null == 0:
                continue
            m = fn.method
            if m == "mean":
                fill = pd.to_numeric(col, errors="coerce").mean()
                work[fn.column] = col.fillna(fill)
            elif m == "median":
                fill = pd.to_numeric(col, errors="coerce").median()
                work[fn.column] = col.fillna(fill)
            elif m == "zero":
                work[fn.column] = col.fillna(0)
            elif m == "ffill":
                work[fn.column] = col.ffill()
            elif m == "bfill":
                work[fn.column] = col.bfill()
            elif m == "value":
                work[fn.column] = col.fillna(fn.value)
            warnings.append(f"Fill nulls in '{fn.column}' ({m}): filled {n_null} missing value(s).")
        except Exception as exc:
            warnings.append(f"Fill nulls on '{fn.column}': {exc}")

    # 4. Rolling window
    for rr in cfg.rolling:
        if not rr.enabled or rr.column not in work.columns:
            continue
        try:
            tmp = work.copy()
            if rr.sort_col and rr.sort_col in tmp.columns:
                tmp = tmp.sort_values(rr.sort_col)
            series = pd.to_numeric(tmp[rr.column], errors="coerce")
            rolled = getattr(series.rolling(rr.window, min_periods=1), rr.agg)()
            dest   = rr.new_name.strip() if rr.new_name.strip() else f"{rr.column}_rolling{rr.window}"
            work[dest] = rolled.values  # re-align by position after sort
            warnings.append(
                f"Rolling {rr.window}-period {rr.agg} on '{rr.column}' → '{dest}'."
            )
        except Exception as exc:
            warnings.append(f"Rolling on '{rr.column}': {exc}")

    # 5. Bin numeric columns
    for br in cfg.bins:
        if not br.enabled or br.column not in work.columns:
            continue
        try:
            series  = pd.to_numeric(work[br.column], errors="coerce")
            dest    = br.new_name.strip() if br.new_name.strip() else f"{br.column}_bin"
            if br.strategy == "quantile":
                work[dest] = pd.qcut(series, br.n_bins, labels=False, duplicates="drop").astype("Int64")
            else:
                work[dest] = pd.cut(series, br.n_bins, labels=False).astype("Int64")
            warnings.append(f"Binned '{br.column}' into {br.n_bins} {br.strategy} bucket(s) → '{dest}'.")
        except Exception as exc:
            warnings.append(f"Bin on '{br.column}': {exc}")

    # 6. Normalize
    for nr in cfg.normalize:
        if not nr.enabled or nr.column not in work.columns:
            continue
        try:
            series = pd.to_numeric(work[nr.column], errors="coerce")
            dest   = nr.new_name.strip() if nr.new_name.strip() else f"{nr.column}_norm"
            m = nr.method
            if m == "min_max":
                lo, hi = series.min(), series.max()
                work[dest] = (series - lo) / (hi - lo) if hi != lo else series * 0
            elif m == "z_score":
                mu, sigma = series.mean(), series.std()
                work[dest] = (series - mu) / sigma if sigma != 0 else series * 0
            elif m == "pct_of_total":
                total = series.sum()
                work[dest] = series / total * 100 if total != 0 else series * 0
            warnings.append(f"Normalized '{nr.column}' ({m}) → '{dest}'.")
        except Exception as exc:
            warnings.append(f"Normalize on '{nr.column}': {exc}")

    # 7. Top-N category filters
    for rule in cfg.top_n:
        if not rule.enabled:
            continue
        if rule.cat_column not in work.columns or rule.num_column not in work.columns:
            warnings.append(f"Top-N: column '{rule.cat_column}' or '{rule.num_column}' not found.")
            continue
        try:
            num_series = pd.to_numeric(work[rule.num_column], errors="coerce")
            tmp        = work.copy()
            tmp["__num__"] = num_series
            top_cats   = (
                tmp.groupby(rule.cat_column)["__num__"]
                .agg(rule.agg)
                .nlargest(rule.n)
                .index
            )
            before = len(work)
            work   = work[work[rule.cat_column].isin(top_cats)]
            removed = before - len(work)
            if removed:
                warnings.append(
                    f"Top-{rule.n} on '{rule.cat_column}': kept {len(top_cats)} categories, "
                    f"removed {removed} row(s)."
                )
        except Exception as exc:
            warnings.append(f"Top-N on '{rule.cat_column}': {exc}")

    # 8. Date parts
    for dp in cfg.date_parts:
        if not dp.enabled or dp.source_column not in work.columns:
            continue
        try:
            parsed = pd.to_datetime(work[dp.source_column], errors="coerce")
            fn     = _DATE_PARTS.get(dp.part)
            if fn is None:
                warnings.append(f"Unknown date part '{dp.part}'.")
                continue
            new_col       = _safe_col_name(dp.source_column, dp.part)
            work[new_col] = fn(parsed)
        except Exception as exc:
            warnings.append(f"Date part '{dp.part}' on '{dp.source_column}': {exc}")

    # 9. Row filters
    for f in cfg.filters:
        if not f.enabled or not f.column or f.column not in work.columns:
            continue
        op = _OPERATORS.get(f.operator)
        if op is None:
            warnings.append(f"Unknown operator '{f.operator}'.")
            continue
        try:
            before  = len(work)
            mask    = op(work[f.column], f.value)
            work    = work[mask]
            removed = before - len(work)
            if removed:
                warnings.append(
                    f"Filter '{f.column} {f.operator} {f.value!r}' removed {removed} row(s)."
                )
        except Exception as exc:
            warnings.append(f"Filter on '{f.column}': {exc}")

    return work, warnings


def available_operators() -> list[str]:
    return list(_OPERATORS.keys())


def available_date_parts() -> list[str]:
    return list(_DATE_PARTS.keys())
