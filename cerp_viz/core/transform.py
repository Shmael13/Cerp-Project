"""
Data transformation layer — pure Python / pandas, no HTTP concerns.

Apply order: derived columns → top-N filters → date parts → row filters.
Derived columns first so downstream transforms can reference them.
Top-N before date parts so cardinality reduction happens on raw values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ── Operator table ─────────────────────────────────────────────────────────────

def _coerce(col: pd.Series, val: str) -> Any:
    """Coerce val to match the dtype of col so equality comparisons work on numerics."""
    if pd.api.types.is_numeric_dtype(col):
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return val


_OPERATORS: dict[str, Any] = {
    "=":           lambda col, val: col == _coerce(col, val),
    "≠":           lambda col, val: col != _coerce(col, val),
    ">":           lambda col, val: pd.to_numeric(col, errors="coerce") > float(val),
    "<":           lambda col, val: pd.to_numeric(col, errors="coerce") < float(val),
    "≥":           lambda col, val: pd.to_numeric(col, errors="coerce") >= float(val),
    "≤":           lambda col, val: pd.to_numeric(col, errors="coerce") <= float(val),
    "contains":    lambda col, val: col.astype(str).str.contains(str(val), case=False, na=False),
    "not contains":lambda col, val: ~col.astype(str).str.contains(str(val), case=False, na=False),
    "is blank":    lambda col, _:   col.isna() | (col.astype(str).str.strip() == ""),
    "is not blank":lambda col, _:   col.notna() & (col.astype(str).str.strip() != ""),
    "starts with": lambda col, val: col.astype(str).str.startswith(str(val), na=False),
}

# ── Date-part table ────────────────────────────────────────────────────────────

_DATE_PARTS: dict[str, Any] = {
    "Year":       lambda s: s.dt.year,
    "Quarter":    lambda s: s.dt.quarter.map(lambda q: f"Q{q}"),
    "Month":      lambda s: s.dt.month_name(),
    "Month Num":  lambda s: s.dt.month,
    "Week Num":   lambda s: s.dt.isocalendar().week.astype(int),
    "Day of Month": lambda s: s.dt.day,
    "Day of Week":lambda s: s.dt.day_name(),
    "Hour":       lambda s: s.dt.hour,
}


def _safe_col_name(base: str, part: str) -> str:
    """Build a valid pandas column name — no spaces, no # or special chars."""
    part_clean = part.lower().replace(" ", "_").replace("#", "num")
    return f"{base}_{part_clean}"


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class FilterRule:
    column:   str
    operator: str   # one of _OPERATORS keys
    value:    str   = ""
    enabled:  bool  = True


@dataclass
class TopNRule:
    """Keep only rows belonging to the top-N categories of cat_column by agg(num_column)."""
    cat_column: str
    num_column: str
    n:          int
    agg:        str = "sum"   # sum | mean | max | count
    enabled:    bool = True


@dataclass
class DerivedColumn:
    name:       str
    expression: str   # evaluated via df.eval()
    enabled:    bool  = True


@dataclass
class DatePart:
    source_column: str
    part:          str   # one of _DATE_PARTS keys
    enabled:       bool  = True


@dataclass
class TransformConfig:
    filters:      list[FilterRule]    = field(default_factory=list)
    top_n:        list[TopNRule]      = field(default_factory=list)
    derived_cols: list[DerivedColumn] = field(default_factory=list)
    date_parts:   list[DatePart]      = field(default_factory=list)


# ── Core apply ─────────────────────────────────────────────────────────────────

def apply_transforms(df: pd.DataFrame, cfg: TransformConfig) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply transforms in order:
      1. Derived columns  (so filters can reference them)
      2. Top-N category filters  (reduce cardinality on raw data)
      3. Date-part extractions
      4. Row filters
    Returns (transformed_df, warnings).
    """
    work     = df.copy()
    warnings: list[str] = []

    # 1. Derived columns
    for dc in cfg.derived_cols:
        if not dc.enabled or not dc.name.strip() or not dc.expression.strip():
            continue
        try:
            # Backtick-quote column names containing spaces or special chars
            expr = dc.expression
            work[dc.name] = work.eval(expr)
        except Exception as exc:
            warnings.append(f"Derived column '{dc.name}': {exc}")

    # 2. Top-N category filters
    for rule in cfg.top_n:
        if not rule.enabled:
            continue
        if rule.cat_column not in work.columns or rule.num_column not in work.columns:
            warnings.append(f"Top-N filter: column '{rule.cat_column}' or '{rule.num_column}' not found.")
            continue
        try:
            num_series = pd.to_numeric(work[rule.num_column], errors="coerce")
            tmp        = work.copy()
            tmp["__num__"] = num_series
            agg_map    = {"sum": "sum", "mean": "mean", "max": "max", "count": "count"}
            agg_fn     = agg_map.get(rule.agg, "sum")
            top_cats   = (
                tmp.groupby(rule.cat_column)["__num__"]
                .agg(agg_fn)
                .nlargest(rule.n)
                .index
            )
            before   = len(work)
            work     = work[work[rule.cat_column].isin(top_cats)]
            removed  = before - len(work)
            if removed:
                warnings.append(
                    f"Top-{rule.n} filter on '{rule.cat_column}': kept {len(top_cats)} categories, "
                    f"removed {removed} row(s)."
                )
        except Exception as exc:
            warnings.append(f"Top-N filter on '{rule.cat_column}': {exc}")

    # 3. Date parts
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

    # 4. Row filters
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
