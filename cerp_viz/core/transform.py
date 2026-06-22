"""
Data transformation layer — pure Python / pandas, no Streamlit.
Applies row filters, derived columns, and date-part extraction to a DataFrame
before it reaches any chart.  All operations are expressed as plain data
structures so they can be serialised into ChartConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


_OPERATORS = {
    "=":        lambda col, val: col == val,
    "≠":        lambda col, val: col != val,
    ">":        lambda col, val: col > _num(val),
    "<":        lambda col, val: col < _num(val),
    "≥":        lambda col, val: col >= _num(val),
    "≤":        lambda col, val: col <= _num(val),
    "contains": lambda col, val: col.astype(str).str.contains(str(val), case=False, na=False),
    "is blank": lambda col, _:   col.isna() | (col.astype(str).str.strip() == ""),
}

_DATE_PARTS = {
    "Year":    lambda s: s.dt.year,
    "Month #": lambda s: s.dt.month,
    "Month":   lambda s: s.dt.month_name(),
    "Quarter": lambda s: s.dt.quarter.map(lambda q: f"Q{q}"),
    "Week #":  lambda s: s.dt.isocalendar().week.astype(int),
    "Day":     lambda s: s.dt.day,
}


def _num(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


@dataclass
class FilterRule:
    column:   str
    operator: str   # one of _OPERATORS keys
    value:    str = ""
    enabled:  bool = True


@dataclass
class DerivedColumn:
    name:       str
    expression: str   # evaluated via df.eval()
    enabled:    bool = True


@dataclass
class DatePart:
    source_column: str
    part:          str   # one of _DATE_PARTS keys
    enabled:       bool = True


@dataclass
class TransformConfig:
    filters:      list[FilterRule]    = field(default_factory=list)
    derived_cols: list[DerivedColumn] = field(default_factory=list)
    date_parts:   list[DatePart]      = field(default_factory=list)


def apply_transforms(df: pd.DataFrame, cfg: TransformConfig) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply transforms in order: derived columns → date parts → filters.
    Returns (transformed_df, warnings).
    Derived columns first so they can be used in filters.
    """
    work = df.copy()
    warnings: list[str] = []

    # 1. Derived columns
    for dc in cfg.derived_cols:
        if not dc.enabled or not dc.name.strip() or not dc.expression.strip():
            continue
        try:
            work[dc.name] = work.eval(dc.expression)
        except Exception as exc:
            warnings.append(f"Derived column '{dc.name}': {exc}")

    # 2. Date parts
    for dp in cfg.date_parts:
        if not dp.enabled or dp.source_column not in work.columns:
            continue
        try:
            parsed = pd.to_datetime(work[dp.source_column], errors="coerce")
            fn     = _DATE_PARTS.get(dp.part)
            if fn is None:
                warnings.append(f"Unknown date part '{dp.part}'.")
                continue
            new_col         = f"{dp.source_column}_{dp.part.lower().replace(' ', '_')}"
            work[new_col]   = fn(parsed)
        except Exception as exc:
            warnings.append(f"Date part '{dp.part}' on '{dp.source_column}': {exc}")

    # 3. Filters
    for f in cfg.filters:
        if not f.enabled or not f.column or f.column not in work.columns:
            continue
        op = _OPERATORS.get(f.operator)
        if op is None:
            warnings.append(f"Unknown operator '{f.operator}'.")
            continue
        try:
            before = len(work)
            mask   = op(work[f.column], f.value)
            work   = work[mask]
            removed = before - len(work)
            if removed:
                warnings.append(
                    f"Filter '{f.column} {f.operator} {f.value}' removed {removed} row(s)."
                )
        except Exception as exc:
            warnings.append(f"Filter on '{f.column}': {exc}")

    return work, warnings


def available_operators() -> list[str]:
    return list(_OPERATORS.keys())


def available_date_parts() -> list[str]:
    return list(_DATE_PARTS.keys())
