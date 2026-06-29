"""
Data simulation: apply numeric rules to a DataFrame's columns to model
what-if scenarios (scale, shift, replace, clip).

Each rule may carry an optional row-level condition so the transformation
is applied only to rows where that condition holds.

Designed to be UI-agnostic — this module knows nothing about HTTP or sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

Operation = str  # "scale" | "add" | "replace" | "clip_min" | "clip_max" | "round"

_OPERATIONS: dict[str, str] = {
    "scale":    "Multiply all values by a factor  (e.g. 1.1 = +10%)",
    "add":      "Add a constant to all values      (e.g. -5 = subtract 5)",
    "replace":  "Set every non-null value to a constant",
    "clip_min": "Floor values at a minimum threshold",
    "clip_max": "Cap values at a maximum threshold",
    "round":    "Round to N decimal places          (e.g. 0 = integers)",
}

CONDITION_OPS = ["==", "!=", ">", ">=", "<", "<="]


def available_operations() -> list[dict[str, str]]:
    return [{"id": k, "label": v} for k, v in _OPERATIONS.items()]


@dataclass
class SimRule:
    column:        str
    operation:     str
    value:         float
    condition_col: str | None  = None
    condition_op:  str | None  = None   # one of CONDITION_OPS
    condition_val: str | None  = None   # stored as string; coerced at eval time


@dataclass
class SimScenario:
    name:  str
    rules: list[SimRule]


def _eval_condition(
    df: pd.DataFrame,
    col: str,
    op: str,
    val_str: str,
) -> "pd.Series[bool]":
    """Return a boolean mask for rows where the condition holds."""
    series = df[col]
    try:
        val: Any = float(val_str)
        numeric = True
    except (ValueError, TypeError):
        val = str(val_str)
        numeric = False

    if numeric:
        s = pd.to_numeric(series, errors="coerce")
        if   op == "==": return s == val
        if   op == "!=": return s != val
        if   op == ">":  return s > val
        if   op == ">=": return s >= val
        if   op == "<":  return s < val
        if   op == "<=": return s <= val
    else:
        s = series.astype(str)
        if   op == "==": return s == val
        if   op == "!=": return s != val
    return pd.Series(True, index=df.index)


def apply_rules(df: pd.DataFrame, rules: list[SimRule]) -> pd.DataFrame:
    """Return a new DataFrame with all rules applied in order.

    Rules on missing or non-numeric columns are silently skipped.
    When a rule has a condition, only matching rows are modified.
    """
    work = df.copy()
    for rule in rules:
        if rule.column not in work.columns:
            continue
        col = pd.to_numeric(work[rule.column], errors="coerce")
        if col.isna().all():
            continue

        has_cond = (
            rule.condition_col
            and rule.condition_col in work.columns
            and rule.condition_op in CONDITION_OPS
            and rule.condition_val is not None
            and str(rule.condition_val).strip() != ""
        )
        if has_cond:
            mask = _eval_condition(
                work,
                rule.condition_col,   # type: ignore[arg-type]
                rule.condition_op,    # type: ignore[arg-type]
                rule.condition_val,   # type: ignore[arg-type]
            )
        else:
            mask = pd.Series(True, index=work.index)

        op, v = rule.operation, rule.value
        new_col = col.copy()
        if   op == "scale":    new_col = col * v
        elif op == "add":      new_col = col + v
        elif op == "replace":  new_col = col.where(col.isna(), v)
        elif op == "clip_min": new_col = col.clip(lower=v)
        elif op == "clip_max": new_col = col.clip(upper=v)
        elif op == "round":    new_col = col.round(int(v))
        else:
            continue

        work[rule.column] = new_col.where(mask, col)
    return work
