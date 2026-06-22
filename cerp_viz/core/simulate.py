"""
Data simulation: apply numeric rules to a DataFrame's columns to model
what-if scenarios (scale, shift, replace, clip).

Designed to be UI-agnostic — this module knows nothing about HTTP or sessions.
New operations can be added by extending _HANDLERS without touching callers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Operation = Literal["scale", "add", "replace", "clip_min", "clip_max", "round"]

_OPERATIONS: dict[str, str] = {
    "scale":    "Multiply all values by a factor  (e.g. 1.1 = +10%)",
    "add":      "Add a constant to all values      (e.g. -5 = subtract 5)",
    "replace":  "Set every non-null value to a constant",
    "clip_min": "Floor values at a minimum threshold",
    "clip_max": "Cap values at a maximum threshold",
    "round":    "Round to N decimal places          (e.g. 0 = integers)",
}


def available_operations() -> list[dict[str, str]]:
    """Return operation descriptors for the frontend."""
    return [{"id": k, "label": v} for k, v in _OPERATIONS.items()]


@dataclass
class SimRule:
    """One numeric transformation applied to a single column."""
    column:    str
    operation: str       # one of _OPERATIONS keys
    value:     float


@dataclass
class SimScenario:
    """Named set of simulation rules forming one what-if scenario."""
    name:  str
    rules: list[SimRule]


def apply_rules(df: pd.DataFrame, rules: list[SimRule]) -> pd.DataFrame:
    """Return a new DataFrame with all rules applied in order.

    Rules on missing or non-numeric columns are silently skipped so a
    partial configuration never crashes the build.
    """
    work = df.copy()
    for rule in rules:
        if rule.column not in work.columns:
            continue
        col = pd.to_numeric(work[rule.column], errors="coerce")
        if col.isna().all():
            continue  # column isn't numeric — skip

        op, v = rule.operation, rule.value
        if   op == "scale":    col = col * v
        elif op == "add":      col = col + v
        elif op == "replace":  col = col.where(col.isna(), v)
        elif op == "clip_min": col = col.clip(lower=v)
        elif op == "clip_max": col = col.clip(upper=v)
        elif op == "round":    col = col.round(int(v))
        # unknown operation: leave column unchanged

        work[rule.column] = col
    return work
