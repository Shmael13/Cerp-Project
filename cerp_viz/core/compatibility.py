from __future__ import annotations

import re as _re
from dataclasses import dataclass

import pandas as pd

from cerp_viz.core.models import ColumnSpec
from cerp_viz.core.registry import registry

_DATE_HINTS = _re.compile(r"date|time|year|month|quarter|week|period|day", _re.I)


def _count_datetime_cols(df: pd.DataFrame) -> int:
    """Count native datetime columns + object columns whose values parse as dates."""
    count = len(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    for col in df.select_dtypes(include="object").columns:
        if _DATE_HINTS.search(col):
            sample = df[col].dropna().head(5)
            if len(sample) == 0:
                continue
            try:
                pd.to_datetime(sample, errors="raise")
                count += 1
            except Exception:
                pass
    return count


@dataclass
class CompatibilityResult:
    compatible: bool
    reason: str = ""


def check(df: pd.DataFrame, column_specs: list[ColumnSpec]) -> CompatibilityResult:
    """
    Counts how many required columns of each dtype a chart needs, then checks
    whether the DataFrame has enough columns of each type to satisfy them.
    """
    available: dict[str, int] = {
        "numeric":     len(df.select_dtypes(include="number").columns),
        "categorical": len(df.select_dtypes(exclude="number").columns),
        "datetime":    _count_datetime_cols(df),
        "any":         len(df.columns),
    }

    needed: dict[str, int] = {}
    for spec in column_specs:
        if spec.required:
            needed[spec.dtype] = needed.get(spec.dtype, 0) + 1

    for dtype, count in needed.items():
        have = available.get(dtype, 0)
        if have < count:
            plural = "s" if count > 1 else ""
            return CompatibilityResult(
                compatible=False,
                reason=f"needs {count} {dtype} column{plural}, found {have}",
            )

    return CompatibilityResult(compatible=True)


def compatible_visualizations(df: pd.DataFrame) -> dict[str, CompatibilityResult]:
    """Returns a result for every registered visualization against the given DataFrame."""
    return {
        name: check(df, registry.get(name)().required_columns())
        for name in registry.names()
    }
