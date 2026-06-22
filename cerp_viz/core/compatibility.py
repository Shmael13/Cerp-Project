from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cerp_viz.core.models import ColumnSpec
from cerp_viz.core.registry import registry


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
        "datetime":    len(df.select_dtypes(include=["datetime", "datetimetz"]).columns),
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
