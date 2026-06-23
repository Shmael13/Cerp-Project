"""
Sunburst Chart — hierarchical proportions in a radial layout.
Up to 3 levels of hierarchy; innermost ring is level1.
"""
from typing import Any, ClassVar

import pandas as pd
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class SunburstChart(BaseVisualization):
    name: ClassVar[str] = "Sunburst"
    description: ClassVar[str] = (
        "Show hierarchical proportions in a radial layout. "
        "Define up to 3 nesting levels; the innermost ring is level 1."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("level1", "categorical", "Level 1 — innermost ring"),
            ColumnSpec("level2", "categorical", "Level 2 (optional)", required=False),
            ColumnSpec("level3", "categorical", "Level 3 — outermost ring (optional)", required=False),
            ColumnSpec("value",  "numeric",     "Value (segment size)"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation", "selectbox", "Aggregation", "sum",
                           {"choices": ["sum", "mean", "count", "max"]}, category="Data"),
            AssumptionSpec("color_scheme", "selectbox", "Color scheme", "Plotly",
                           {"choices": ["Plotly", "Pastel", "Dark2", "Set3", "Paired"]}, category="Display"),
            AssumptionSpec("max_depth",    "number_input", "Max depth (0 = show all levels)", 0,
                           {"min": 0, "max": 3, "step": 1}, category="Display"),
            AssumptionSpec("branchvalues", "selectbox", "Branch value mode", "total",
                           {"choices": ["total", "remainder"]}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        level1_col = columns["level1"]
        level2_col = columns.get("level2")
        level3_col = columns.get("level3")
        value_col  = columns["value"]
        warnings: list[str] = []

        path = [c for c in [level1_col, level2_col, level3_col] if c]

        work = df[path + [value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col])

        agg = params["aggregation"]
        if agg == "count":
            work["__count__"] = 1
            agg_col = "__count__"
        else:
            agg_col = value_col

        # Clean nulls in path columns (sunburst errors on NaN path values)
        for c in path:
            work[c] = work[c].fillna("(blank)").astype(str)

        color_map = {
            "Plotly": px.colors.qualitative.Plotly,
            "Pastel": px.colors.qualitative.Pastel,
            "Dark2":  px.colors.qualitative.Dark2,
            "Set3":   px.colors.qualitative.Set3,
            "Paired": px.colors.qualitative.Pastel2,
        }

        max_depth = int(params["max_depth"]) or None

        try:
            fig = px.sunburst(
                work,
                path=path,
                values=agg_col,
                color_discrete_sequence=color_map.get(params["color_scheme"], px.colors.qualitative.Plotly),
                branchvalues=params["branchvalues"],
                maxdepth=max_depth,
            )
        except Exception as exc:
            raise ValueError(f"Sunburst build failed: {exc}") from exc

        fig.update_layout(template="plotly_white")
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Value: %{value:,.2f}<br>Share: %{percentParent:.1%}<extra></extra>"
        )

        if agg == "count":
            warnings.append("Value shows count of rows (value column ignored for count aggregation).")

        return BuildResult(figure=fig, warnings=warnings)


registry.register(SunburstChart)
