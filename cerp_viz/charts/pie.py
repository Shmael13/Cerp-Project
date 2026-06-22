from typing import Any, ClassVar

import pandas as pd
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_COLOR_SEQUENCES = {
    "Plotly":  px.colors.qualitative.Plotly,
    "Pastel":  px.colors.qualitative.Pastel,
    "Dark":    px.colors.qualitative.Dark24,
    "Bold":    px.colors.qualitative.Bold,
    "Vivid":   px.colors.qualitative.Vivid,
    "Antique": px.colors.qualitative.Antique,
}


class PieChart(BaseVisualization):
    name: ClassVar[str] = "Pie / Donut"
    description: ClassVar[str] = "Part-to-whole composition across categories."
    supports_comparison: ClassVar[bool] = False

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("names",  "categorical", "Category (Slices)"),
            ColumnSpec("values", "numeric",     "Value (Size of Slice)"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation", "selectbox",    "Aggregation",             "sum",
                           {"choices": ["sum", "mean", "count", "max", "min"]}, category="Data"),
            AssumptionSpec("top_n",       "number_input", "Show Top N (0 = all)",    0,
                           {"min": 0, "max": 50, "step": 1}, category="Data"),
            AssumptionSpec("hole",        "slider",       "Donut Hole Size (0 = pie)", 0.0,
                           {"min": 0.0, "max": 0.85, "step": 0.05}, category="Display"),
            AssumptionSpec("show_pct",    "toggle",       "Show Percentages",        True,
                           {}, category="Display"),
            AssumptionSpec("show_values", "toggle",       "Show Values",             False,
                           {}, category="Display"),
            AssumptionSpec("pull_largest","toggle",       "Pull Out Largest Slice",  False,
                           {}, category="Display"),
            AssumptionSpec("sort_slices", "toggle",       "Sort by Value",           True,
                           {}, category="Display"),
            AssumptionSpec("color_scheme","selectbox",    "Color Scheme",            "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        names_col  = columns["names"]
        values_col = columns["values"]
        warnings: list[str] = []

        work = df.copy()
        work[values_col] = pd.to_numeric(work[values_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[names_col, values_col])
        if (dropped := before - len(work)):
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        # Aggregate
        work = work.groupby(names_col, as_index=False)[values_col].agg(params["aggregation"])

        # Remove zero/negative slices (meaningless in pie)
        neg = (work[values_col] <= 0).sum()
        if neg:
            work = work[work[values_col] > 0]
            warnings.append(f"Removed {neg} slice(s) with zero or negative values.")

        if params["sort_slices"]:
            work = work.sort_values(values_col, ascending=False)

        top_n = int(params["top_n"])
        if top_n > 0 and len(work) > top_n:
            other_val = work.iloc[top_n:][values_col].sum()
            work = work.iloc[:top_n].copy()
            if other_val > 0:
                other_row = pd.DataFrame({names_col: ["Other"], values_col: [other_val]})
                work = pd.concat([work, other_row], ignore_index=True)
            warnings.append(f"Top {top_n} slices shown; remainder grouped as 'Other'.")

        # Pull largest slice
        pull = [0.0] * len(work)
        if params["pull_largest"] and len(work) > 0:
            largest_idx = work[values_col].idxmax()
            pull[work.index.get_loc(largest_idx)] = 0.15

        text_parts = []
        if params["show_pct"]:
            text_parts.append("percent")
        if params["show_values"]:
            text_parts.append("value")
        textinfo = "+".join(text_parts) if text_parts else "none"

        fig = px.pie(
            work,
            names=names_col,
            values=values_col,
            hole=params["hole"],
            color_discrete_sequence=_COLOR_SEQUENCES[params["color_scheme"]],
            template="plotly_white",
        )
        fig.update_traces(
            pull=pull,
            textinfo=textinfo,
            textposition="outside" if params["show_pct"] else "inside",
        )
        fig.update_layout(showlegend=True)

        if params["hole"] > 0:
            # Add total in centre for donut
            total = work[values_col].sum()
            fig.add_annotation(
                text=f"<b>{total:,.0f}</b><br>Total",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False,
            )

        return BuildResult(figure=fig, warnings=warnings)


registry.register(PieChart)
