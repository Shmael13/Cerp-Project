"""
Slope Chart — highlight change between two time points across categories.
Left axis = start value, right axis = end value.
"""
from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_COLOR_SEQUENCES = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Dark":   px.colors.qualitative.Dark24,
    "Bold":   px.colors.qualitative.Bold,
    "Vivid":  px.colors.qualitative.Vivid,
}
_POS_COLOR = "#27ae60"
_NEG_COLOR = "#e74c3c"
_NEU_COLOR = "#95a5a6"


class SlopeChart(BaseVisualization):
    name: ClassVar[str] = "Slope Chart"
    description: ClassVar[str] = (
        "Show change between two points for multiple categories. "
        "Requires a start value and end value column (wide format)."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("category",    "categorical", "Category (one line per value)"),
            ColumnSpec("start_value", "numeric",     "Start value (left axis)"),
            ColumnSpec("end_value",   "numeric",     "End value (right axis)"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("start_label",       "text",   "Left axis label",      "Before",
                           {}, category="Display"),
            AssumptionSpec("end_label",         "text",   "Right axis label",     "After",
                           {}, category="Display"),
            AssumptionSpec("color_by_direction","toggle", "Colour lines by direction (↑green ↓red)", True,
                           {}, category="Display"),
            AssumptionSpec("show_values",       "toggle", "Show values at endpoints", True,
                           {}, category="Display"),
            AssumptionSpec("line_width",        "slider", "Line width", 2,
                           {"min": 1, "max": 6, "step": 1}, category="Display"),
            AssumptionSpec("top_n", "number_input", "Show top N by absolute change (0 = all)", 0,
                           {"min": 0, "max": 50, "step": 1}, category="Data"),
            AssumptionSpec("color_scheme", "selectbox", "Color scheme (when not direction-colored)", "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        cat_col   = columns["category"]
        start_col = columns["start_value"]
        end_col   = columns["end_value"]
        warnings: list[str] = []

        work = df[[cat_col, start_col, end_col]].copy()
        work[start_col] = pd.to_numeric(work[start_col], errors="coerce")
        work[end_col]   = pd.to_numeric(work[end_col],   errors="coerce")
        work = work.dropna(subset=[start_col, end_col])

        # Aggregate duplicates by category
        work = work.groupby(cat_col, as_index=False)[[start_col, end_col]].mean()
        work["__change__"] = work[end_col] - work[start_col]

        top_n = int(params["top_n"])
        if top_n > 0:
            work = work.reindex(
                work["__change__"].abs().nlargest(top_n).index
            )
            warnings.append(f"Showing top {top_n} categories by absolute change.")

        start_lbl = str(params["start_label"]) or "Before"
        end_lbl   = str(params["end_label"])   or "After"
        colors    = _COLOR_SEQUENCES[params["color_scheme"]]
        color_dir = bool(params["color_by_direction"])

        fig = go.Figure()

        for idx, row in work.iterrows():
            cat     = row[cat_col]
            sv, ev  = float(row[start_col]), float(row[end_col])
            change  = ev - sv
            if color_dir:
                color = _POS_COLOR if change > 0 else _NEG_COLOR if change < 0 else _NEU_COLOR
            else:
                color = colors[int(idx) % len(colors)]

            # Draw connecting line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[sv, ev],
                mode="lines+markers",
                line=dict(width=params["line_width"], color=color),
                marker=dict(size=8, color=color),
                name=str(cat),
                showlegend=True,
                hovertemplate=f"<b>{cat}</b><br>{start_lbl}: {sv:,.2f}<br>{end_lbl}: {ev:,.2f}<br>Δ: {change:+,.2f}<extra></extra>",
            ))

            if params["show_values"]:
                for x_pos, val in [(0, sv), (1, ev)]:
                    fig.add_annotation(
                        x=x_pos + (-0.04 if x_pos == 0 else 0.04),
                        y=val,
                        text=f"{val:,.1f}",
                        showarrow=False,
                        font=dict(size=9, color=color),
                        xanchor="right" if x_pos == 0 else "left",
                    )

        fig.update_xaxes(
            tickvals=[0, 1],
            ticktext=[start_lbl, end_lbl],
            range=[-0.3, 1.3],
            showgrid=False,
        )
        fig.update_yaxes(title="Value")
        fig.update_layout(template="plotly_white", hovermode="closest")
        return BuildResult(figure=fig, warnings=warnings)


registry.register(SlopeChart)
