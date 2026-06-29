from typing import Any, ClassVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class StripPlot(BaseVisualization):
    name: ClassVar[str] = "Strip Plot"
    description: ClassVar[str] = (
        "Shows every individual data point grouped by category. "
        "Unlike Box/Violin, no information is hidden — ideal for datasets where "
        "seeing outliers and clusters matters more than a statistical summary."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",     "categorical", "Category column (grouping)", required=True),
            ColumnSpec("y",     "numeric",     "Value column",               required=True),
            ColumnSpec("color", "categorical", "Color group (optional)",     required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("jitter",      "slider",    "Jitter width (0 = strict strip)", 0.3,
                           {"min": 0.0, "max": 1.0, "step": 0.05}, category="Display"),
            AssumptionSpec("orientation", "selectbox", "Orientation",                     "v",
                           {"choices": ["v", "h"]},                 category="Display"),
            AssumptionSpec("marker_size", "slider",    "Marker size",                     6,
                           {"min": 2, "max": 20, "step": 1},        category="Display"),
            AssumptionSpec("opacity",     "slider",    "Opacity",                         0.7,
                           {"min": 0.05, "max": 1.0, "step": 0.05}, category="Display"),
            AssumptionSpec("show_box",    "toggle",    "Overlay box summary",             False,
                           {},                                       category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []
        x_col     = columns["x"]
        y_col     = columns["y"]
        color_col = columns.get("color")
        orient    = params["orientation"]

        plot_x = x_col if orient == "v" else y_col
        plot_y = y_col if orient == "v" else x_col

        fig = px.strip(
            df,
            x=plot_x,
            y=plot_y,
            color=color_col if color_col and color_col in df.columns else None,
            stripmode="overlay",
            template="plotly_white",
            title=f"{y_col} by {x_col}",
        )
        fig.update_traces(
            jitter=float(params["jitter"]),
            marker=dict(size=int(params["marker_size"]), opacity=float(params["opacity"])),
        )

        if params["show_box"]:
            box_kw = (
                dict(x=df[x_col], y=df[y_col]) if orient == "v"
                else dict(x=df[y_col], y=df[x_col])
            )
            fig.add_trace(
                go.Box(
                    **box_kw,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color="rgba(0,0,0,0.4)", width=1),
                    whiskerwidth=0.5,
                    boxpoints=False,
                    showlegend=False,
                    name="Summary",
                )
            )

        if len(df) > 5000:
            warnings.append(f"Large dataset ({len(df):,} rows) — reduce data or enable jitter for readability.")
        return BuildResult(figure=fig, warnings=warnings)


registry.register(StripPlot)
