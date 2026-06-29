from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_SCALES = ["Blues", "Viridis", "Plasma", "Inferno", "Magma", "Turbo", "Hot", "YlOrRd", "Cividis"]
_AGGS   = ["count", "sum", "avg", "min", "max"]


class DensityHeatmap(BaseVisualization):
    name: ClassVar[str] = "Density Heatmap"
    description: ClassVar[str] = (
        "2-D histogram showing where data concentrates across two numeric axes. "
        "Great for large datasets where scatter plots become overplotted."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x", "numeric", "X axis (numeric)",                              required=True),
            ColumnSpec("y", "numeric", "Y axis (numeric)",                              required=True),
            ColumnSpec("z", "numeric", "Value to aggregate (optional, defaults to count)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("nbins_x",      "slider",    "X bins",           20,
                           {"min": 5, "max": 100, "step": 5},  category="Data"),
            AssumptionSpec("nbins_y",      "slider",    "Y bins",           20,
                           {"min": 5, "max": 100, "step": 5},  category="Data"),
            AssumptionSpec("aggregation",  "selectbox", "Aggregation",      "count",
                           {"choices": _AGGS},                  category="Data"),
            AssumptionSpec("color_scale",  "selectbox", "Color scale",      "Blues",
                           {"choices": _SCALES},                category="Display"),
            AssumptionSpec("log_scale",    "toggle",    "Log color scale",  False,
                           {},                                  category="Display"),
            AssumptionSpec("show_contour", "toggle",    "Overlay contours", False,
                           {},                                  category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []
        x_col = columns["x"]
        y_col = columns["y"]
        z_col = columns.get("z")

        agg   = params["aggregation"]
        z_arg = z_col if (z_col and z_col in df.columns and agg != "count") else None

        fig = px.density_heatmap(
            df,
            x=x_col,
            y=y_col,
            z=z_arg,
            histfunc=agg if agg != "count" else "count",
            nbinsx=int(params["nbins_x"]),
            nbinsy=int(params["nbins_y"]),
            color_continuous_scale=params["color_scale"],
            template="plotly_white",
            title=(
                f"Density of {y_col} vs {x_col}"
                + (f" — {agg}({z_col})" if z_arg else "")
            ),
        )

        if params["log_scale"]:
            for trace in fig.data:
                if hasattr(trace, "z") and trace.z is not None:
                    trace.z = np.log1p(trace.z)
            fig.update_coloraxes(colorbar_title="log(1+n)")
            warnings.append("Color shows log(1+count) — easier reading of highly skewed distributions.")

        if params["show_contour"]:
            contour = px.density_contour(df, x=x_col, y=y_col).data[0]
            contour.line.color = "white"
            contour.line.width = 1
            fig.add_trace(contour)

        nb_x = int(params["nbins_x"])
        nb_y = int(params["nbins_y"])
        warnings.append(f"{len(df):,} rows binned into {nb_x}×{nb_y} grid.")
        return BuildResult(figure=fig, warnings=warnings)


registry.register(DensityHeatmap)
