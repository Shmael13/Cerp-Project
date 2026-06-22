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
    "Antique": px.colors.qualitative.Antique,
}


class AreaChart(BaseVisualization):
    name: ClassVar[str] = "Area Chart"
    description: ClassVar[str] = "Show cumulative totals or part-to-whole trends over time."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",      "any",        "X Axis (time or ordered)"),
            ColumnSpec("y",      "numeric",    "Y Axis (value)"),
            ColumnSpec("series", "categorical", "Stack By / Series (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation",   "selectbox",    "Aggregation",            "sum",
                           {"choices": ["sum", "mean", "median", "count", "max", "min"]}, category="Data"),
            AssumptionSpec("y_multiplier",  "slider",       "Y Multiplier",            1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("rolling_window","slider",        "Rolling Average Window",  1,
                           {"min": 1, "max": 30, "step": 1}, category="Data"),
            AssumptionSpec("cumulative",    "toggle",        "Cumulative Sum",          False,
                           {}, category="Data"),
            AssumptionSpec("fill_mode",     "selectbox",    "Fill Mode",              "tozeroy",
                           {"choices": ["tozeroy", "tonexty", "none"]}, category="Display"),
            AssumptionSpec("groupnorm",     "selectbox",    "Normalise to",           "none",
                           {"choices": ["none", "fraction", "percent"]}, category="Display"),
            AssumptionSpec("line_mode",     "selectbox",    "Line Style",             "lines",
                           {"choices": ["lines", "lines+markers"]}, category="Display"),
            AssumptionSpec("opacity",       "slider",       "Fill Opacity",            0.6,
                           {"min": 0.1, "max": 1.0, "step": 0.05}, category="Display"),
            AssumptionSpec("color_scheme",  "selectbox",    "Color Scheme",           "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        x_col      = columns["x"]
        y_col      = columns["y"]
        series_col = columns.get("series")
        warnings: list[str] = []

        work = df.copy()
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[x_col, y_col])
        if (dropped := before - len(work)):
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        if params["y_multiplier"] != 1.0:
            work[y_col] = work[y_col] * params["y_multiplier"]
            warnings.append(f"Values multiplied by {params['y_multiplier']}×.")

        # Aggregate if series present (avoid double-counting repeated x values)
        group_cols = [x_col] + ([series_col] if series_col else [])
        work = work.groupby(group_cols, as_index=False)[y_col].agg(params["aggregation"])

        window = int(params["rolling_window"])
        if window > 1:
            if series_col:
                work[y_col] = (
                    work.groupby(series_col)[y_col]
                    .transform(lambda s: s.rolling(window, min_periods=1).mean())
                )
            else:
                work[y_col] = work[y_col].rolling(window, min_periods=1).mean()
            warnings.append(f"Rolling average applied (window = {window}).")

        if params["cumulative"]:
            if series_col:
                work[y_col] = work.groupby(series_col)[y_col].cumsum()
            else:
                work[y_col] = work[y_col].cumsum()
            warnings.append("Cumulative sum applied.")

        norm = params["groupnorm"]
        groupnorm_kw = norm if norm != "none" else None

        fig = px.area(
            work, x=x_col, y=y_col,
            color=series_col,
            groupnorm=groupnorm_kw,
            color_discrete_sequence=_COLOR_SEQUENCES[params["color_scheme"]],
            template="plotly_white",
        )

        fill = params["fill_mode"]
        fig.update_traces(
            mode=params["line_mode"],
            fill=fill if fill != "none" else "none",
            opacity=params["opacity"],
        )

        if groupnorm_kw:
            warnings.append(f"Values normalised to {norm}.")

        return BuildResult(figure=fig, warnings=warnings)

    def compare(self, df, columns, scenarios):
        from cerp_viz.charts.compare_utils import overlay_scenarios
        return overlay_scenarios(self, df, columns, scenarios)


registry.register(AreaChart)
