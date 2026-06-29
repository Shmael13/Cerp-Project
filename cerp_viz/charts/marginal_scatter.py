from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_MARGINAL_TYPES = ["histogram", "box", "violin", "rug"]
_TRENDLINES     = ["None", "ols", "lowess"]
_COLOR_SEQS = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold":   px.colors.qualitative.Bold,
    "Dark":   px.colors.qualitative.Dark24,
}


class MarginalScatter(BaseVisualization):
    name: ClassVar[str] = "Marginal Scatter"
    description: ClassVar[str] = (
        "Scatter plot with marginal distribution plots on both axes — "
        "simultaneously shows bivariate correlation and each variable's distribution shape."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",     "numeric",     "X axis",                        required=True),
            ColumnSpec("y",     "numeric",     "Y axis",                        required=True),
            ColumnSpec("color", "categorical", "Color group (optional)",        required=False),
            ColumnSpec("size",  "numeric",     "Bubble size column (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("marginal_x",   "selectbox", "X margin type",     "histogram",
                           {"choices": _MARGINAL_TYPES},                    category="Display"),
            AssumptionSpec("marginal_y",   "selectbox", "Y margin type",     "box",
                           {"choices": _MARGINAL_TYPES},                    category="Display"),
            AssumptionSpec("trendline",    "selectbox", "Trendline",         "ols",
                           {"choices": _TRENDLINES},                        category="Display"),
            AssumptionSpec("opacity",      "slider",    "Point opacity",     0.65,
                           {"min": 0.1, "max": 1.0, "step": 0.05},         category="Display"),
            AssumptionSpec("marker_size",  "slider",    "Marker size",       6,
                           {"min": 2, "max": 20, "step": 1},               category="Display"),
            AssumptionSpec("color_scheme", "selectbox", "Color scheme",      "Plotly",
                           {"choices": list(_COLOR_SEQS)},                  category="Display"),
            AssumptionSpec("log_x",        "toggle",    "Log X axis",        False,
                           {},                                              category="Display"),
            AssumptionSpec("log_y",        "toggle",    "Log Y axis",        False,
                           {},                                              category="Display"),
            AssumptionSpec("sample_n",     "number_input", "Sample rows (0 = all)", 0,
                           {"min": 0, "max": 5000, "step": 100},           category="Data"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []
        x_col     = columns["x"]
        y_col     = columns["y"]
        color_col = columns.get("color")
        size_col  = columns.get("size")

        work = df[[c for c in [x_col, y_col, color_col, size_col]
                   if c and c in df.columns]].copy()
        work = work.dropna(subset=[x_col, y_col])

        sample_n = int(params["sample_n"])
        if sample_n > 0 and len(work) > sample_n:
            work = work.sample(sample_n, random_state=42)
            warnings.append(f"Sampled {sample_n:,} rows for performance.")

        if size_col and size_col in work.columns:
            work[size_col] = pd.to_numeric(work[size_col], errors="coerce").abs()
            work = work[work[size_col].notna() & (work[size_col] > 0)]

        trendline = params["trendline"] if params["trendline"] != "None" else None
        color_arg = color_col if color_col and color_col in work.columns else None
        size_arg  = size_col  if size_col  and size_col  in work.columns else None

        try:
            fig = px.scatter(
                work,
                x=x_col,
                y=y_col,
                color=color_arg,
                size=size_arg,
                marginal_x=params["marginal_x"],
                marginal_y=params["marginal_y"],
                trendline=trendline,
                opacity=float(params["opacity"]),
                log_x=bool(params["log_x"]),
                log_y=bool(params["log_y"]),
                color_discrete_sequence=_COLOR_SEQS[params["color_scheme"]],
                template="plotly_white",
                title=f"{y_col} vs {x_col}",
            )
        except Exception as exc:
            warnings.append(f"Trendline skipped ({exc}). Rendering without.")
            fig = px.scatter(
                work,
                x=x_col,
                y=y_col,
                color=color_arg,
                size=size_arg,
                marginal_x=params["marginal_x"],
                marginal_y=params["marginal_y"],
                opacity=float(params["opacity"]),
                log_x=bool(params["log_x"]),
                log_y=bool(params["log_y"]),
                color_discrete_sequence=_COLOR_SEQS[params["color_scheme"]],
                template="plotly_white",
                title=f"{y_col} vs {x_col}",
            )

        fig.update_traces(
            selector=dict(mode="markers"),
            marker=dict(size=int(params["marker_size"])),
        )

        try:
            pair = work[[x_col, y_col]].dropna()
            r = float(pair[x_col].corr(pair[y_col]))
            if np.isfinite(r):
                direction = "positive" if r > 0 else "negative"
                strength  = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
                warnings.append(
                    f"Pearson r = {r:.3f} — {strength} {direction} correlation "
                    f"({len(pair):,} non-null pairs)."
                )
        except Exception:
            pass

        if params["log_x"]:
            warnings.append("X axis is on a log scale.")
        if params["log_y"]:
            warnings.append("Y axis is on a log scale.")

        return BuildResult(figure=fig, warnings=warnings)


registry.register(MarginalScatter)
