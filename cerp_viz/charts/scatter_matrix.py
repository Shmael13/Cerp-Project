from typing import Any, ClassVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_COLOR_SEQUENCES = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold":   px.colors.qualitative.Bold,
    "Dark":   px.colors.qualitative.Dark24,
}


class ScatterMatrix(BaseVisualization):
    name: ClassVar[str] = "Scatter Matrix"
    description: ClassVar[str] = (
        "All pairwise scatter plots across numeric columns (SPLOM). "
        "Auto-discovers numeric columns — the color group column is optional."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("_a",    "numeric",     "Numeric col (chart uses all numeric cols)", required=True),
            ColumnSpec("_b",    "numeric",     "Numeric col (chart uses all numeric cols)", required=True),
            ColumnSpec("color", "categorical", "Color group (optional)",                    required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("max_cols",      "slider",    "Max columns to show", 6,
                           {"min": 2, "max": 15, "step": 1},                category="Data"),
            AssumptionSpec("sample_n",      "number_input", "Sample rows (0 = all)", 0,
                           {"min": 0, "max": 5000, "step": 100},            category="Data"),
            AssumptionSpec("opacity",       "slider",    "Point opacity",       0.6,
                           {"min": 0.05, "max": 1.0, "step": 0.05},         category="Display"),
            AssumptionSpec("marker_size",   "slider",    "Marker size",         4,
                           {"min": 1, "max": 15, "step": 1},                category="Display"),
            AssumptionSpec("show_diagonal", "selectbox", "Diagonal",           "histogram",
                           {"choices": ["histogram", "scatter", "box"]},     category="Display"),
            AssumptionSpec("color_scheme",  "selectbox", "Color scheme",       "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)},              category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []
        color_col    = columns.get("color")
        numeric_cols = list(df.select_dtypes("number").columns)

        if len(numeric_cols) < 2:
            return BuildResult(figure=go.Figure(), warnings=["Need at least 2 numeric columns."])

        max_cols = int(params["max_cols"])
        if len(numeric_cols) > max_cols:
            numeric_cols = numeric_cols[:max_cols]
            warnings.append(f"Showing first {max_cols} numeric columns (adjust 'Max columns' to see more).")

        work = df[numeric_cols + ([color_col] if color_col and color_col in df.columns else [])].copy()

        sample_n = int(params["sample_n"])
        if sample_n > 0 and len(work) > sample_n:
            work = work.sample(sample_n, random_state=42)
            warnings.append(f"Sampled {sample_n:,} rows for performance.")

        fig = px.scatter_matrix(
            work,
            dimensions=numeric_cols,
            color=color_col if color_col and color_col in work.columns else None,
            color_discrete_sequence=_COLOR_SEQUENCES[params["color_scheme"]],
            opacity=float(params["opacity"]),
            template="plotly_white",
        )
        fig.update_traces(
            diagonal_visible=True,
            showupperhalf=True,
            marker=dict(size=int(params["marker_size"])),
        )

        diag = params["show_diagonal"]
        if diag == "histogram":
            fig.update_traces(diagonal_visible=True)
        elif diag == "box":
            for trace in fig.data:
                if hasattr(trace, "diagonal"):
                    trace.diagonal.visible = True

        fig.update_layout(height=max(400, 150 * len(numeric_cols)))
        warnings.append(f"SPLOM: {len(numeric_cols)} × {len(numeric_cols)} = {len(numeric_cols)**2} panels.")
        return BuildResult(figure=fig, warnings=warnings)


registry.register(ScatterMatrix)
