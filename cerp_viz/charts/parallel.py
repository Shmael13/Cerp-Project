"""
Parallel Coordinates — explore patterns across multiple numeric dimensions.
Each row becomes a polyline crossing all dimension axes.
"""
from typing import Any, ClassVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class ParallelCoordinates(BaseVisualization):
    name: ClassVar[str] = "Parallel Coordinates"
    description: ClassVar[str] = (
        "Visualise patterns across multiple numeric dimensions simultaneously. "
        "Each row is drawn as a line crossing all axes — useful for spotting clusters and outliers."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("dim1",  "numeric",     "Dimension 1"),
            ColumnSpec("dim2",  "numeric",     "Dimension 2"),
            ColumnSpec("dim3",  "numeric",     "Dimension 3 (optional)", required=False),
            ColumnSpec("dim4",  "numeric",     "Dimension 4 (optional)", required=False),
            ColumnSpec("dim5",  "numeric",     "Dimension 5 (optional)", required=False),
            ColumnSpec("dim6",  "numeric",     "Dimension 6 (optional)", required=False),
            ColumnSpec("color", "any",         "Color (numeric or categorical, optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("color_scheme", "selectbox", "Color scale (numeric color col)", "Viridis",
                           {"choices": ["Viridis", "Plasma", "Turbo", "RdBu", "Spectral"]}, category="Display"),
            AssumptionSpec("line_opacity", "slider", "Line opacity", 0.5,
                           {"min": 0.05, "max": 1.0, "step": 0.05}, category="Display"),
            AssumptionSpec("max_rows", "number_input", "Max rows to plot (0 = all)", 2000,
                           {"min": 0, "max": 20000, "step": 500}, category="Data"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []

        dim_roles = [f"dim{i}" for i in range(1, 7)]
        dim_cols  = [columns[r] for r in dim_roles if columns.get(r)]
        color_col = columns.get("color")

        if len(dim_cols) < 2:
            raise ValueError("Parallel Coordinates requires at least 2 dimension columns.")

        keep = dim_cols + ([color_col] if color_col else [])
        work = df[keep].copy()
        for c in dim_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work = work.dropna(subset=dim_cols)

        max_rows = int(params["max_rows"])
        if max_rows > 0 and len(work) > max_rows:
            work = work.sample(max_rows, random_state=42)
            warnings.append(f"Sampled {max_rows} rows for performance.")

        if color_col:
            # Try numeric first; fall back to categorical ordinal encoding
            color_num = pd.to_numeric(work[color_col], errors="coerce")
            if color_num.notna().mean() > 0.8:
                work["__color__"] = color_num
            else:
                cats = work[color_col].astype("category")
                work["__color__"] = cats.cat.codes
                warnings.append(f"'{color_col}' encoded as ordinal integers for colour scale.")
            color_key = "__color__"
        else:
            color_key = None

        dimensions = [
            dict(label=c, values=work[c].tolist(),
                 range=[work[c].min(), work[c].max()])
            for c in dim_cols
        ]

        line_cfg: dict = dict(color=work[color_key].tolist() if color_key else "#3498db",
                              colorscale=params["color_scheme"] if color_key else None,
                              showscale=bool(color_key and color_col))

        fig = go.Figure(go.Parcoords(
            line=line_cfg,
            dimensions=dimensions,
        ))
        fig.update_traces(line_color=None if color_key else "rgba(52,152,219,0.5)")
        fig.update_layout(template="plotly_white")

        if not color_key:
            # Plotly Parcoords doesn't support opacity directly; inject via marker alpha
            warnings.append("Line opacity is approximate on Parallel Coordinates; use colour to highlight groups.")

        return BuildResult(figure=fig, warnings=warnings)


registry.register(ParallelCoordinates)
