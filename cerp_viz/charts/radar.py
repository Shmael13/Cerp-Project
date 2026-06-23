"""
Radar / Spider Chart — compare multiple metrics across categories.
Expects long-format data: one row per (entity, metric, value).
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


class RadarChart(BaseVisualization):
    name: ClassVar[str] = "Radar Chart"
    description: ClassVar[str] = (
        "Compare multiple metrics across categories in a spider-web shape. "
        "Data must be in long format: (entity, metric, value) per row."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("entity",  "categorical", "Entity (one polygon per value)"),
            ColumnSpec("metric",  "categorical", "Metric / Spoke label"),
            ColumnSpec("value",   "numeric",     "Value (spoke length)"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation", "selectbox", "Aggregation (if multiple rows per entity/metric)", "mean",
                           {"choices": ["mean", "sum", "max", "min"]}, category="Data"),
            AssumptionSpec("normalize",   "toggle",    "Normalize each metric to 0–1 scale", False,
                           {}, category="Data"),
            AssumptionSpec("top_n_entities", "number_input", "Max entities to show (0 = all)", 8,
                           {"min": 0, "max": 30, "step": 1}, category="Data"),
            AssumptionSpec("fill_opacity", "slider",   "Fill opacity",  0.15,
                           {"min": 0.0, "max": 0.6, "step": 0.05}, category="Display"),
            AssumptionSpec("line_width",   "slider",   "Line width",    2,
                           {"min": 1, "max": 5, "step": 1}, category="Display"),
            AssumptionSpec("color_scheme", "selectbox","Color scheme", "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        entity_col = columns["entity"]
        metric_col = columns["metric"]
        value_col  = columns["value"]
        warnings: list[str] = []

        work = df[[entity_col, metric_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col])

        # Aggregate duplicates
        agg = params["aggregation"]
        pivoted = work.groupby([entity_col, metric_col], as_index=False)[value_col].agg(agg)
        pivoted = pivoted.pivot(index=entity_col, columns=metric_col, values=value_col).fillna(0)

        # Restrict entities
        top_n = int(params["top_n_entities"])
        if top_n > 0 and len(pivoted) > top_n:
            # Keep top-N by sum of all metrics
            top_ents = pivoted.sum(axis=1).nlargest(top_n).index
            pivoted  = pivoted.loc[top_ents]
            warnings.append(f"Showing top {top_n} entities by total metric value.")

        # Normalize metrics to 0–1
        if params["normalize"]:
            for col in pivoted.columns:
                lo, hi = pivoted[col].min(), pivoted[col].max()
                pivoted[col] = (pivoted[col] - lo) / (hi - lo) if hi != lo else 0.0
            warnings.append("Each metric normalized to 0–1 scale.")

        metrics  = list(pivoted.columns)
        entities = list(pivoted.index)
        colors   = _COLOR_SEQUENCES[params["color_scheme"]]

        fig = go.Figure()
        for idx, entity in enumerate(entities):
            values = list(pivoted.loc[entity])
            color  = colors[idx % len(colors)]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],            # close the polygon
                theta=metrics + [metrics[0]],
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", f",{params['fill_opacity']})") if color.startswith("rgb") else color,
                line=dict(color=color, width=params["line_width"]),
                name=str(entity),
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, showline=False)),
            template="plotly_white",
        )
        return BuildResult(figure=fig, warnings=warnings)


registry.register(RadarChart)
