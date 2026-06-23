"""
Bump Chart — ranks over time.
Ranks each category by its aggregated value within each time period;
y-axis is inverted so rank 1 sits at the top.
"""
from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

import plotly.express as px

_COLOR_SEQUENCES = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Dark":   px.colors.qualitative.Dark24,
    "Bold":   px.colors.qualitative.Bold,
    "Vivid":  px.colors.qualitative.Vivid,
}


class BumpChart(BaseVisualization):
    name: ClassVar[str] = "Bump Chart"
    description: ClassVar[str] = (
        "Track how rankings change over time across categories. "
        "Each line is a category; the y-axis shows rank (1 = best)."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("time",     "any",         "Time / X Axis (ordinal or date)"),
            ColumnSpec("value",    "numeric",      "Value to rank by"),
            ColumnSpec("category", "categorical",  "Category (one line per value)"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation", "selectbox", "Aggregation within period", "sum",
                           {"choices": ["sum", "mean", "max", "min", "count"]}, category="Data"),
            AssumptionSpec("top_n", "number_input", "Show top N categories (0 = all)", 10,
                           {"min": 0, "max": 50, "step": 1}, category="Data"),
            AssumptionSpec("ascending", "toggle", "Rank ascending (lower value = rank 1)", False,
                           {}, category="Data"),
            AssumptionSpec("show_markers", "toggle", "Show markers at each period", True,
                           {}, category="Display"),
            AssumptionSpec("line_width",   "slider",  "Line width", 2,
                           {"min": 1, "max": 6, "step": 1}, category="Display"),
            AssumptionSpec("color_scheme", "selectbox", "Color scheme", "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        time_col  = columns["time"]
        value_col = columns["value"]
        cat_col   = columns["category"]
        warnings: list[str] = []

        work = df[[time_col, value_col, cat_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col])

        # Aggregate within each (time, category) cell
        agg = params["aggregation"]
        grouped = work.groupby([time_col, cat_col], as_index=False)[value_col].agg(agg)

        # Rank within each time period
        asc = bool(params["ascending"])
        grouped["__rank__"] = (
            grouped.groupby(time_col)[value_col]
            .rank(ascending=asc, method="min")
            .astype(int)
        )

        # Restrict to top-N categories by median rank (lowest = best)
        top_n = int(params["top_n"])
        if top_n > 0:
            med_rank = grouped.groupby(cat_col)["__rank__"].median()
            top_cats = med_rank.nsmallest(top_n).index
            grouped  = grouped[grouped[cat_col].isin(top_cats)]

        # Sort x axis (preserve original order if possible)
        try:
            grouped[time_col] = pd.to_datetime(grouped[time_col], errors="ignore")
        except Exception:
            pass
        time_order = grouped.groupby(time_col)[value_col].sum().sort_index().index.tolist()

        colors  = _COLOR_SEQUENCES[params["color_scheme"]]
        cats    = grouped[cat_col].unique().tolist()
        max_rank = int(grouped["__rank__"].max())

        mode    = "lines+markers" if params["show_markers"] else "lines"
        fig     = go.Figure()

        for idx, cat in enumerate(cats):
            sub = grouped[grouped[cat_col] == cat].set_index(time_col).reindex(time_order)
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=time_order,
                y=sub["__rank__"].tolist(),
                mode=mode,
                name=str(cat),
                line=dict(width=params["line_width"], color=color),
                marker=dict(size=8, color=color),
                hovertemplate=f"<b>{cat}</b><br>Period: %{{x}}<br>Rank: %{{y}}<br>Value: %{{customdata:.2f}}<extra></extra>",
                customdata=sub[value_col].tolist(),
            ))

        fig.update_yaxes(
            autorange="reversed",
            title="Rank",
            tickmode="linear", tick0=1, dtick=1,
            range=[max_rank + 0.5, 0.5],
        )
        fig.update_xaxes(title=time_col)
        fig.update_layout(template="plotly_white", hovermode="x unified")

        if top_n > 0 and len(cats) == top_n:
            warnings.append(f"Showing top {top_n} categories by median rank.")

        return BuildResult(figure=fig, warnings=warnings)


registry.register(BumpChart)
