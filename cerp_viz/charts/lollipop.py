from typing import Any, ClassVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_AGGS = ["sum", "mean", "median", "count", "max", "min"]
_SORTS = ["Value (desc)", "Value (asc)", "Category (A-Z)", "Category (Z-A)"]
_COLOR_SEQS = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold":   px.colors.qualitative.Bold,
    "Dark":   px.colors.qualitative.Dark24,
}


def _stem_trace(cats, vals, baseline: float, orient: str, color: str) -> go.Scatter:
    """Single trace for all stems using None-separator trick."""
    xs, ys = [], []
    for c, v in zip(cats, vals):
        if orient == "v":
            xs += [c, c, None]
            ys += [baseline, v, None]
        else:
            xs += [baseline, v, None]
            ys += [c, c, None]
    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=color, width=1.5),
        showlegend=False,
        hoverinfo="skip",
    )


def _dot_trace(cats, vals, name: str, color: str, marker_size: int,
               show_text: bool, orient: str) -> go.Scatter:
    text = [f"{v:,.2f}" for v in vals] if show_text else None
    text_pos = "top center" if orient == "v" else "middle right"
    return go.Scatter(
        x=cats if orient == "v" else vals,
        y=vals if orient == "v" else cats,
        mode="markers+text" if show_text else "markers",
        marker=dict(size=marker_size, color=color, line=dict(color="white", width=1)),
        text=text,
        textposition=text_pos,
        textfont=dict(size=10),
        name=name,
        hovertemplate="%{x}: %{y:,.2f}<extra></extra>" if orient == "v"
                      else "%{y}: %{x:,.2f}<extra></extra>",
    )


class LollipopChart(BaseVisualization):
    name: ClassVar[str] = "Lollipop Chart"
    description: ClassVar[str] = (
        "Dot-and-stem chart — a cleaner alternative to bar charts for ranking categories. "
        "Minimal ink, high legibility, works well for many categories."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",     "categorical", "Category column",          required=True),
            ColumnSpec("y",     "numeric",     "Value column",             required=True),
            ColumnSpec("color", "categorical", "Group by color (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation",  "selectbox",    "Aggregation",                    "sum",
                           {"choices": _AGGS},                                  category="Data"),
            AssumptionSpec("top_n",        "number_input", "Top N categories (0 = all)",     0,
                           {"min": 0, "max": 100, "step": 1},                   category="Data"),
            AssumptionSpec("sort_by",      "selectbox",    "Sort by",                        "Value (desc)",
                           {"choices": _SORTS},                                  category="Display"),
            AssumptionSpec("orientation",  "selectbox",    "Orientation",                    "v",
                           {"choices": ["v", "h"]},                             category="Display"),
            AssumptionSpec("marker_size",  "slider",       "Dot size",                       12,
                           {"min": 4, "max": 30, "step": 1},                    category="Display"),
            AssumptionSpec("stem_opacity", "slider",       "Stem opacity",                   0.45,
                           {"min": 0.1, "max": 1.0, "step": 0.05},             category="Display"),
            AssumptionSpec("show_values",  "toggle",       "Show values on dots",            False,
                           {},                                                   category="Display"),
            AssumptionSpec("show_mean",    "toggle",       "Show grand-mean line",           False,
                           {},                                                   category="Display"),
            AssumptionSpec("color_scheme", "selectbox",    "Color scheme",                   "Plotly",
                           {"choices": list(_COLOR_SEQS)},                      category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []
        x_col     = columns["x"]
        y_col     = columns["y"]
        color_col = columns.get("color")

        work = df.copy()
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        before = len(work)
        work = work.dropna(subset=[x_col, y_col])
        if (dropped := before - len(work)):
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        agg        = params["aggregation"]
        orient     = params["orientation"]
        marker_sz  = int(params["marker_size"])
        show_text  = bool(params["show_values"])
        palette    = _COLOR_SEQS[params["color_scheme"]]
        stem_color = f"rgba(120,120,120,{params['stem_opacity']:.2f})"

        group_cols = [x_col] + ([color_col] if color_col else [])
        work = work.groupby(group_cols, as_index=False)[y_col].agg(agg)

        sort = params["sort_by"]
        if sort == "Value (desc)":
            work = work.sort_values(y_col, ascending=False)
        elif sort == "Value (asc)":
            work = work.sort_values(y_col, ascending=True)
        elif sort == "Category (A-Z)":
            work = work.sort_values(x_col)
        else:
            work = work.sort_values(x_col, ascending=False)

        top_n = int(params["top_n"])
        if top_n > 0:
            top_cats = work.groupby(x_col)[y_col].sum().abs().nlargest(top_n).index
            work = work[work[x_col].isin(top_cats)]
            warnings.append(f"Showing top {top_n} categories.")

        baseline = 0.0
        fig = go.Figure()

        if color_col and color_col in work.columns:
            groups = work[color_col].unique()
            for i, grp in enumerate(groups):
                sub  = work[work[color_col] == grp]
                clr  = palette[i % len(palette)]
                cats = sub[x_col].tolist()
                vals = sub[y_col].tolist()
                fig.add_trace(_stem_trace(cats, vals, baseline, orient,
                                          stem_color))
                fig.add_trace(_dot_trace(cats, vals, str(grp), clr,
                                         marker_sz, show_text, orient))
        else:
            cats = work[x_col].tolist()
            vals = work[y_col].tolist()
            colors = [palette[i % len(palette)] for i in range(len(cats))]
            fig.add_trace(_stem_trace(cats, vals, baseline, orient, stem_color))
            fig.add_trace(go.Scatter(
                x=cats if orient == "v" else vals,
                y=vals if orient == "v" else cats,
                mode="markers+text" if show_text else "markers",
                marker=dict(size=marker_sz, color=colors,
                            line=dict(color="white", width=1)),
                text=[f"{v:,.2f}" for v in vals] if show_text else None,
                textposition="top center" if orient == "v" else "middle right",
                textfont=dict(size=10),
                showlegend=False,
                hovertemplate=(
                    "%{x}: %{y:,.2f}<extra></extra>" if orient == "v"
                    else "%{y}: %{x:,.2f}<extra></extra>"
                ),
            ))

        if params["show_mean"]:
            grand_mean = float(work[y_col].mean())
            if orient == "v":
                fig.add_hline(y=grand_mean, line_dash="dot", line_color="#888",
                              annotation_text=f"Mean {grand_mean:,.1f}",
                              annotation_position="top left")
            else:
                fig.add_vline(x=grand_mean, line_dash="dot", line_color="#888",
                              annotation_text=f"Mean {grand_mean:,.1f}",
                              annotation_position="top right")

        cat_order = work[x_col].unique().tolist()
        if orient == "v":
            fig.update_layout(
                xaxis=dict(categoryorder="array", categoryarray=cat_order,
                           title=x_col),
                yaxis=dict(title=f"{agg}({y_col})"),
            )
        else:
            fig.update_layout(
                yaxis=dict(categoryorder="array", categoryarray=cat_order,
                           title=x_col),
                xaxis=dict(title=f"{agg}({y_col})"),
            )

        fig.update_layout(
            template="plotly_white",
            title=f"{agg.capitalize()} of {y_col} by {x_col}",
        )

        n_cats = work[x_col].nunique()
        if n_cats > 30:
            warnings.append(f"{n_cats} categories — consider Top N to reduce clutter.")
        return BuildResult(figure=fig, warnings=warnings)


registry.register(LollipopChart)
