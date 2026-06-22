from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class BulletChart(BaseVisualization):
    name: ClassVar[str] = "Bullet Chart"
    description: ClassVar[str] = "Actual vs target with performance ranges — the richest KPI visual."
    supports_comparison: ClassVar[bool] = False

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("label",  "categorical", "Metric Name (Category)"),
            ColumnSpec("actual", "numeric",     "Actual Value"),
            ColumnSpec("target", "numeric",     "Target Value"),
            ColumnSpec("low",    "numeric",     "Poor Range Limit (optional)",  required=False),
            ColumnSpec("high",   "numeric",     "Good Range Limit (optional)",  required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation",  "selectbox", "Aggregation per Metric", "sum",
                           {"choices": ["sum", "mean", "median", "max", "min"]}, category="Data"),
            AssumptionSpec("top_n",        "number_input", "Show Top N Metrics (0 = all)", 0,
                           {"min": 0, "max": 50, "step": 1}, category="Data"),
            AssumptionSpec("orientation",  "selectbox", "Orientation",            "Horizontal",
                           {"choices": ["Horizontal", "Vertical"]}, category="Display"),
            AssumptionSpec("bar_color",    "selectbox", "Actual Bar Color",       "steelblue",
                           {"choices": ["steelblue", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]}, category="Display"),
            AssumptionSpec("show_values",  "toggle",    "Annotate Values",        True,
                           {}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        label_col  = columns["label"]
        actual_col = columns["actual"]
        target_col = columns["target"]
        low_col    = columns.get("low")
        high_col   = columns.get("high")
        warnings: list[str] = []
        agg = params["aggregation"]

        work = df.copy()
        for c in [actual_col, target_col]:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[label_col, actual_col, target_col])
        if (dropped := before - len(work)):
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        # Aggregate to one row per label
        agg_dict: dict[str, str] = {actual_col: agg, target_col: agg}
        if low_col and low_col in work.columns:
            work[low_col] = pd.to_numeric(work[low_col], errors="coerce")
            agg_dict[low_col] = "mean"
        if high_col and high_col in work.columns:
            work[high_col] = pd.to_numeric(work[high_col], errors="coerce")
            agg_dict[high_col] = "mean"

        work[label_col] = work[label_col].astype(str)
        work = work.groupby(label_col, as_index=False).agg(agg_dict)
        work = work.sort_values(actual_col, ascending=False)

        top_n = int(params["top_n"])
        if top_n > 0:
            work = work.head(top_n)
            warnings.append(f"Showing top {top_n} metrics by actual value.")

        if len(work) == 0:
            return BuildResult(
                figure=go.Figure().update_layout(title="No data", template="plotly_white"),
                warnings=["No rows after filtering."],
            )

        horizontal = params["orientation"] == "Horizontal"
        n = len(work)
        fig = make_subplots(
            rows=n if horizontal else 1,
            cols=1 if horizontal else n,
            subplot_titles=list(work[label_col]),
        )

        bar_color  = params["bar_color"]
        show_vals  = params["show_values"]

        for i, row in enumerate(work.itertuples(), start=1):
            actual = getattr(row, actual_col)
            target = getattr(row, target_col)

            # Range bands (use ±20% of target if not provided)
            low  = getattr(row, low_col,  target * 0.8) if low_col  and low_col  in work.columns else target * 0.8
            high = getattr(row, high_col, target * 1.2) if high_col and high_col in work.columns else target * 1.2
            max_range = max(actual, target, high) * 1.1

            subplot_kw = {"row": i, "col": 1} if horizontal else {"row": 1, "col": i}

            # Poor range (red tint)
            fig.add_trace(go.Bar(
                x=[low if horizontal else None],
                y=[None if horizontal else low],
                orientation="h" if horizontal else "v",
                marker_color="rgba(214,39,40,0.15)",
                showlegend=i == 1,
                name="Poor range",
                hoverinfo="skip",
            ), **subplot_kw)

            # Satisfactory range (yellow tint)
            fig.add_trace(go.Bar(
                x=[(high - low) if horizontal else None],
                y=[None if horizontal else (high - low)],
                orientation="h" if horizontal else "v",
                marker_color="rgba(255,188,0,0.2)",
                showlegend=i == 1,
                name="OK range",
                hoverinfo="skip",
            ), **subplot_kw)

            # Good range (green tint)
            fig.add_trace(go.Bar(
                x=[(max_range - high) if horizontal else None],
                y=[None if horizontal else (max_range - high)],
                orientation="h" if horizontal else "v",
                marker_color="rgba(44,160,44,0.15)",
                showlegend=i == 1,
                name="Good range",
                hoverinfo="skip",
            ), **subplot_kw)

            # Actual bar (thinner, on top)
            text_val = [f"{actual:,.1f}"] if show_vals else None
            fig.add_trace(go.Bar(
                x=[actual if horizontal else None],
                y=[None if horizontal else actual],
                orientation="h" if horizontal else "v",
                marker_color=bar_color,
                width=0.4,
                showlegend=i == 1,
                name="Actual",
                text=text_val,
                textposition="outside",
                hovertemplate=f"Actual: {actual:,.2f}<extra></extra>",
            ), **subplot_kw)

            # Target marker (thin dark line using scatter)
            fig.add_trace(go.Scatter(
                x=[target, target] if horizontal else [i - 0.6, i + 0.6 - 1],
                y=[0.3, 0.7] if horizontal else [target, target],
                mode="lines",
                line=dict(color="black", width=3),
                showlegend=i == 1,
                name="Target",
                hovertemplate=f"Target: {target:,.2f}<extra></extra>",
            ), **subplot_kw)

        axis_kw = dict(range=[0, work[[actual_col, target_col]].max().max() * 1.15])
        fig.update_layout(
            template="plotly_white",
            barmode="stack",
            height=max(120 * n, 300),
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
        )
        if horizontal:
            fig.update_xaxes(**axis_kw)
        else:
            fig.update_yaxes(**axis_kw)

        return BuildResult(figure=fig, warnings=warnings)


registry.register(BulletChart)
