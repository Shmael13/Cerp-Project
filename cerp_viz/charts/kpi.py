from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class KPIChart(BaseVisualization):
    name: ClassVar[str] = "KPI Tiles"
    description: ClassVar[str] = "At-a-glance metric cards — show totals, growth, and targets side by side."
    supports_comparison: ClassVar[bool] = False

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("value",     "numeric", "Primary Metric"),
            ColumnSpec("value2",    "numeric", "Second Metric (optional)",  required=False),
            ColumnSpec("value3",    "numeric", "Third Metric (optional)",   required=False),
            ColumnSpec("value4",    "numeric", "Fourth Metric (optional)",  required=False),
            ColumnSpec("reference", "numeric", "Reference / Prior Period (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation",  "selectbox", "Aggregation",              "sum",
                           {"choices": ["sum", "mean", "median", "count", "max", "min"]}, category="Data"),
            AssumptionSpec("delta_mode",   "selectbox", "Delta Display",            "absolute",
                           {"choices": ["absolute", "relative", "none"]}, category="Display"),
            AssumptionSpec("number_format","selectbox", "Number Format",            ",.0f",
                           {"choices": [",.0f", ",.2f", ",.1f", ".2%", ".1%", ".0%"]}, category="Display"),
            AssumptionSpec("prefix",       "selectbox", "Prefix",                   "",
                           {"choices": ["", "$", "€", "£", "¥", "#"]}, category="Display"),
            AssumptionSpec("suffix",       "selectbox", "Suffix",                   "",
                           {"choices": ["", "%", "K", "M", "B", " units", " days"]}, category="Display"),
            AssumptionSpec("increasing_color", "selectbox", "Increasing = Good?",  "green",
                           {"choices": ["green", "red"]}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []
        agg = params["aggregation"]
        prefix = params["prefix"]
        suffix = params["suffix"]
        fmt    = params["number_format"]
        inc_color = "#2ca02c" if params["increasing_color"] == "green" else "#d62728"
        dec_color = "#d62728" if params["increasing_color"] == "green" else "#2ca02c"

        # Collect metric columns in order
        metric_roles  = ["value", "value2", "value3", "value4"]
        metric_labels = [columns.get(r) for r in metric_roles]
        metrics = [(col, col) for col in metric_labels if col and col in df.columns]

        if not metrics:
            return BuildResult(
                figure=go.Figure().update_layout(
                    title="No numeric columns selected", template="plotly_white"
                ),
                warnings=["No valid metric columns found."],
            )

        ref_col = columns.get("reference")

        n = len(metrics)
        fig = make_subplots(
            rows=1, cols=n,
            subplot_titles=[label for _, label in metrics],
            specs=[[{"type": "indicator"}] * n],
        )

        for i, (col, label) in enumerate(metrics, start=1):
            work = df[col].dropna()
            val = float(getattr(work, agg)() if hasattr(work, agg) else work.sum())

            indicator_kw: dict[str, Any] = dict(
                mode="number+delta" if ref_col and ref_col in df.columns and params["delta_mode"] != "none" else "number",
                value=val,
                number={
                    "prefix": prefix,
                    "suffix": suffix,
                    "valueformat": fmt,
                    "font": {"size": 42},
                },
                title={"text": f"<b>{label}</b>", "font": {"size": 15}},
            )

            if ref_col and ref_col in df.columns and params["delta_mode"] != "none":
                ref_val = float(getattr(df[ref_col].dropna(), agg)())
                delta_val = val - ref_val
                if params["delta_mode"] == "relative" and ref_val != 0:
                    delta_val = (val - ref_val) / abs(ref_val)
                    delta_fmt = ".1%"
                else:
                    delta_fmt = fmt
                indicator_kw["delta"] = {
                    "reference": ref_val,
                    "valueformat": delta_fmt,
                    "increasing": {"color": inc_color},
                    "decreasing": {"color": dec_color},
                }

            fig.add_trace(go.Indicator(**indicator_kw), row=1, col=i)

        fig.update_layout(
            template="plotly_white",
            height=220,
            margin=dict(t=60, b=20, l=20, r=20),
        )

        if ref_col and ref_col not in df.columns:
            warnings.append(f"Reference column '{ref_col}' not found — delta not shown.")

        return BuildResult(figure=fig, warnings=warnings)


registry.register(KPIChart)
