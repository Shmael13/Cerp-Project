from __future__ import annotations
import copy
from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class WaterfallChart(BaseVisualization):
    name: ClassVar[str] = "Waterfall"
    description: ClassVar[str] = "Show cumulative impact of sequential positive/negative values."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("label", "categorical", "Label"),
            ColumnSpec("value", "numeric",     "Value"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("growth_pct",     "slider",    "Adjust All Values (%)",     0.0,
                           {"min": -50.0, "max": 100.0, "step": 0.5}, category="Data"),
            AssumptionSpec("sort_mode",      "selectbox", "Sort Order",                "As in Data",
                           {"choices": ["As in Data", "By Value (desc)", "By Abs Value (desc)"]}, category="Data"),
            AssumptionSpec("threshold_pct",  "slider",    "Hide Items Below (% of max abs)", 0.0,
                           {"min": 0.0, "max": 50.0, "step": 0.5}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("show_total",     "toggle",    "Show Total Bar",             True,
                           {}, category="Display"),
            AssumptionSpec("positive_color", "selectbox", "Positive Color",             "#2ecc71",
                           {"choices": ["#2ecc71", "#27ae60", "#3498db", "#1abc9c"]}, category="Display"),
            AssumptionSpec("negative_color", "selectbox", "Negative Color",             "#e74c3c",
                           {"choices": ["#e74c3c", "#c0392b", "#e67e22", "#d35400"]}, category="Display"),
            AssumptionSpec("total_color",    "selectbox", "Total Bar Color",            "#2c3e50",
                           {"choices": ["#2c3e50", "#34495e", "#7f8c8d", "#95a5a6"]}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        label_col = columns["label"]
        value_col = columns["value"]
        warnings: list[str] = []

        work = df[[label_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[label_col, value_col])
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values in '{label_col}' or '{value_col}'.")

        if params["growth_pct"] != 0.0:
            work[value_col] = work[value_col] * (1 + params["growth_pct"] / 100)
            warnings.append(f"All values adjusted by {params['growth_pct']:+.1f}%.")

        # Threshold: hide items whose absolute value is below X% of the max
        threshold_pct = float(params["threshold_pct"])
        if threshold_pct > 0.0 and len(work) > 0:
            max_abs = work[value_col].abs().max()
            cutoff = max_abs * (threshold_pct / 100)
            before_t = len(work)
            work = work[work[value_col].abs() >= cutoff]
            removed = before_t - len(work)
            if removed:
                warnings.append(f"Hid {removed} item(s) with absolute value < {threshold_pct:.1f}% of max ({cutoff:.2f}).")

        sort_mode = params["sort_mode"]
        if sort_mode == "By Value (desc)":
            work = work.sort_values(value_col, ascending=False)
        elif sort_mode == "By Abs Value (desc)":
            work = work.reindex(work[value_col].abs().sort_values(ascending=False).index)

        labels  = list(work[label_col])
        values  = list(work[value_col])
        measure = ["relative"] * len(values)

        if params["show_total"]:
            labels.append("Total")
            values.append(sum(values))
            measure.append("total")

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measure,
            x=labels,
            y=values,
            connector={"line": {"color": "#bdc3c7"}},
            increasing={"marker": {"color": params["positive_color"]}},
            decreasing={"marker": {"color": params["negative_color"]}},
            totals={"marker":    {"color": params["total_color"]}},
        ))
        fig.update_layout(template="plotly_white", showlegend=False,
                          yaxis_title=value_col)
        return BuildResult(figure=fig, warnings=warnings)

    def compare(
        self,
        df: pd.DataFrame,
        columns: dict[str, str | None],
        scenarios: dict[str, dict[str, Any]],
    ) -> BuildResult:
        """
        Waterfall traces can't be meaningfully overlaid, so comparison is shown
        as a grouped bar chart — one bar group per label, one bar per scenario.
        """
        from cerp_viz.charts.compare_utils import _PALETTE

        combined = go.Figure()
        all_warnings: list[str] = []

        for i, (name, params) in enumerate(scenarios.items()):
            result = self.build(df, columns, params)
            w = result.figure.data[0]
            color = _PALETTE[i % len(_PALETTE)]

            combined.add_trace(go.Bar(
                name=name,
                x=list(w.x),
                y=list(w.y),
                marker_color=color,
                legendgroup=name,
            ))
            all_warnings.extend(f"[{name}] {msg}" for msg in result.warnings)

        combined.update_layout(
            template="plotly_white",
            barmode="group",
            showlegend=True,
            legend_title_text="Scenario",
            yaxis_title=columns["value"],
        )
        return BuildResult(figure=combined, warnings=all_warnings)


registry.register(WaterfallChart)
