from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class TornadoChart(BaseVisualization):
    name: ClassVar[str] = "Tornado Chart"
    description: ClassVar[str] = "Rank factors by signed impact to reveal which assumptions drive outcomes most."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("label", "categorical", "Factor / Variable"),
            ColumnSpec("value", "numeric",     "Impact Value (signed)"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("multiplier",        "slider",       "Value Multiplier",          1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("top_n",             "number_input", "Top N Factors (0 = all)",   0,
                           {"min": 0, "max": 50, "step": 1}, category="Data"),
            AssumptionSpec("direction_filter",  "selectbox",    "Show Factors",              "All",
                           {"choices": ["All", "Positive only", "Negative only"]}, category="Data"),
            AssumptionSpec("min_impact_pct",    "slider",       "Hide Below (% of max abs)", 0.0,
                           {"min": 0.0, "max": 50.0, "step": 1.0}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("positive_color",    "selectbox",    "Positive Impact Color",     "#2ecc71",
                           {"choices": ["#2ecc71", "#27ae60", "#3498db", "#1abc9c"]}, category="Display"),
            AssumptionSpec("negative_color",    "selectbox",    "Negative Impact Color",     "#e74c3c",
                           {"choices": ["#e74c3c", "#c0392b", "#e67e22", "#d35400"]}, category="Display"),
            AssumptionSpec("show_baseline",     "toggle",       "Show Zero Baseline",        True,
                           {}, category="Display"),
            AssumptionSpec("show_values",       "toggle",       "Show Values on Bars",       True,
                           {}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        label_col = columns["label"]
        value_col = columns["value"]
        warnings: list[str] = []

        work = df[[label_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

        before = len(work)
        work = work.dropna()
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        if params["multiplier"] != 1.0:
            work[value_col] = work[value_col] * params["multiplier"]
            warnings.append(f"Impact values multiplied by {params['multiplier']}×.")

        work = work.groupby(label_col, as_index=False)[value_col].sum()

        # Direction filter
        direction = params["direction_filter"]
        if direction == "Positive only":
            work = work[work[value_col] >= 0]
            warnings.append("Showing only positive-impact factors.")
        elif direction == "Negative only":
            work = work[work[value_col] < 0]
            warnings.append("Showing only negative-impact factors.")

        # Threshold filter (hide tiny bars)
        min_pct = float(params["min_impact_pct"])
        if min_pct > 0.0 and len(work) > 0:
            max_abs = work[value_col].abs().max()
            cutoff = max_abs * (min_pct / 100)
            before_t = len(work)
            work = work[work[value_col].abs() >= cutoff]
            removed = before_t - len(work)
            if removed:
                warnings.append(f"Hid {removed} factor(s) below {min_pct:.0f}% of max impact ({cutoff:.2f}).")

        # Top-N
        top_n = int(params["top_n"])
        if top_n > 0:
            idx = work[value_col].abs().nlargest(top_n).index
            work = work.loc[idx]
            warnings.append(f"Showing top {top_n} factors by absolute impact.")

        # Sort ascending so the largest bar sits at the top in a horizontal chart
        work = work.reindex(work[value_col].abs().sort_values(ascending=True).index)

        colors = [
            params["positive_color"] if v >= 0 else params["negative_color"]
            for v in work[value_col]
        ]

        text_vals = work[value_col].round(2).astype(str) if params["show_values"] else None
        text_pos  = "outside" if params["show_values"] else None

        fig = go.Figure(go.Bar(
            x=work[value_col],
            y=work[label_col].astype(str),
            orientation="h",
            marker_color=colors,
            text=text_vals,
            textposition=text_pos,
        ))

        if params["show_baseline"]:
            fig.add_vline(x=0, line_width=1, line_color="#2c3e50")

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            xaxis_title="Impact",
            yaxis_title="Factor",
        )
        return BuildResult(figure=fig, warnings=warnings)

    def compare(self, df, columns, scenarios):
        from cerp_viz.charts.compare_utils import overlay_scenarios
        return overlay_scenarios(self, df, columns, scenarios)


registry.register(TornadoChart)
