from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class FunnelChart(BaseVisualization):
    name: ClassVar[str] = "Funnel Chart"
    description: ClassVar[str] = "Show stage-by-stage conversion or reduction through a process."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("stage", "categorical", "Stage / Step"),
            ColumnSpec("value", "numeric",     "Value / Count"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("aggregation",    "selectbox", "Aggregation",                    "sum",
                           {"choices": ["sum", "mean", "count", "max"]}, category="Data"),
            AssumptionSpec("multiplier",     "slider",    "Value Multiplier",                1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("sort_order",     "selectbox", "Stage Order",                    "By Value (desc)",
                           {"choices": ["By Value (desc)", "By Value (asc)", "As in Data"]}, category="Data"),
            AssumptionSpec("min_stage_pct",  "slider",    "Drop Stages Below (% of first)", 0.0,
                           {"min": 0.0, "max": 50.0, "step": 1.0}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("show_pct",       "toggle",    "Show Conversion %",               True, {}, category="Display"),
            AssumptionSpec("connector",      "toggle",    "Show Connector Lines",             True, {}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        stage_col = columns["stage"]
        value_col = columns["value"]
        warnings: list[str] = []

        work = df[[stage_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

        before = len(work)
        work = work.dropna()
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        work = work.groupby(stage_col, as_index=False)[value_col].agg(params["aggregation"])

        if params["multiplier"] != 1.0:
            work[value_col] = work[value_col] * params["multiplier"]
            warnings.append(f"Values multiplied by {params['multiplier']}×.")

        sort = params["sort_order"]
        if sort == "By Value (desc)":
            work = work.sort_values(value_col, ascending=False)
        elif sort == "By Value (asc)":
            work = work.sort_values(value_col, ascending=True)

        # Drop stages that fall below min_stage_pct % of the first (largest) stage
        min_pct = float(params["min_stage_pct"])
        if min_pct > 0.0 and len(work) > 0:
            first_val = work[value_col].iloc[0]
            if first_val > 0:
                cutoff = first_val * (min_pct / 100)
                before_s = len(work)
                work = work[work[value_col] >= cutoff]
                removed = before_s - len(work)
                if removed:
                    warnings.append(f"Dropped {removed} stage(s) below {min_pct:.0f}% of top stage ({cutoff:.2f}).")

        text_info = "value+percent previous" if params["show_pct"] else "value"

        fig = go.Figure(go.Funnel(
            y=work[stage_col].astype(str),
            x=work[value_col],
            textinfo=text_info,
            connector={"visible": params["connector"]},
        ))
        fig.update_layout(template="plotly_white")
        return BuildResult(figure=fig, warnings=warnings)

    def compare(self, df, columns, scenarios):
        from cerp_viz.charts.compare_utils import overlay_scenarios
        return overlay_scenarios(self, df, columns, scenarios)


registry.register(FunnelChart)
