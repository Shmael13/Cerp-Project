from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_NODE_COLORS = {
    "Blue":   "rgba(31, 119, 180, 0.8)",
    "Green":  "rgba(44, 160, 44, 0.8)",
    "Orange": "rgba(255, 127, 14, 0.8)",
    "Purple": "rgba(148, 103, 189, 0.8)",
    "Red":    "rgba(214, 39, 40, 0.8)",
}


class SankeyChart(BaseVisualization):
    name: ClassVar[str] = "Sankey Diagram"
    description: ClassVar[str] = "Visualize flows and transfers between categories."

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("source", "categorical", "Source"),
            ColumnSpec("target", "categorical", "Target"),
            ColumnSpec("value",  "numeric",     "Flow Value"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("value_multiplier", "slider",       "Value Multiplier",         1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("top_n",            "number_input", "Top N Flows (0 = all)",    0,
                           {"min": 0, "max": 200, "step": 1}, category="Data"),
            AssumptionSpec("min_flow_pct",     "slider",       "Hide Flows Below (% of total)", 0.0,
                           {"min": 0.0, "max": 20.0, "step": 0.5}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("link_opacity",     "slider",       "Link Opacity",              0.5,
                           {"min": 0.05, "max": 1.0, "step": 0.05}, category="Display"),
            AssumptionSpec("node_thickness",   "slider",       "Node Thickness",            20,
                           {"min": 5, "max": 60, "step": 1}, category="Display"),
            AssumptionSpec("node_color",       "selectbox",    "Node Color",                "Blue",
                           {"choices": list(_NODE_COLORS)}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        source_col = columns["source"]
        target_col = columns["target"]
        value_col  = columns["value"]
        warnings: list[str] = []

        missing = [role for role, col in [("Source", source_col), ("Target", target_col), ("Value", value_col)]
                   if not col or col not in df.columns]
        if missing:
            raise ValueError(f"Required column(s) not mapped: {', '.join(missing)}. "
                             "Please select columns in the sidebar.")

        if source_col == target_col:
            raise ValueError("Source and Target must be different columns.")

        work = df[[source_col, target_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce") * params["value_multiplier"]

        # Drop rows with NaN nodes or non-positive flows — Sankey requires both
        before = len(work)
        work = work.dropna(subset=[source_col, target_col, value_col])
        work = work[work[value_col] > 0]
        dropped = before - len(work)
        if dropped:
            warnings.append(
                f"Dropped {dropped} row(s) with missing node names or non-positive flow values "
                f"(Sankey requires non-empty nodes and positive flows)."
            )

        if work.empty:
            raise ValueError("No valid flows remain after removing missing/non-positive values.")

        # Coerce node labels to string to avoid NaN-node or mixed-type issues
        work[source_col] = work[source_col].astype(str)
        work[target_col] = work[target_col].astype(str)

        work = work.groupby([source_col, target_col], as_index=False)[value_col].sum()
        work = work[work[value_col] > 0]

        # Min-flow-pct threshold: drop flows that are too small relative to total
        min_pct = float(params["min_flow_pct"])
        if min_pct > 0.0:
            total_flow = work[value_col].sum()
            cutoff = total_flow * (min_pct / 100)
            before_t = len(work)
            work = work[work[value_col] >= cutoff]
            removed = before_t - len(work)
            if removed:
                warnings.append(f"Dropped {removed} minor flow(s) below {min_pct:.1f}% of total ({cutoff:.2f}).")

        # Top-N after threshold
        top_n = int(params["top_n"])
        if top_n > 0:
            work = work.nlargest(top_n, value_col)
            warnings.append(f"Showing top {top_n} flows by value.")

        if work.empty:
            raise ValueError("All flows were filtered out. Lower the thresholds.")

        if params["value_multiplier"] != 1.0:
            warnings.append(f"Flow values multiplied by {params['value_multiplier']}×.")

        all_nodes = list(pd.unique(work[[source_col, target_col]].values.ravel("K")))
        node_idx  = {n: i for i, n in enumerate(all_nodes)}

        node_color = _NODE_COLORS[params["node_color"]]
        opacity    = params["link_opacity"]

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=int(params["node_thickness"]),
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=[node_color] * len(all_nodes),
            ),
            link=dict(
                source=work[source_col].map(node_idx).tolist(),
                target=work[target_col].map(node_idx).tolist(),
                value=work[value_col].tolist(),
                color=[f"rgba(180,180,180,{opacity})"] * len(work),
            ),
        ))
        fig.update_layout(template="plotly_white")
        return BuildResult(figure=fig, warnings=warnings)


registry.register(SankeyChart)
