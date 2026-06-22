from typing import Any, ClassVar

import pandas as pd
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_COLORSCALES = ["RdBu", "Viridis", "Blues", "Oranges", "Tealgrn", "Plasma", "Turbo", "YlOrRd"]


class TreemapChart(BaseVisualization):
    name: ClassVar[str] = "Treemap"
    description: ClassVar[str] = "Hierarchical part-to-whole — size encodes value, color encodes a second metric."
    supports_comparison: ClassVar[bool] = False

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("level1", "categorical", "Top-Level Category"),
            ColumnSpec("level2", "categorical", "Sub-Category (optional)", required=False),
            ColumnSpec("level3", "categorical", "Sub-Sub-Category (optional)", required=False),
            ColumnSpec("value",  "numeric",     "Size (value)"),
            ColumnSpec("color",  "numeric",     "Color Metric (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation", "selectbox",    "Aggregation",          "sum",
                           {"choices": ["sum", "mean", "count", "max", "min"]}, category="Data"),
            AssumptionSpec("top_n",       "number_input", "Show Top N Leaves (0 = all)", 0,
                           {"min": 0, "max": 200, "step": 5}, category="Data"),
            AssumptionSpec("maxdepth",    "selectbox",    "Expand Depth",         "All",
                           {"choices": ["All", "1", "2", "3"]}, category="Display"),
            AssumptionSpec("show_values", "toggle",       "Show Values in Labels", True,
                           {}, category="Display"),
            AssumptionSpec("show_pct",    "toggle",       "Show % of Parent",      False,
                           {}, category="Display"),
            AssumptionSpec("colorscale",  "selectbox",    "Color Scale",          "RdBu",
                           {"choices": _COLORSCALES}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        l1_col    = columns["level1"]
        l2_col    = columns.get("level2")
        l3_col    = columns.get("level3")
        val_col   = columns["value"]
        color_col = columns.get("color")
        warnings: list[str] = []

        work = df.copy()
        work[val_col] = pd.to_numeric(work[val_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[l1_col, val_col])
        if (dropped := before - len(work)):
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        neg = (work[val_col] <= 0).sum()
        if neg:
            work = work[work[val_col] > 0]
            warnings.append(f"Removed {neg} row(s) with zero or negative values (not representable in a treemap).")

        # Build path list (only non-None, non-duplicate levels)
        path_cols = []
        for c in [l1_col, l2_col, l3_col]:
            if c and c not in path_cols:
                path_cols.append(c)

        # Aggregate: group by all path cols + optional color col
        agg_group = path_cols[:]
        if color_col:
            agg_group.append(color_col)
        agg_cols = {val_col: params["aggregation"]}
        if color_col:
            agg_cols[color_col] = "mean"

        work = work.groupby(agg_group, as_index=False).agg(agg_cols)

        top_n = int(params["top_n"])
        if top_n > 0:
            top_leaves = (
                work.groupby(path_cols[-1])[val_col].sum()
                .nlargest(top_n).index
            )
            before_top = len(work)
            work = work[work[path_cols[-1]].isin(top_leaves)]
            if len(work) < before_top:
                warnings.append(f"Showing top {top_n} leaf-level items by value.")

        maxdepth_str = params["maxdepth"]
        maxdepth = None if maxdepth_str == "All" else int(maxdepth_str)

        text_template = "%{label}"
        if params["show_values"]:
            text_template += "<br>%{value:,.0f}"
        if params["show_pct"]:
            text_template += "<br>%{percentParent:.1%}"

        fig = px.treemap(
            work,
            path=[px.Constant("All")] + path_cols,
            values=val_col,
            color=color_col if color_col else val_col,
            color_continuous_scale=params["colorscale"],
            maxdepth=maxdepth,
            template="plotly_white",
        )
        fig.update_traces(
            texttemplate=text_template,
            hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>%{percentParent:.1%} of parent<extra></extra>",
        )
        fig.update_layout(margin=dict(t=30, l=5, r=5, b=5))

        return BuildResult(figure=fig, warnings=warnings)


registry.register(TreemapChart)
