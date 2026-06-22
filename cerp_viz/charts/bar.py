from typing import Any, ClassVar

import pandas as pd
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_COLOR_SEQUENCES = {
    "Plotly":  px.colors.qualitative.Plotly,
    "Pastel":  px.colors.qualitative.Pastel,
    "Dark":    px.colors.qualitative.Dark24,
    "Bold":    px.colors.qualitative.Bold,
    "Vivid":   px.colors.qualitative.Vivid,
}


class BarChart(BaseVisualization):
    name: ClassVar[str] = "Bar Chart"
    description: ClassVar[str] = "Compare values across categories with optional grouping."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",     "categorical", "Category (X Axis)"),
            ColumnSpec("y",     "numeric",     "Value (Y Axis)"),
            ColumnSpec("color", "categorical", "Group By (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("aggregation",    "selectbox",    "Aggregation",           "sum",
                           {"choices": ["sum", "mean", "median", "count", "max", "min"]}, category="Data"),
            AssumptionSpec("multiplier",     "slider",       "Value Multiplier",       1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("top_n",          "number_input", "Show Top N Categories (0 = all)", 0,
                           {"min": 0, "max": 100, "step": 1}, category="Data"),
            AssumptionSpec("min_value",      "number_input", "Minimum Value Threshold", 0.0,
                           {"min": 0.0, "max": 1e9, "step": 1.0}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("sort_by",        "selectbox",    "Sort By",               "Value (desc)",
                           {"choices": ["Value (desc)", "Value (asc)", "Category (A-Z)", "Category (Z-A)"]}, category="Display"),
            AssumptionSpec("orientation",    "selectbox",    "Orientation",           "Vertical",
                           {"choices": ["Vertical", "Horizontal"]}, category="Display"),
            AssumptionSpec("barmode",        "selectbox",    "Bar Mode",              "group",
                           {"choices": ["group", "stack", "relative"]}, category="Display"),
            AssumptionSpec("show_target",    "toggle",       "Show Target Line",      False,
                           {}, category="Display"),
            AssumptionSpec("target_value",   "number_input", "Target Value",          0.0,
                           {"min": None, "max": None, "step": 1.0}, category="Display"),
            AssumptionSpec("color_scheme",   "selectbox",    "Color Scheme",          "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
            # ── Statistics ────────────────────────────────────────────────────
            AssumptionSpec("show_error_bars","toggle",       "Show Error Bars (±1 SD)", False,
                           {}, category="Statistics"),
            AssumptionSpec("show_mean_line", "toggle",       "Show Grand Mean Line",    False,
                           {}, category="Statistics"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        x_col     = columns["x"]
        y_col     = columns["y"]
        color_col = columns.get("color")
        warnings: list[str] = []

        work = df.copy()
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[x_col, y_col])
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values in '{x_col}' or '{y_col}'.")

        if params["multiplier"] != 1.0:
            work[y_col] = work[y_col] * params["multiplier"]
            warnings.append(f"'{y_col}' values multiplied by {params['multiplier']}×.")

        group_cols = [x_col] + ([color_col] if color_col else [])
        work = work.groupby(group_cols, as_index=False)[y_col].agg(params["aggregation"])

        # Threshold filter
        min_val = float(params["min_value"])
        if min_val > 0:
            before_thresh = len(work)
            work = work[work[y_col].abs() >= min_val]
            removed = before_thresh - len(work)
            if removed:
                warnings.append(f"Hid {removed} category(ies) with absolute value < {min_val}.")

        # Sort
        sort = params["sort_by"]
        if sort == "Value (desc)":
            work = work.sort_values(y_col, ascending=False)
        elif sort == "Value (asc)":
            work = work.sort_values(y_col, ascending=True)
        elif sort == "Category (A-Z)":
            work = work.sort_values(x_col)
        else:
            work = work.sort_values(x_col, ascending=False)

        # Top-N filter (applied after sort so it respects sort order)
        top_n = int(params["top_n"])
        if top_n > 0:
            # Keep top-N by absolute value across all group values
            top_cats = (
                work.groupby(x_col)[y_col].sum().abs()
                .nlargest(top_n).index
            )
            before_top = len(work)
            work = work[work[x_col].isin(top_cats)]
            if len(work) < before_top:
                warnings.append(f"Showing top {top_n} categories by aggregated absolute value.")

        orient = "h" if params["orientation"] == "Horizontal" else "v"
        x, y = (y_col, x_col) if orient == "h" else (x_col, y_col)

        fig = px.bar(
            work, x=x, y=y,
            color=color_col,
            barmode=params["barmode"],
            orientation=orient,
            color_discrete_sequence=_COLOR_SEQUENCES[params["color_scheme"]],
            template="plotly_white",
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)

        if params["show_target"]:
            target = float(params["target_value"])
            if orient == "v":
                fig.add_hline(y=target, line_dash="dash", line_color="#e74c3c",
                              annotation_text=f"Target: {target:,.1f}",
                              annotation_position="top right")
            else:
                fig.add_vline(x=target, line_dash="dash", line_color="#e74c3c",
                              annotation_text=f"Target: {target:,.1f}",
                              annotation_position="top right")

        # ── Statistical overlays ──────────────────────────────────────────────
        if params.get("show_error_bars") and not color_col:
            # Compute per-category std dev from *pre-aggregated* df
            raw_group = df.copy()
            raw_group[y_col] = pd.to_numeric(raw_group[y_col], errors="coerce")
            err = raw_group.groupby(x_col)[y_col].std().reindex(work[x_col]).fillna(0)
            if orient == "v":
                fig.update_traces(error_y=dict(type="data", array=list(err), visible=True))
            else:
                fig.update_traces(error_x=dict(type="data", array=list(err), visible=True))
            warnings.append("Error bars show ±1 SD of raw values per category.")

        if params.get("show_mean_line"):
            grand_mean = float(work[y_col].mean())
            if orient == "v":
                fig.add_hline(y=grand_mean, line_dash="dot", line_color="#555",
                              annotation_text=f"Mean: {grand_mean:,.1f}",
                              annotation_position="top left")
            else:
                fig.add_vline(x=grand_mean, line_dash="dot", line_color="#555",
                              annotation_text=f"Mean: {grand_mean:,.1f}",
                              annotation_position="top left")

        return BuildResult(figure=fig, warnings=warnings)

    def compare(self, df, columns, scenarios):
        from cerp_viz.charts.compare_utils import overlay_scenarios
        return overlay_scenarios(self, df, columns, scenarios)


registry.register(BarChart)
