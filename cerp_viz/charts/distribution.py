from __future__ import annotations
import copy
from typing import Any, ClassVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_COLOR_SEQUENCES = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Dark":   px.colors.qualitative.Dark24,
    "Bold":   px.colors.qualitative.Bold,
    "Vivid":  px.colors.qualitative.Vivid,
}


class DistributionChart(BaseVisualization):
    name: ClassVar[str] = "Distribution"
    description: ClassVar[str] = "Explore the statistical spread of a numeric variable via histogram, box, or violin."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("value", "numeric",     "Value"),
            ColumnSpec("group", "categorical", "Group By (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("multiplier",     "slider",    "Value Multiplier",     1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("remove_outliers","toggle",    "Remove IQR Outliers",  False,
                           {}, category="Data"),
            AssumptionSpec("iqr_factor",     "slider",    "IQR Fence Factor",     1.5,
                           {"min": 1.0, "max": 5.0, "step": 0.25}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("chart_type",     "selectbox", "Chart Type",           "Histogram",
                           {"choices": ["Histogram", "Box Plot", "Violin"]}, category="Display"),
            AssumptionSpec("nbins",          "slider",    "Histogram Bins",       30,
                           {"min": 5, "max": 150, "step": 5}, category="Display"),
            AssumptionSpec("show_mean",      "toggle",    "Show Mean Line",       True,  {}, category="Display"),
            AssumptionSpec("show_median",    "toggle",    "Show Median Line",     False, {}, category="Display"),
            AssumptionSpec("color_scheme",   "selectbox", "Color Scheme",         "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
            # ── Statistics ────────────────────────────────────────────────────
            AssumptionSpec("show_normal_fit","toggle",    "Overlay Normal Fit Curve", False,
                           {}, category="Statistics"),
            AssumptionSpec("show_skew_info", "toggle",   "Annotate Skewness & Kurtosis", True,
                           {}, category="Statistics"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        value_col = columns["value"]
        group_col = columns.get("group")
        warnings: list[str] = []

        work = df.copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[value_col])
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values in '{value_col}'.")

        if params["multiplier"] != 1.0:
            work[value_col] = work[value_col] * params["multiplier"]
            warnings.append(f"'{value_col}' values multiplied by {params['multiplier']}×.")

        if params["remove_outliers"]:
            k = float(params["iqr_factor"])
            q1, q3 = work[value_col].quantile(0.25), work[value_col].quantile(0.75)
            iqr = q3 - q1
            before_iqr = len(work)
            work = work[(work[value_col] >= q1 - k * iqr) & (work[value_col] <= q3 + k * iqr)]
            removed = before_iqr - len(work)
            if removed:
                warnings.append(f"Removed {removed} outlier row(s) outside IQR × {k} fence "
                                 f"([{q1 - k*iqr:.2f}, {q3 + k*iqr:.2f}]).")

        colors = _COLOR_SEQUENCES[params["color_scheme"]]
        chart_type = params["chart_type"]

        if chart_type == "Histogram":
            fig = px.histogram(
                work, x=value_col, color=group_col,
                nbins=int(params["nbins"]),
                color_discrete_sequence=colors,
                barmode="overlay",
                opacity=0.75,
                template="plotly_white",
            )
            if params["show_mean"]:
                mean_val = work[value_col].mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="#e74c3c",
                              annotation_text=f"Mean: {mean_val:.2f}",
                              annotation_position="top right")
                warnings.append(f"Mean of '{value_col}': {mean_val:.2f}.")

            if params["show_median"]:
                median_val = work[value_col].median()
                fig.add_vline(x=median_val, line_dash="dot", line_color="#3498db",
                              annotation_text=f"Median: {median_val:.2f}",
                              annotation_position="bottom right")
                warnings.append(f"Median of '{value_col}': {median_val:.2f}.")

        elif chart_type == "Box Plot":
            fig = px.box(
                work, x=group_col, y=value_col,
                color=group_col,
                color_discrete_sequence=colors,
                template="plotly_white",
                points="outliers",
            )

        else:  # Violin
            fig = px.violin(
                work, x=group_col, y=value_col,
                color=group_col,
                color_discrete_sequence=colors,
                box=True,
                template="plotly_white",
            )

        # ── Statistical overlays (histogram only, no group) ──────────────────
        if chart_type == "Histogram" and not group_col:
            from cerp_viz.core.stats import normality_stats

            ns = normality_stats(work[value_col])

            if ns and params.get("show_normal_fit") and "x_fit" in ns:
                # Scale density curve to match histogram counts
                n_rows = len(work)
                bin_width = (work[value_col].max() - work[value_col].min()) / int(params["nbins"])
                scale = n_rows * bin_width if bin_width > 0 else 1.0
                fig.add_trace(go.Scatter(
                    x=ns["x_fit"],
                    y=ns["y_fit"] * scale,
                    mode="lines",
                    line=dict(color="#e74c3c", width=2, dash="solid"),
                    name="Normal fit",
                    showlegend=True,
                ))

            if ns and params.get("show_skew_info"):
                skew = ns.get("skewness", 0.0)
                kurt = ns.get("kurtosis", 0.0)
                direction = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "symmetric"
                tail = "heavy-tailed" if kurt > 1 else "light-tailed" if kurt < -1 else "normal-tailed"
                shapiro_text = ""
                if "shapiro_p" in ns:
                    p = ns["shapiro_p"]
                    shapiro_text = f"  Normality (Shapiro-Wilk): p={'<0.001' if p < 0.001 else f'{p:.3f}'}"
                annotation = (
                    f"Skewness: {skew:+.2f} ({direction})<br>"
                    f"Kurtosis: {kurt:+.2f} ({tail})"
                    + (f"<br>{shapiro_text}" if shapiro_text else "")
                )
                fig.add_annotation(
                    xref="paper", yref="paper", x=0.99, y=0.99,
                    text=annotation, showarrow=False,
                    font=dict(size=11, color="#333"),
                    bgcolor="rgba(255,255,255,0.8)",
                    borderpad=6, xanchor="right", yanchor="top",
                )

        return BuildResult(figure=fig, warnings=warnings)

    def compare(
        self,
        df: pd.DataFrame,
        columns: dict[str, str | None],
        scenarios: dict[str, dict[str, Any]],
    ) -> BuildResult:
        """
        Override the generic overlay to always render as Box Plot for comparison —
        histograms from different scenarios with different bin ranges overlap badly.
        """
        from cerp_viz.charts.compare_utils import _PALETTE

        combined = go.Figure()
        all_warnings: list[str] = []
        value_col = columns["value"]

        for i, (name, params) in enumerate(scenarios.items()):
            box_params = dict(params, chart_type="Box Plot")
            result = self.build(df, columns, box_params)
            color = _PALETTE[i % len(_PALETTE)]

            for trace in result.figure.data:
                t = copy.deepcopy(trace)
                t.name = name
                t.legendgroup = name
                if hasattr(t, "marker") and t.marker is not None:
                    t.marker.color = color
                if hasattr(t, "line") and t.line is not None:
                    t.line.color = color
                combined.add_trace(t)
            all_warnings.extend(f"[{name}] {w}" for w in result.warnings)

        if all_warnings and not any("Box Plot" in w for w in all_warnings):
            all_warnings.insert(0, "Chart type overridden to Box Plot for scenario comparison.")

        combined.update_layout(
            template="plotly_white",
            showlegend=True,
            legend_title_text="Scenario",
            yaxis_title=value_col,
        )
        return BuildResult(figure=combined, warnings=all_warnings)


registry.register(DistributionChart)
