from typing import Any, ClassVar

import pandas as pd
import plotly.express as px

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


class ScatterChart(BaseVisualization):
    name: ClassVar[str] = "Scatter Plot"
    description: ClassVar[str] = "Explore relationships between two numeric variables."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",     "numeric",     "X Axis"),
            ColumnSpec("y",     "numeric",     "Y Axis"),
            ColumnSpec("size",  "numeric",     "Bubble Size (optional)",  required=False),
            ColumnSpec("color", "categorical", "Color Group (optional)",  required=False),
            ColumnSpec("label", "any",         "Hover Label (optional)",  required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("x_multiplier",    "slider",    "X Multiplier",        1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("y_multiplier",    "slider",    "Y Multiplier",        1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("remove_outliers", "toggle",    "Remove Outliers (IQR)", False,
                           {}, category="Data"),
            AssumptionSpec("iqr_multiplier",  "slider",    "IQR Fence Multiplier", 1.5,
                           {"min": 1.0, "max": 5.0, "step": 0.25}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("trendline",       "selectbox", "Trendline",           "None",
                           {"choices": ["None", "ols", "lowess"]}, category="Display"),
            AssumptionSpec("point_size",      "slider",    "Point Size",           8,
                           {"min": 3, "max": 30, "step": 1}, category="Display"),
            AssumptionSpec("opacity",         "slider",    "Opacity",              0.8,
                           {"min": 0.1, "max": 1.0, "step": 0.05}, category="Display"),
            AssumptionSpec("log_x",           "toggle",    "Log Scale (X)",        False,
                           {}, category="Display"),
            AssumptionSpec("log_y",           "toggle",    "Log Scale (Y)",        False,
                           {}, category="Display"),
            AssumptionSpec("color_scheme",    "selectbox", "Color Scheme",        "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
            # ── Statistics ────────────────────────────────────────────────────
            AssumptionSpec("show_r2",         "toggle",    "Annotate R² on trendline", True,
                           {}, category="Statistics"),
            AssumptionSpec("show_ci_band",    "toggle",    "95% Confidence Band",  False,
                           {}, category="Statistics"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        x_col     = columns["x"]
        y_col     = columns["y"]
        size_col  = columns.get("size")
        color_col = columns.get("color")
        label_col = columns.get("label")
        warnings: list[str] = []

        work = df.copy()
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce") * params["x_multiplier"]
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce") * params["y_multiplier"]

        before = len(work)
        work = work.dropna(subset=[x_col, y_col])
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values in '{x_col}' or '{y_col}'.")

        if params["x_multiplier"] != 1.0:
            warnings.append(f"'{x_col}' values multiplied by {params['x_multiplier']}×.")
        if params["y_multiplier"] != 1.0:
            warnings.append(f"'{y_col}' values multiplied by {params['y_multiplier']}×.")

        if params["remove_outliers"]:
            k = float(params["iqr_multiplier"])
            before_iqr = len(work)
            for col in [x_col, y_col]:
                q1, q3 = work[col].quantile(0.25), work[col].quantile(0.75)
                iqr = q3 - q1
                work = work[(work[col] >= q1 - k * iqr) & (work[col] <= q3 + k * iqr)]
            removed = before_iqr - len(work)
            if removed:
                warnings.append(f"Removed {removed} outlier row(s) using IQR × {k} fence.")

        # px.scatter requires strictly positive values for size encoding
        if size_col:
            work[size_col] = pd.to_numeric(work[size_col], errors="coerce").abs()
            neg_before = len(work)
            work = work.dropna(subset=[size_col])
            work = work[work[size_col] > 0]
            size_dropped = neg_before - len(work)
            if size_dropped:
                warnings.append(
                    f"Dropped {size_dropped} row(s) where '{size_col}' was zero, negative, or missing "
                    f"(bubble size requires positive values)."
                )

        trendline = None if params["trendline"] == "None" else params["trendline"]

        fig = px.scatter(
            work, x=x_col, y=y_col,
            color=color_col,
            size=size_col,
            hover_name=label_col,
            trendline=trendline,
            opacity=params["opacity"],
            log_x=params["log_x"],
            log_y=params["log_y"],
            color_discrete_sequence=_COLOR_SEQUENCES[params["color_scheme"]],
            template="plotly_white",
        )
        if not size_col:
            # Selector avoids applying marker_size to trendline line traces
            fig.update_traces(marker_size=params["point_size"], selector={"mode": "markers"})

        if params["log_x"]:
            warnings.append("X axis is on a log scale.")
        if params["log_y"]:
            warnings.append("Y axis is on a log scale.")

        # ── Statistical overlays ──────────────────────────────────────────────
        if trendline and params.get("show_r2"):
            from cerp_viz.core.stats import ols_fit, significance_stars
            fit = ols_fit(work[x_col].values, work[y_col].values)
            if fit:
                stars = significance_stars(fit.get("p_value", float("nan")))
                label = f"R² = {fit['r2']:.3f} {stars}  (n={fit['n']})"
                fig.add_annotation(
                    xref="paper", yref="paper", x=0.01, y=0.99,
                    text=label, showarrow=False,
                    font=dict(size=12, color="#333"),
                    bgcolor="rgba(255,255,255,0.75)",
                    borderpad=4, xanchor="left", yanchor="top",
                )

        if params.get("show_ci_band") and not color_col:
            from cerp_viz.core.stats import ols_fit
            import plotly.graph_objects as go
            fit = ols_fit(work[x_col].values, work[y_col].values)
            if fit and len(fit.get("x", [])) >= 3:
                x_s   = fit["x"]
                y_hat = fit["y_pred"]
                n, se = fit["n"], fit.get("se", 0)
                # Simple OLS prediction interval ± 1.96 × residual SE
                ss_xx = ((x_s - x_s.mean()) ** 2).sum()
                lev   = 1 / n + (x_s - x_s.mean()) ** 2 / ss_xx if ss_xx else 0
                from cerp_viz.core.stats import ols_fit as _f  # noqa
                resid_se = float(
                    ((fit["y"] - y_hat) ** 2).sum() / max(n - 2, 1)
                ) ** 0.5
                band = 1.96 * resid_se * (1 + lev) ** 0.5
                idx  = x_s.argsort()
                x_ord, y_u, y_l = x_s[idx], (y_hat + band)[idx], (y_hat - band)[idx]
                fig.add_trace(go.Scatter(
                    x=list(x_ord) + list(x_ord[::-1]),
                    y=list(y_u)   + list(y_l[::-1]),
                    fill="toself",
                    fillcolor="rgba(100,100,200,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True, name="95% CI",
                ))

        return BuildResult(figure=fig, warnings=warnings)

    def compare(self, df, columns, scenarios):
        from cerp_viz.charts.compare_utils import overlay_scenarios
        return overlay_scenarios(self, df, columns, scenarios)


registry.register(ScatterChart)
