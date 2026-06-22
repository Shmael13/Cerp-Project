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
    "Antique": px.colors.qualitative.Antique,
}


class LineChart(BaseVisualization):
    name: ClassVar[str] = "Line Chart"
    description: ClassVar[str] = "Track trends over time or any ordered dimension."
    supports_comparison: ClassVar[bool] = True

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",      "any",         "X Axis (time or ordered)"),
            ColumnSpec("y",      "numeric",      "Y Axis (value)"),
            ColumnSpec("series", "categorical",  "Series / Group (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("y_multiplier",    "slider",    "Y Multiplier",               1.0,
                           {"min": 0.1, "max": 10.0, "step": 0.1}, category="Data"),
            AssumptionSpec("y_offset",        "number_input", "Y Baseline Offset",       0.0,
                           {"min": None, "max": None, "step": 1.0}, category="Data"),
            AssumptionSpec("rolling_window",  "slider",    "Rolling Average Window",      1,
                           {"min": 1, "max": 30, "step": 1}, category="Data"),
            AssumptionSpec("pct_change",      "toggle",    "Show % Change (period-over-period)", False,
                           {}, category="Data"),
            AssumptionSpec("cumulative",      "toggle",    "Cumulative Sum",              False,
                           {}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("mode",            "selectbox", "Line Mode",                  "lines+markers",
                           {"choices": ["lines", "lines+markers", "markers"]}, category="Display"),
            AssumptionSpec("show_area",       "toggle",    "Fill Area Under Line",        False,
                           {}, category="Display"),
            AssumptionSpec("color_scheme",    "selectbox", "Color Scheme",               "Plotly",
                           {"choices": list(_COLOR_SEQUENCES)}, category="Display"),
            # ── Statistics ────────────────────────────────────────────────────
            AssumptionSpec("show_ci_band",   "toggle",    "±1σ Confidence Band",          False,
                           {}, category="Statistics"),
            AssumptionSpec("ci_window",      "slider",    "CI Band Rolling Window",         5,
                           {"min": 2, "max": 30, "step": 1}, category="Statistics"),
            AssumptionSpec("show_anomalies", "toggle",    "Highlight Anomalies (>2σ)",     False,
                           {}, category="Statistics"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        x_col      = columns["x"]
        y_col      = columns["y"]
        series_col = columns.get("series")
        warnings: list[str] = []

        work = df.copy()
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[y_col])
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values in '{y_col}'.")

        if params["y_multiplier"] != 1.0:
            work[y_col] = work[y_col] * params["y_multiplier"]
            warnings.append(f"'{y_col}' values multiplied by {params['y_multiplier']}×.")

        offset = float(params["y_offset"])
        if offset != 0.0:
            work[y_col] = work[y_col] + offset
            warnings.append(f"Added baseline offset of {offset:+.2f} to '{y_col}'.")

        window = int(params["rolling_window"])
        if window > 1:
            if series_col:
                work[y_col] = (
                    work.groupby(series_col)[y_col]
                    .transform(lambda s: s.rolling(window, min_periods=1).mean())
                )
            else:
                work[y_col] = work[y_col].rolling(window, min_periods=1).mean()
            warnings.append(f"Rolling average applied (window = {window}).")

        if params["pct_change"]:
            if series_col:
                work[y_col] = (
                    work.groupby(series_col)[y_col]
                    .transform(lambda s: s.pct_change() * 100)
                )
            else:
                work[y_col] = work[y_col].pct_change() * 100
            before_pct = len(work)
            work = work.dropna(subset=[y_col])
            dropped_pct = before_pct - len(work)
            if dropped_pct:
                warnings.append(f"Period-over-period % change: dropped {dropped_pct} leading row(s) with no prior period.")
            else:
                warnings.append("Values converted to period-over-period % change.")

        if params["cumulative"]:
            if series_col:
                work[y_col] = work.groupby(series_col)[y_col].cumsum()
            else:
                work[y_col] = work[y_col].cumsum()
            warnings.append(f"Cumulative sum applied to '{y_col}'.")

        fig = px.line(
            work, x=x_col, y=y_col,
            color=series_col,
            color_discrete_sequence=_COLOR_SEQUENCES[params["color_scheme"]],
            template="plotly_white",
        )
        fig.update_traces(mode=params["mode"])
        if params["show_area"]:
            fig.update_traces(fill="tozeroy")

        # ── Statistical overlays (only on single-series, non-transformed) ─────
        if not series_col and not params.get("pct_change") and not params.get("cumulative"):
            y_vals = work[y_col].reset_index(drop=True)

            if params.get("show_ci_band"):
                from cerp_viz.core.stats import ci_band
                import plotly.graph_objects as go
                ci_w   = int(params.get("ci_window", 5))
                bands  = ci_band(y_vals, window=ci_w)
                x_vals = list(range(len(y_vals))) if x_col not in work.columns else list(work[x_col])
                fig.add_trace(go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=list(bands["upper"]) + list(bands["lower"][::-1]),
                    fill="toself",
                    fillcolor="rgba(100,150,255,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True, name="±1σ band",
                ))

            if params.get("show_anomalies"):
                from cerp_viz.core.stats import anomaly_mask
                import plotly.graph_objects as go
                mask   = anomaly_mask(y_vals)
                x_vals = list(range(len(y_vals))) if x_col not in work.columns else list(work[x_col])
                anom_x = [x for x, m in zip(x_vals, mask) if m]
                anom_y = [y for y, m in zip(y_vals, mask) if m]
                if anom_x:
                    fig.add_trace(go.Scatter(
                        x=anom_x, y=anom_y,
                        mode="markers",
                        marker=dict(color="#e74c3c", size=10, symbol="circle-open", line=dict(width=2)),
                        showlegend=True, name=f"Anomaly (>{2}σ)",
                    ))
                    warnings.append(f"Highlighted {len(anom_x)} anomalous point(s) (>2σ from mean).")

        return BuildResult(figure=fig, warnings=warnings)

    def compare(self, df, columns, scenarios):
        from cerp_viz.charts.compare_utils import overlay_scenarios
        return overlay_scenarios(self, df, columns, scenarios)


registry.register(LineChart)
