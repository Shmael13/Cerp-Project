from __future__ import annotations

from datetime import date as _date
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

try:
    from scipy.stats import norm as _norm
    def _z(pct: float) -> float:
        return float(_norm.ppf((1 + pct / 100) / 2))
except ImportError:
    _Z_TABLE = {50: 0.674, 60: 0.842, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    def _z(pct: float) -> float:
        keys = sorted(_Z_TABLE)
        for i in range(len(keys) - 1):
            if keys[i] <= pct <= keys[i + 1]:
                t = (pct - keys[i]) / (keys[i + 1] - keys[i])
                return _Z_TABLE[keys[i]] + t * (_Z_TABLE[keys[i + 1]] - _Z_TABLE[keys[i]])
        return 1.96


def _to_ordinal(dates: np.ndarray) -> np.ndarray:
    return np.array([pd.Timestamp(d).toordinal() for d in dates], dtype=float)


def _from_ordinal(ordinals: np.ndarray) -> list[pd.Timestamp]:
    return [pd.Timestamp(_date.fromordinal(int(o))) for o in ordinals]


def _poly_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return np.polyfit(x, y, degree)


def _poly_pred(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.polyval(coeffs, x)


def _prediction_interval(
    y_pred: np.ndarray,
    x_new: np.ndarray,
    x_train: np.ndarray,
    residuals: np.ndarray,
    z: float,
    ddof: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x_train)
    se = float(np.std(residuals, ddof=min(ddof, n - 1))) or 1e-9
    x_mean = x_train.mean()
    ss_xx = ((x_train - x_mean) ** 2).sum() or 1e-10
    lev = 1 / n + (x_new - x_mean) ** 2 / ss_xx
    half = z * se * np.sqrt(1 + lev)
    return y_pred - half, y_pred + half


def _ma_smooth(y: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()


def _ma_extrapolate(ma: np.ndarray, n: int) -> np.ndarray:
    x = np.arange(len(ma), dtype=float)
    coeffs = np.polyfit(x, ma, 1)
    x_fut = np.arange(len(ma), len(ma) + n, dtype=float)
    return np.polyval(coeffs, x_fut)


def _fit_and_forecast(
    x_ord: np.ndarray,
    y: np.ndarray,
    x_fut_ord: np.ndarray,
    trend_type: str,
    ma_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if trend_type == "Moving Avg":
        y_fit = _ma_smooth(y, ma_window)
        y_fore = _ma_extrapolate(y_fit, len(x_fut_ord))
        residuals = y - y_fit
    else:
        degree = 3 if trend_type == "Polynomial" else 1
        coeffs = _poly_fit(x_ord, y, degree)
        y_fit  = _poly_pred(coeffs, x_ord)
        y_fore = _poly_pred(coeffs, x_fut_ord)
        residuals = y - y_fit
    return y_fit, y_fore, residuals


# ── Seasonal decomposition (additive, no external dependencies) ───────────────

def _auto_period(gaps_days: np.ndarray) -> int:
    median_gap = float(np.median(gaps_days))
    if median_gap <= 2:
        return 7       # daily data → weekly seasonality
    if median_gap <= 10:
        return 4       # weekly data → quarterly seasonality
    if median_gap <= 45:
        return 12      # monthly data → annual seasonality
    return 4           # quarterly data → annual seasonality


def _classical_decompose(
    y: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Additive decomposition: trend (centred MA) + seasonal + residual."""
    n = len(y)
    # Centered moving average for trend
    half = period // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = y[i - half : i + half + 1].mean()

    # Fill edges with nearest valid value
    first_valid = next((i for i in range(n) if not np.isnan(trend[i])), 0)
    last_valid  = next((i for i in range(n - 1, -1, -1) if not np.isnan(trend[i])), n - 1)
    trend[:first_valid] = trend[first_valid]
    trend[last_valid + 1:] = trend[last_valid]

    detrended = y - trend

    # Seasonal: average per phase
    seasonal_proto = np.zeros(period)
    for phase in range(period):
        vals = detrended[phase::period]
        seasonal_proto[phase] = np.nanmean(vals) if len(vals) > 0 else 0.0
    seasonal_proto -= seasonal_proto.mean()   # force zero-mean

    seasonal = np.array([seasonal_proto[i % period] for i in range(n)])
    residual = y - trend - seasonal
    return trend, seasonal, residual


def _build_decomposition_fig(
    dates: list[pd.Timestamp],
    y: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray,
    val_col: str,
    period: int,
) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.06,
        row_heights=[0.35, 0.25, 0.2, 0.2],
    )

    fig.add_trace(go.Scatter(x=dates, y=y, mode="lines+markers",
                             name="Observed", line=dict(color="steelblue", width=1.5),
                             marker=dict(size=3)), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=trend, mode="lines",
                             name="Trend", line=dict(color="darkorange", width=2)),
                  row=2, col=1)

    fig.add_trace(go.Bar(x=dates, y=seasonal, name="Seasonal",
                         marker_color="mediumseagreen", opacity=0.7),
                  row=3, col=1)

    zero_color = ["firebrick" if r > 0 else "steelblue" for r in residual]
    fig.add_trace(go.Bar(x=dates, y=residual, name="Residual",
                         marker_color=zero_color, opacity=0.65),
                  row=4, col=1)

    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=40),
        height=600,
        xaxis4_title="Date",
        yaxis1_title=val_col,
        yaxis2_title="Trend",
        yaxis3_title="Seasonal",
        yaxis4_title="Residual",
    )
    return fig


class ForecastChart(BaseVisualization):
    name: ClassVar[str] = "Forecast"
    description: ClassVar[str] = "Fit a trend to historical data and project it forward with a confidence band."

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("date",  "datetime", "Date column"),
            ColumnSpec("value", "numeric",  "Value column"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("view_mode",        "selectbox",    "View mode",           "Forecast",
                           {"choices": ["Forecast", "Decompose"]},               category="Data"),
            AssumptionSpec("trend_type",       "selectbox",    "Trend type",          "Linear",
                           {"choices": ["Linear", "Polynomial", "Moving Avg"]}, category="Data"),
            AssumptionSpec("forecast_periods", "slider",       "Forecast periods",    12,
                           {"min": 1, "max": 60, "step": 1},                    category="Data"),
            AssumptionSpec("confidence_pct",   "slider",       "Confidence %",        95,
                           {"min": 50, "max": 99, "step": 5},                   category="Data"),
            AssumptionSpec("ma_window",        "slider",       "Moving avg window",   7,
                           {"min": 2, "max": 52, "step": 1},                    category="Data"),
            AssumptionSpec("season_period",    "selectbox",    "Season period",       "Auto",
                           {"choices": ["Auto", "4 (Quarterly)", "7 (Weekly)",
                                        "12 (Monthly)", "52 (Weekly/year)"]},   category="Data"),
            # Prior / baseline controls
            AssumptionSpec("comparison_col",   "column_picker", "Actual / baseline column", "(none)",
                           {"dtype": "numeric"},                                 category="Data"),
            AssumptionSpec("anchor_to_last",   "toggle",       "Anchor forecast to last actual", False,
                           {},                                                   category="Data"),
            AssumptionSpec("prior_value",      "number_input", "Manual prior (anchor value)", 0.0,
                           {"min": None, "max": None, "step": 1.0},             category="Data"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        date_col     = columns["date"]
        val_col      = columns["value"]
        comp_col     = params.get("comparison_col") or None   # "(none)" → None via column_picker
        anchor_last  = bool(params.get("anchor_to_last", False))
        prior_value  = float(params.get("prior_value", 0.0))
        warnings: list[str] = []

        extra_cols = [c for c in [comp_col] if c and c in df.columns]
        work = df[[date_col, val_col] + extra_cols].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[val_col]  = pd.to_numeric(work[val_col],  errors="coerce")
        if comp_col and comp_col in work.columns:
            work[comp_col] = pd.to_numeric(work[comp_col], errors="coerce")
        before = len(work)
        work = work.dropna(subset=[date_col, val_col]).sort_values(date_col).reset_index(drop=True)
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing or unparseable values.")
        if len(work) < 4:
            return BuildResult(figure=go.Figure(), warnings=["Need at least 4 data points to forecast."])

        dates  = work[date_col].values
        y      = work[val_col].values.astype(float)
        x_ord  = _to_ordinal(dates)

        gaps   = np.diff(x_ord)
        period = float(np.median(gaps)) if len(gaps) else 1.0

        view_mode = params.get("view_mode", "Forecast")

        if view_mode == "Decompose":
            return self._build_decompose(dates, y, x_ord, gaps, params, val_col, date_col, warnings)

        # ── Forecast mode ──────────────────────────────────────────────────────
        n_fore    = int(params["forecast_periods"])
        z         = _z(float(params["confidence_pct"]))
        trend     = params["trend_type"]
        ma_window = int(params["ma_window"])

        x_fut_ord    = x_ord[-1] + np.arange(1, n_fore + 1) * period
        hist_dates   = _from_ordinal(x_ord)
        future_dates = _from_ordinal(x_fut_ord)

        y_fit, y_fore, residuals = _fit_and_forecast(x_ord, y, x_fut_ord, trend, ma_window)

        # Apply anchor / prior
        if anchor_last:
            offset  = float(y[-1]) - float(y_fore[0])
            y_fore  = y_fore + offset
            warnings.append(f"Forecast anchored to last actual value ({y[-1]:,.2f}).")
        elif prior_value != 0.0:
            offset  = float(prior_value) - float(y_fore[0])
            y_fore  = y_fore + offset
            warnings.append(f"Forecast anchored to manual prior value ({prior_value:,.2f}).")

        ddof = 4 if trend == "Polynomial" else 2
        lo_h, hi_h = _prediction_interval(y_fit,  x_ord,     x_ord, residuals, z, ddof)
        lo_f, hi_f = _prediction_interval(y_fore, x_fut_ord, x_ord, residuals, z, ddof)

        ci_label = f"{int(params['confidence_pct'])}% CI"
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_dates + hist_dates[::-1],
            y=list(hi_h) + list(lo_h[::-1]),
            fill="toself", fillcolor="rgba(99,110,250,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{ci_label} (fit)", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(hi_f) + list(lo_f[::-1]),
            fill="toself", fillcolor="rgba(239,85,59,0.14)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{ci_label} (forecast)", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=hist_dates, y=y_fit,
            mode="lines", name=f"{trend} fit",
            line=dict(color="rgb(99,110,250)", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=y_fore,
            mode="lines", name="Forecast",
            line=dict(color="rgb(239,85,59)", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=hist_dates, y=y,
            mode="markers", name=val_col,
            marker=dict(color="rgba(99,110,250,0.55)", size=5),
        ))

        # Optional: overlay comparison / actual column
        if comp_col and comp_col in work.columns:
            comp_y = work[comp_col].values.astype(float)
            fig.add_trace(go.Scatter(
                x=hist_dates, y=comp_y,
                mode="lines+markers", name=comp_col,
                line=dict(color="rgb(0,180,100)", width=2),
                marker=dict(size=4),
            ))
            warnings.append(f"Overlaying '{comp_col}' as actual / baseline.")

        fig.add_vline(
            x=hist_dates[-1].strftime("%Y-%m-%d"),
            line_dash="dot", line_color="gray",
            annotation_text="  forecast →",
            annotation_position="top right",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title=date_col,
            yaxis_title=val_col,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        warnings.append(
            f"{trend} trend · {n_fore} periods forward · {period:.0f}-day interval · {ci_label}"
        )
        return BuildResult(figure=fig, warnings=warnings)

    def _build_decompose(
        self,
        dates: np.ndarray,
        y: np.ndarray,
        x_ord: np.ndarray,
        gaps: np.ndarray,
        params: dict[str, Any],
        val_col: str,
        date_col: str,
        warnings: list[str],
    ) -> BuildResult:
        season_str = params.get("season_period", "Auto")
        if season_str == "Auto":
            period = _auto_period(gaps) if len(gaps) else 12
            warnings.append(f"Auto-detected season period: {period}")
        else:
            period = int(season_str.split()[0])

        if len(y) < period * 2:
            return BuildResult(
                figure=go.Figure(),
                warnings=[f"Need at least {period * 2} data points to decompose with period={period}."],
            )

        trend_arr, seasonal_arr, residual_arr = _classical_decompose(y, period)
        hist_dates = _from_ordinal(x_ord)

        std_resid = float(np.nanstd(residual_arr))
        std_y     = float(np.std(y)) or 1.0
        noise_pct = 100 * std_resid / std_y
        warnings.append(
            f"Additive decomposition · period={period} · "
            f"residual noise = {noise_pct:.1f}% of signal std"
        )

        fig = _build_decomposition_fig(
            hist_dates, y, trend_arr, seasonal_arr, residual_arr, val_col, period
        )
        return BuildResult(figure=fig, warnings=warnings)


registry.register(ForecastChart)
