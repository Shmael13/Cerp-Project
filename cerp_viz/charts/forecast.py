from __future__ import annotations

from datetime import date as _date
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
            AssumptionSpec("trend_type",       "selectbox",   "Trend type",          "Linear",
                           {"choices": ["Linear", "Polynomial", "Moving Avg"]}, category="Data"),
            AssumptionSpec("forecast_periods", "slider",      "Forecast periods",    12,
                           {"min": 1, "max": 60, "step": 1},                  category="Data"),
            AssumptionSpec("confidence_pct",   "slider",      "Confidence %",        95,
                           {"min": 50, "max": 99, "step": 5},                 category="Data"),
            AssumptionSpec("ma_window",        "slider",      "Moving avg window",   7,
                           {"min": 2, "max": 52, "step": 1},                  category="Data"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        date_col = columns["date"]
        val_col  = columns["value"]
        warnings: list[str] = []

        work = df[[date_col, val_col]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[val_col]  = pd.to_numeric(work[val_col],  errors="coerce")
        before = len(work)
        work = work.dropna().sort_values(date_col).reset_index(drop=True)
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

        n_fore    = int(params["forecast_periods"])
        z         = _z(float(params["confidence_pct"]))
        trend     = params["trend_type"]
        ma_window = int(params["ma_window"])

        x_fut_ord   = x_ord[-1] + np.arange(1, n_fore + 1) * period
        hist_dates  = _from_ordinal(x_ord)
        future_dates = _from_ordinal(x_fut_ord)

        y_fit, y_fore, residuals = _fit_and_forecast(x_ord, y, x_fut_ord, trend, ma_window)

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


registry.register(ForecastChart)
