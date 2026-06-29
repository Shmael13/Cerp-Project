from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_AGG_FUNCS   = {"Sum": "sum", "Mean": "mean", "Count": "count", "Max": "max"}


def _build_calendar_matrix(
    daily: pd.DataFrame, year: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    jan1  = pd.Timestamp(year=year, month=1, day=1)
    dec31 = pd.Timestamp(year=year, month=12, day=31)

    full_range = pd.date_range(jan1, dec31, freq="D")
    grid = pd.Series(np.nan, index=full_range, dtype=float)
    for _, row in daily.iterrows():
        d = row["date"]
        if d in grid.index:
            grid[d] = row["value"]

    start_dow = jan1.dayofweek
    n_weeks   = (len(full_range) + start_dow + 6) // 7

    matrix      = np.full((7, n_weeks), np.nan)
    date_labels = np.full((7, n_weeks), "", dtype=object)

    for i, date in enumerate(full_range):
        col = (i + start_dow) // 7
        row = date.dayofweek
        matrix[row, col]      = grid[date]
        date_labels[row, col] = date.strftime("%b %d, %Y")

    week_labels = [""] * n_weeks
    for date in full_range:
        if date.day == 1:
            week_col = (date.dayofyear - 1 + start_dow) // 7
            week_labels[week_col] = date.strftime("%b")

    return matrix, date_labels, week_labels


def _build_yoy_matrix(
    daily: pd.DataFrame, year: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a 7×53 matrix aligned on week-of-year (Jan 1 = week 0).
    This ensures columns are visually aligned across different years."""
    data = daily[daily["date"].dt.year == year].copy()

    matrix = np.full((7, 53), np.nan)
    labels = np.full((7, 53), "", dtype=object)

    for _, row in data.iterrows():
        d = row["date"]
        doy = d.timetuple().tm_yday - 1   # 0-based
        col = min(doy // 7, 52)
        dow = d.dayofweek
        matrix[dow, col] = row["value"]
        labels[dow, col] = d.strftime("%b %d")

    # Month-start labels for the x axis
    week_labels = [""] * 53
    for month in range(1, 13):
        try:
            first = pd.Timestamp(year=year, month=month, day=1)
            col   = min((first.timetuple().tm_yday - 1) // 7, 52)
            if not week_labels[col]:
                week_labels[col] = first.strftime("%b")
        except Exception:
            pass

    return matrix, labels, week_labels


def _build_yoy_fig(
    daily: pd.DataFrame,
    years: list[int],
    color_scale: str,
    agg_label: str,
    val_col: str,
) -> go.Figure:
    n = len(years)
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=[str(y) for y in years],
        shared_xaxes=True,
        vertical_spacing=max(0.02, 0.15 / n),
    )

    # Shared colour scale across years so the same colour means the same value
    all_vals = daily["value"].dropna()
    zmin = float(all_vals.min()) if len(all_vals) else 0.0
    zmax = float(all_vals.max()) if len(all_vals) else 1.0

    last_week_labels: list[str] = [""] * 53

    for i, year in enumerate(years):
        matrix, cell_labels, week_labels = _build_yoy_matrix(daily, year)
        if any(week_labels):
            last_week_labels = week_labels

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=list(range(53)),
                y=_DOW_LABELS,
                colorscale=color_scale,
                zmin=zmin,
                zmax=zmax,
                showscale=(i == 0),
                hoverongaps=False,
                xgap=1,
                ygap=1,
                customdata=cell_labels,
                hovertemplate=f"{year}: %{{customdata}}<br>Value: %{{z:.2f}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(showgrid=False, row=i + 1, col=1)

    # Apply month labels on the last (bottom) subplot
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(53)),
        ticktext=last_week_labels,
        showgrid=False,
        row=n,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        title=f"{val_col} — Year-over-Year ({agg_label} per day)",
        height=max(300, 170 * n),
        margin=dict(l=10, r=10, t=60, b=30),
    )
    return fig


class CalendarHeatmap(BaseVisualization):
    name: ClassVar[str] = "Calendar Heatmap"
    description: ClassVar[str] = "GitHub-style daily value calendar — spot patterns by day and week."

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("date",  "datetime", "Date"),
            ColumnSpec("value", "numeric",  "Value"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("view_mode", "selectbox", "View mode", "Single Year",
                           {"choices": ["Single Year", "Year-over-Year"]}, category="Data"),
            AssumptionSpec("aggregation", "selectbox", "Aggregate daily values by",
                           "Sum", {"choices": list(_AGG_FUNCS)}, category="Data"),
            AssumptionSpec("year", "number_input", "Year (0 = latest in data)",
                           0, {"min": 0, "max": 2100, "step": 1}, category="Data"),
            AssumptionSpec("compare_n_years", "slider", "Years to compare (YoY mode)",
                           3, {"min": 2, "max": 6, "step": 1}, category="Data"),
            AssumptionSpec("color_scale", "selectbox", "Color scale", "Greens",
                           {"choices": ["Greens", "Blues", "Reds", "Viridis", "Plasma", "YlOrRd"]},
                           category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        date_col = columns["date"]
        val_col  = columns["value"]
        warnings: list[str] = []

        work = df[[date_col, val_col]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        before = len(work)
        work   = work.dropna(subset=[date_col, val_col])
        if len(work) < before:
            warnings.append(f"Dropped {before - len(work)} row(s) with unparseable dates or missing values.")

        agg_fn = _AGG_FUNCS[params["aggregation"]]
        daily  = (
            work.groupby(work[date_col].dt.normalize())[val_col]
            .agg(agg_fn)
            .reset_index()
        )
        daily.columns = ["date", "value"]

        years = sorted(daily["date"].dt.year.unique().tolist())
        if not years:
            return BuildResult(figure=go.Figure(), warnings=["No valid date data found."])

        view_mode = params.get("view_mode", "Single Year")

        # ── Year-over-Year mode ───────────────────────────────────────────────
        if view_mode == "Year-over-Year":
            n_years = int(params.get("compare_n_years", 3))
            selected_years = years[-n_years:]
            if len(selected_years) < 2:
                warnings.append("Need at least 2 years of data for Year-over-Year comparison.")
                view_mode = "Single Year"
            else:
                warnings.append(
                    f"Comparing {len(selected_years)} years: "
                    + ", ".join(str(y) for y in selected_years)
                    + " — colour scale shared across all years."
                )
                fig = _build_yoy_fig(daily, selected_years, params["color_scale"],
                                     params["aggregation"], val_col)
                return BuildResult(figure=fig, warnings=warnings)

        # ── Single Year mode ─────────────────────────────────────────────────
        year_param = int(params.get("year", 0))
        year = year_param if year_param in years else years[-1]
        if year_param != 0 and year_param not in years:
            warnings.append(f"Year {year_param} not in data — showing {year} instead.")

        matrix, date_labels, week_labels = _build_calendar_matrix(
            daily[daily["date"].dt.year == year], year
        )

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=list(range(matrix.shape[1])),
            y=_DOW_LABELS,
            colorscale=params["color_scale"],
            showscale=True,
            hoverongaps=False,
            xgap=2,
            ygap=2,
            customdata=date_labels,
            hovertemplate="%{customdata}<br>Value: %{z:.2f}<extra></extra>",
        ))

        fig.update_layout(
            template="plotly_white",
            title=f"{val_col} — {year} ({params['aggregation']} per day)",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(week_labels))),
                ticktext=week_labels,
                showgrid=False,
            ),
            yaxis=dict(showgrid=False, autorange="reversed"),
            height=250,
        )
        return BuildResult(figure=fig, warnings=warnings)


registry.register(CalendarHeatmap)
