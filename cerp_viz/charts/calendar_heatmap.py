from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_AGG_FUNCS   = {"Sum": "sum", "Mean": "mean", "Count": "count", "Max": "max"}


def _build_calendar_matrix(daily: pd.DataFrame, year: int) -> tuple[np.ndarray, list[str]]:
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    dec31 = pd.Timestamp(year=year, month=12, day=31)

    # Fill full year grid so weeks align properly
    full_range = pd.date_range(jan1, dec31, freq="D")
    grid = pd.Series(index=full_range, dtype=float)
    for _, row in daily.iterrows():
        d = row["date"]
        if d in grid.index:
            grid[d] = row["value"]

    start_dow = jan1.dayofweek  # 0=Mon
    total_days = len(full_range) + start_dow
    n_weeks = (total_days + 6) // 7

    matrix = np.full((7, n_weeks), np.nan)
    for i, date in enumerate(full_range):
        col = (i + start_dow) // 7
        row = date.dayofweek
        matrix[row, col] = grid[date]

    # Month tick labels at first week of each month
    week_labels = [""] * n_weeks
    for date in full_range:
        if date.day == 1:
            week_col = (date.dayofyear - 1 + start_dow) // 7
            week_labels[week_col] = date.strftime("%b")

    return matrix, week_labels


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
            AssumptionSpec("aggregation", "selectbox", "Aggregate daily values by",
                           "Sum", {"choices": list(_AGG_FUNCS)}, category="Data"),
            AssumptionSpec("year", "number_input", "Year (0 = latest in data)",
                           0, {"min": 0, "max": 2100, "step": 1}, category="Data"),
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
        work = work.dropna(subset=[date_col, val_col])
        if len(work) < before:
            warnings.append(f"Dropped {before - len(work)} row(s) with unparseable dates or missing values.")

        agg_fn = _AGG_FUNCS[params["aggregation"]]
        daily = (
            work.groupby(work[date_col].dt.normalize())[val_col]
            .agg(agg_fn)
            .reset_index()
        )
        daily.columns = ["date", "value"]

        years = sorted(daily["date"].dt.year.unique())
        if not years:
            return BuildResult(figure=go.Figure(), warnings=["No valid date data found."])

        year_param = int(params.get("year", 0))
        year = year_param if year_param in years else years[-1]
        if year_param != 0 and year_param not in years:
            warnings.append(f"Year {year_param} not in data — showing {year} instead.")

        matrix, week_labels = _build_calendar_matrix(daily[daily["date"].dt.year == year], year)

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=list(range(matrix.shape[1])),
            y=_DOW_LABELS,
            colorscale=params["color_scale"],
            showscale=True,
            hoverongaps=False,
            xgap=2,
            ygap=2,
            customdata=matrix,
            hovertemplate="%{y}<br>Week %{x}<br>Value: %{z:.2f}<extra></extra>",
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
