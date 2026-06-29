"""
Bar Race — animated horizontal bar chart that shows categories racing in
rank across time periods. One Plotly animation frame per period; bars are
sorted by value each frame so viewers watch rankings shift in real-time.
"""
from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_COLOR_SEQS = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold":   px.colors.qualitative.Bold,
    "Vivid":  px.colors.qualitative.Vivid,
    "Dark":   px.colors.qualitative.Dark24,
}

_AGG_FUNCS = ["sum", "mean", "max", "min"]


def _assign_colors(categories: list[str], palette: list[str]) -> dict[str, str]:
    return {cat: palette[i % len(palette)] for i, cat in enumerate(sorted(categories))}


def _make_bar_trace(
    row_data: pd.DataFrame,
    color_map: dict[str, str],
    show_value: bool,
) -> go.Bar:
    text = [f" {v:,.1f}" for v in row_data["value"]] if show_value else None
    return go.Bar(
        x=row_data["value"].tolist(),
        y=row_data["category"].tolist(),
        orientation="h",
        marker_color=[color_map.get(c, "#636EFA") for c in row_data["category"]],
        text=text,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,.2f}<extra></extra>",
    )


class BarRaceChart(BaseVisualization):
    name: ClassVar[str] = "Bar Race"
    description: ClassVar[str] = (
        "Animated horizontal bars that race through time — "
        "watch categories rise and fall in rank period by period."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("time",     "any",        "Time period column (one frame per value)"),
            ColumnSpec("value",    "numeric",    "Numeric value to rank by"),
            ColumnSpec("category", "categorical", "Category (one bar per value)"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("aggregation",     "selectbox",  "Aggregation per period", "sum",
                           {"choices": _AGG_FUNCS},              category="Data"),
            AssumptionSpec("top_n",           "slider",     "Top N categories shown", 10,
                           {"min": 3, "max": 25, "step": 1},     category="Data"),
            AssumptionSpec("frame_duration",  "slider",     "Frame duration (ms)",    700,
                           {"min": 100, "max": 3000, "step": 100}, category="Display"),
            AssumptionSpec("color_scheme",    "selectbox",  "Color scheme",           "Plotly",
                           {"choices": list(_COLOR_SEQS)},        category="Display"),
            AssumptionSpec("show_value",      "toggle",     "Show value labels",      True,
                           {},                                    category="Display"),
            AssumptionSpec("ascending",       "toggle",     "Rank ascending (lower = rank 1)", False,
                           {},                                    category="Data"),
        ]

    def build(
        self,
        df: pd.DataFrame,
        columns: dict[str, str | None],
        params: dict[str, Any],
    ) -> BuildResult:
        time_col = columns["time"]
        val_col  = columns["value"]
        cat_col  = columns["category"]
        warnings: list[str] = []

        work = df[[time_col, val_col, cat_col]].copy()
        work[val_col] = pd.to_numeric(work[val_col], errors="coerce")
        work = work.dropna(subset=[val_col])

        # Sort time axis
        try:
            work[time_col] = pd.to_datetime(work[time_col], errors="ignore")
        except Exception:
            pass
        work = work.sort_values(time_col)

        agg_fn  = params["aggregation"]
        top_n   = int(params["top_n"])
        asc     = bool(params["ascending"])
        dur_ms  = int(params["frame_duration"])
        palette = _COLOR_SEQS[params["color_scheme"]]
        show_v  = bool(params["show_value"])

        # Aggregate: (time, category) → scalar value
        grouped = (
            work.groupby([time_col, cat_col], sort=True)[val_col]
            .agg(agg_fn)
            .reset_index()
        )
        grouped.columns = ["time", "category", "value"]

        periods = grouped["time"].unique().tolist()
        if len(periods) < 2:
            return BuildResult(
                figure=go.Figure(),
                warnings=["Bar Race needs at least 2 distinct time periods."],
            )

        # Determine consistent top-N category set (by overall rank)
        overall_rank = (
            grouped.groupby("category")["value"]
            .agg("mean")
            .sort_values(ascending=asc)
        )
        top_cats = overall_rank.index[:top_n].tolist()
        if len(top_cats) < top_n:
            warnings.append(f"Only {len(top_cats)} categories found (requested {top_n}).")

        all_cats = grouped["category"].unique().tolist()
        color_map = _assign_colors(all_cats, palette)

        # Global x-axis max for fixed scale across frames
        top_data   = grouped[grouped["category"].isin(top_cats)]
        global_max = float(top_data["value"].max()) if len(top_data) else 1.0
        x_pad      = global_max * 0.15

        def _frame_data(period) -> pd.DataFrame:
            sub = grouped[grouped["time"] == period].copy()
            sub = sub[sub["category"].isin(top_cats)]
            sub = (
                sub.set_index("category")[["value"]]
                .reindex(top_cats, fill_value=0)
                .reset_index()
            )
            sub = sub.sort_values("value", ascending=True)  # largest at top after y-inversion
            return sub

        # Build first frame as the initial figure state
        first_df = _frame_data(periods[0])
        init_trace = _make_bar_trace(first_df, color_map, show_v)

        fig = go.Figure(
            data=[init_trace],
            layout=go.Layout(
                xaxis=dict(
                    range=[0, global_max + x_pad],
                    showgrid=True,
                    gridcolor="#e8e8e8",
                    title=f"{agg_fn}({val_col})",
                ),
                yaxis=dict(showgrid=False, autorange=False,
                           range=[-0.5, top_n - 0.5]),
                template="plotly_white",
                margin=dict(l=10, r=80, t=60, b=40),
                height=max(350, 38 * len(top_cats) + 100),
                title=dict(
                    text=f"<b>{str(periods[0])}</b>",
                    x=0.5,
                    font=dict(size=20),
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        x=0.0,
                        y=1.12,
                        xanchor="left",
                        buttons=[
                            dict(
                                label="▶  Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame":      {"duration": dur_ms, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": max(50, dur_ms // 4)},
                                    },
                                ],
                            ),
                            dict(
                                label="⏸  Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame":      {"duration": 0, "redraw": False},
                                        "mode":       "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        currentvalue=dict(
                            prefix=f"{time_col}: ",
                            font=dict(size=14),
                            visible=True,
                            xanchor="center",
                        ),
                        pad=dict(t=10, b=10),
                        x=0.0,
                        len=1.0,
                        steps=[
                            dict(
                                method="animate",
                                label=str(p),
                                args=[
                                    [str(p)],
                                    {
                                        "mode":       "immediate",
                                        "frame":      {"duration": dur_ms, "redraw": True},
                                        "transition": {"duration": max(50, dur_ms // 4)},
                                    },
                                ],
                            )
                            for p in periods
                        ],
                    )
                ],
            ),
        )

        # Build animation frames
        frames = []
        for period in periods:
            fd = _frame_data(period)
            frames.append(go.Frame(
                data=[_make_bar_trace(fd, color_map, show_v)],
                name=str(period),
                layout=go.Layout(
                    title_text=f"<b>{str(period)}</b>",
                    yaxis=dict(
                        tickvals=list(range(len(fd))),
                        ticktext=fd["category"].tolist(),
                        autorange=False,
                        range=[-0.5, top_n - 0.5],
                    ),
                ),
            ))
        fig.frames = frames

        warnings.append(
            f"{len(periods)} frames · top {len(top_cats)} categories · "
            f"{dur_ms} ms/frame — press ▶ Play to animate."
        )
        return BuildResult(figure=fig, warnings=warnings)


registry.register(BarRaceChart)
