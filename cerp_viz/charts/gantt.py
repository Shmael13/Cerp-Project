from __future__ import annotations

import pandas as pd
import plotly.express as px

from cerp_viz.core.models import BuildResult, ColumnSpec, AssumptionSpec
from cerp_viz.core.registry import registry


@registry.register
class GanttChart:
    name = "Gantt Chart"
    supports_comparison = False

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec(role="task",  label="Task / Activity",  dtype="categorical", required=True),
            ColumnSpec(role="start", label="Start Date",        dtype="datetime",   required=True),
            ColumnSpec(role="end",   label="End Date",          dtype="datetime",   required=True),
            ColumnSpec(role="color", label="Group / Category",  dtype="categorical", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec(key="sort_by",     label="Sort tasks by",  type="select",
                           options=["Start", "End", "Task", "Duration"], default="Start"),
            AssumptionSpec(key="show_today",  label="Show today line", type="bool",   default=True),
            AssumptionSpec(key="row_height",  label="Row height (px)", type="int",
                           min=20, max=60, default=32),
        ]

    def compatible(self, df: pd.DataFrame) -> tuple[bool, str]:
        dt_cols  = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        str_cols = list(df.select_dtypes(include="object").columns)
        if len(str_cols) < 1:
            return False, "No categorical column available for task names."
        if len(dt_cols) < 2 and len(str_cols) < 3:
            return False, "Needs at least 2 date columns (start + end) and 1 categorical column."
        return True, ""

    def build(self, df: pd.DataFrame, col_mapping: dict, params: dict) -> BuildResult:
        task_col  = col_mapping.get("task")
        start_col = col_mapping.get("start")
        end_col   = col_mapping.get("end")
        color_col = col_mapping.get("color")

        if not task_col or not start_col or not end_col:
            raise ValueError("Task, Start, and End columns are required.")

        warns: list[str] = []
        work = df[[task_col, start_col, end_col] +
                  ([color_col] if color_col else [])].copy()

        work[start_col] = pd.to_datetime(work[start_col], errors="coerce")
        work[end_col]   = pd.to_datetime(work[end_col],   errors="coerce")

        n_before = len(work)
        work = work.dropna(subset=[start_col, end_col])
        if len(work) < n_before:
            warns.append(f"Dropped {n_before - len(work)} row(s) with unparseable dates.")

        bad_order = work[start_col] >= work[end_col]
        if bad_order.any():
            warns.append(f"{bad_order.sum()} row(s) have Start ≥ End and were excluded.")
            work = work[~bad_order]

        if work.empty:
            raise ValueError("No valid rows remain after date validation.")

        work["_duration_days"] = (work[end_col] - work[start_col]).dt.days

        sort_by = params.get("sort_by", "Start")
        sort_map = {
            "Start":    start_col,
            "End":      end_col,
            "Task":     task_col,
            "Duration": "_duration_days",
        }
        work = work.sort_values(sort_map.get(sort_by, start_col))

        row_height  = int(params.get("row_height", 32))
        n_tasks     = work[task_col].nunique()
        fig_height  = max(200, row_height * n_tasks + 80)

        fig = px.timeline(
            work,
            x_start=start_col,
            x_end=end_col,
            y=task_col,
            color=color_col if color_col else None,
            custom_data=["_duration_days"],
            labels={task_col: "Task", start_col: "Start", end_col: "End"},
            height=fig_height,
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Start: %{base|%Y-%m-%d}<br>"
                "End: %{x|%Y-%m-%d}<br>"
                "Duration: %{customdata[0]} day(s)<extra></extra>"
            )
        )

        fig.update_yaxes(autorange="reversed")

        if params.get("show_today", True):
            today = pd.Timestamp.today().normalize()
            fig.add_vline(
                x=today.timestamp() * 1000,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text="Today",
                annotation_position="top right",
            )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=task_col,
            margin=dict(l=10, r=10, t=40, b=40),
            legend_title=color_col or "",
        )

        return BuildResult(figure=fig, warnings=warns)
