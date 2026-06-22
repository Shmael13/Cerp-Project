from typing import Any, ClassVar

import pandas as pd
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class HeatmapChart(BaseVisualization):
    name: ClassVar[str] = "Heatmap"
    description: ClassVar[str] = "Visualize values across two categorical dimensions."

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",     "categorical", "X Axis"),
            ColumnSpec("y",     "categorical", "Y Axis"),
            ColumnSpec("value", "numeric",     "Value"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            # ── Data ─────────────────────────────────────────────────────────
            AssumptionSpec("aggregation",   "selectbox", "Aggregation",          "mean",
                           {"choices": ["sum", "mean", "count", "max", "min"]}, category="Data"),
            AssumptionSpec("normalize",     "selectbox", "Normalize By",         "None",
                           {"choices": ["None", "Row", "Column", "Total"]}, category="Data"),
            AssumptionSpec("fill_missing",  "selectbox", "Fill Missing Cells",   "Leave blank",
                           {"choices": ["Leave blank", "Zero", "Column mean"]}, category="Data"),
            AssumptionSpec("top_n_x",       "number_input", "Top N X Categories (0 = all)", 0,
                           {"min": 0, "max": 100, "step": 1}, category="Data"),
            AssumptionSpec("top_n_y",       "number_input", "Top N Y Categories (0 = all)", 0,
                           {"min": 0, "max": 100, "step": 1}, category="Data"),
            # ── Display ───────────────────────────────────────────────────────
            AssumptionSpec("color_scale",   "selectbox", "Color Scale",          "RdBu",
                           {"choices": ["RdBu", "Viridis", "Blues", "Reds", "YlOrRd", "Plasma", "Cividis"]}, category="Display"),
            AssumptionSpec("reverse_scale", "toggle",    "Reverse Color Scale",  False,
                           {}, category="Display"),
            AssumptionSpec("show_values",   "toggle",    "Show Values in Cells", True,
                           {}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        x_col   = columns["x"]
        y_col   = columns["y"]
        val_col = columns["value"]
        warnings: list[str] = []

        missing = [role for role, col in [("X", x_col), ("Y", y_col), ("Value", val_col)]
                   if not col or col not in df.columns]
        if missing:
            raise ValueError(f"Required column(s) not mapped: {', '.join(missing)}. "
                             "Please select columns in the sidebar.")

        if x_col == y_col:
            raise ValueError(
                f"X Axis and Y Axis must be different columns "
                f"(both are set to '{x_col}'). Pick two distinct columns."
            )

        work = df[[x_col, y_col, val_col]].copy()
        work[val_col] = pd.to_numeric(work[val_col], errors="coerce")

        before = len(work)
        work = work.dropna(subset=[val_col])
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values in '{val_col}'.")

        pivot = work.pivot_table(
            index=y_col, columns=x_col, values=val_col,
            aggfunc=params["aggregation"],
        )

        # Top-N filtering on axes
        top_n_x = int(params["top_n_x"])
        top_n_y = int(params["top_n_y"])

        if top_n_x > 0 and len(pivot.columns) > top_n_x:
            top_x_cats = pivot.sum(axis=0).abs().nlargest(top_n_x).index
            pivot = pivot[top_x_cats]
            warnings.append(f"X axis limited to top {top_n_x} categories by total value.")

        if top_n_y > 0 and len(pivot.index) > top_n_y:
            top_y_cats = pivot.sum(axis=1).abs().nlargest(top_n_y).index
            pivot = pivot.loc[top_y_cats]
            warnings.append(f"Y axis limited to top {top_n_y} categories by total value.")

        # Fill missing cells
        fill = params["fill_missing"]
        if fill == "Zero":
            pivot = pivot.fillna(0)
            warnings.append("Missing pivot cells filled with 0.")
        elif fill == "Column mean":
            pivot = pivot.fillna(pivot.mean(axis=0))
            warnings.append("Missing pivot cells filled with column mean.")

        # Normalisation
        norm = params["normalize"]
        if norm == "Row":
            pivot = pivot.div(pivot.sum(axis=1), axis=0)
            warnings.append("Values normalized by row total.")
        elif norm == "Column":
            pivot = pivot.div(pivot.sum(axis=0), axis=1)
            warnings.append("Values normalized by column total.")
        elif norm == "Total":
            pivot = pivot / pivot.values.sum()
            warnings.append("Values normalized by grand total.")

        fig = px.imshow(
            pivot,
            color_continuous_scale=params["color_scale"],
            text_auto=".2f" if params["show_values"] else False,
            aspect="auto",
            template="plotly_white",
        )
        if params["reverse_scale"]:
            fig.update_coloraxes(reversescale=True)
        return BuildResult(figure=fig, warnings=warnings)


registry.register(HeatmapChart)
