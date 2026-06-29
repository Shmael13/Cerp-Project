from typing import Any, ClassVar

import pandas as pd
import plotly.express as px

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry


class BoxPlot(BaseVisualization):
    name: ClassVar[str] = "Box Plot"
    description: ClassVar[str] = "Show distribution spread and outliers across categories."

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("x",     "categorical", "Category (X)"),
            ColumnSpec("y",     "numeric",     "Value (Y)"),
            ColumnSpec("color", "categorical", "Color Group (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("show_points", "selectbox", "Show Points", "outliers",
                           {"choices": ["outliers", "all", "none"]}, category="Display"),
            AssumptionSpec("orientation", "selectbox", "Orientation", "v",
                           {"choices": ["v", "h"]}, category="Display"),
            AssumptionSpec("notched",     "toggle",    "Notched boxes", False,
                           {}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        x_col     = columns["x"]
        y_col     = columns["y"]
        color_col = columns.get("color")
        warnings: list[str] = []

        work = df.dropna(subset=[c for c in [x_col, y_col] if c])
        dropped = len(df) - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} row(s) with missing values.")

        points = params["show_points"]
        if points == "none":
            points = False

        fig = px.box(
            work,
            x=x_col if params["orientation"] == "v" else y_col,
            y=y_col if params["orientation"] == "v" else x_col,
            color=color_col,
            points=points,
            notched=params["notched"],
            template="plotly_white",
        )
        return BuildResult(figure=fig, warnings=warnings)


registry.register(BoxPlot)
