"""
Shared helper for scenario-comparison overlays.
Calls build() once per scenario then merges traces into one figure.
Lives in charts/ (not core/) because it imports plotly.
"""
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any

import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.models import BuildResult

if TYPE_CHECKING:
    import pandas as pd
    from cerp_viz.core.base import BaseVisualization

_PALETTE = px.colors.qualitative.Plotly

# Trace types where marker.color gives a meaningful per-scenario tint
_MARKER_TINTABLE = {"bar", "scatter", "scattergl", "funnel", "histogram", "barpolar"}

# Trace types where line.color gives the primary colour
_LINE_TINTABLE = {"scatter", "scattergl", "funnel"}


def overlay_scenarios(
    viz: "BaseVisualization",
    df: "pd.DataFrame",
    columns: dict[str, str | None],
    scenarios: dict[str, dict[str, Any]],
) -> BuildResult:
    """
    Call viz.build() for each scenario, deepcopy the resulting traces, tag them
    with the scenario name, tint them with a per-scenario colour, and combine
    into one figure.  Works for any chart type whose traces can be overlaid.
    """
    combined: go.Figure = go.Figure()
    all_warnings: list[str] = []
    has_bar = False
    has_histogram = False

    for i, (scenario_name, params) in enumerate(scenarios.items()):
        result = viz.build(df, columns, params)
        color = _PALETTE[i % len(_PALETTE)]

        first = True
        for trace in result.figure.data:
            t = copy.deepcopy(trace)
            trace_type = getattr(t, "type", "")

            t.name = scenario_name
            t.legendgroup = scenario_name
            t.showlegend = first

            # Tint lines (scatter / line charts)
            if trace_type in _LINE_TINTABLE and hasattr(t, "line") and t.line is not None:
                t.line.color = color

            # Tint markers / fills
            if trace_type in _MARKER_TINTABLE and hasattr(t, "marker") and t.marker is not None:
                t.marker.color = color

            # Group bars so each scenario gets its own column
            if trace_type == "bar":
                t.offsetgroup = scenario_name
                has_bar = True

            if trace_type == "histogram":
                # Semi-transparent so overlapping histograms are both visible
                t.opacity = 0.6
                has_histogram = True

            combined.add_trace(t)
            first = False

        all_warnings.extend(f"[{scenario_name}] {w}" for w in result.warnings)

    # Choose barmode based on what trace types are present
    if has_histogram:
        barmode = "overlay"
    elif has_bar:
        barmode = "group"
    else:
        barmode = "group"  # harmless default for non-bar charts

    combined.update_layout(
        template="plotly_white",
        barmode=barmode,
        showlegend=True,
        legend_title_text="Scenario",
    )
    return BuildResult(figure=combined, warnings=all_warnings)
