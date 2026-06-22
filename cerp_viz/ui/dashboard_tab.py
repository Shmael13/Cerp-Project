"""
Dashboard tab — renders a configurable grid of charts from DashboardConfig.
Each slot gets its own expander for inline re-configuration.
"""
from __future__ import annotations
from typing import Any

import streamlit as st

from cerp_viz.core.dashboard import DashboardConfig, SlotConfig
from cerp_viz.core.registry import registry
from cerp_viz.core.theme import Theme, apply_theme
from cerp_viz.renderers.streamlit_renderer import StreamlitRenderer

import pandas as pd

_LAYOUTS = {
    "1 column":  1,
    "2 columns": 2,
    "3 columns": 3,
}


def render_dashboard(
    df: pd.DataFrame,
    dashboard: DashboardConfig,
    theme: Theme,
    available_names: list[str],
) -> None:
    st.subheader("📋 Dashboard")

    # ── Top controls ─────────────────────────────────────────────────────────
    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_l:
        layout_name = st.selectbox(
            "Layout", list(_LAYOUTS.keys()), index=1, key="dashboard_layout"
        )
    with ctrl_r:
        if st.button("🗑 Clear All", key="dashboard_clear", use_container_width=True):
            dashboard.clear()
            st.rerun()

    if dashboard.count() == 0:
        st.info(
            "No charts on the dashboard yet.\n\n"
            "Go to **💡 Quick Start** and click **➕ Add to Dashboard** on any suggestion, "
            "or use the **Add Chart** button below to add one manually."
        )
    else:
        n_cols = _LAYOUTS[layout_name]
        cols   = st.columns(n_cols)

        to_remove: int | None = None

        for idx, slot in enumerate(dashboard.slots):
            col = cols[idx % n_cols]
            with col:
                with st.container(border=True):
                    header_l, header_r = st.columns([4, 1])
                    header_l.markdown(f"**{slot.title or slot.chart_name}**")
                    if header_r.button("✕", key=f"dash_remove_{idx}", help="Remove this chart"):
                        to_remove = idx

                    # Render the chart
                    try:
                        viz    = registry.get(slot.chart_name)()
                        result = viz.build(df, slot.columns, slot.params)
                        apply_theme(result.figure, theme)
                        result.figure.update_layout(
                            height=320,
                            margin=dict(t=30, b=30, l=30, r=30),
                            showlegend=True,
                        )
                        StreamlitRenderer().render(result.figure)
                        if result.warnings:
                            with st.expander("⚠️ Warnings", expanded=False):
                                for w in result.warnings:
                                    st.caption(w)
                    except Exception as exc:
                        st.error(f"Render failed: {exc}")

                    # Inline reconfigure
                    with st.expander("⚙️ Reconfigure", expanded=False):
                        _render_slot_config(idx, slot, df, available_names)

        if to_remove is not None:
            dashboard.remove(to_remove)
            st.rerun()

    # ── Add chart manually ────────────────────────────────────────────────────
    st.divider()
    with st.expander("➕ Add Chart to Dashboard", expanded=False):
        _render_add_chart(dashboard, df, available_names)

    # ── Export dashboard as HTML ──────────────────────────────────────────────
    if dashboard.count() > 0:
        st.divider()
        if st.button("⬇️ Export Dashboard as HTML", key="dashboard_export_html"):
            _export_dashboard_html(dashboard, df, theme)


def _render_slot_config(
    idx: int,
    slot: SlotConfig,
    df: pd.DataFrame,
    available_names: list[str],
) -> None:
    """Inline mini-config for a single dashboard slot."""
    from cerp_viz.ui import assumption_panel

    new_chart = st.selectbox(
        "Chart type", available_names,
        index=available_names.index(slot.chart_name) if slot.chart_name in available_names else 0,
        key=f"dash_chart_{idx}",
    )

    if new_chart != slot.chart_name:
        slot.chart_name = new_chart
        viz = registry.get(new_chart)()
        slot.columns = {s.role: None for s in viz.required_columns()}
        slot.params  = {s.key: s.default for s in viz.assumptions()}

    viz   = registry.get(slot.chart_name)()
    specs = viz.required_columns()

    all_cols = list(df.columns)
    st.markdown("**Columns**")
    for spec in specs:
        choices = ["(none)"] + all_cols if not spec.required else all_cols
        current = slot.columns.get(spec.role)
        default_idx = choices.index(current) if current in choices else 0
        chosen = st.selectbox(spec.label, choices, index=default_idx,
                               key=f"dash_col_{idx}_{spec.role}")
        slot.columns[spec.role] = None if chosen == "(none)" else chosen

    st.markdown("**Title**")
    slot.title = st.text_input("Chart title", value=slot.title or slot.chart_name,
                               key=f"dash_title_{idx}")


def _render_add_chart(
    dashboard: DashboardConfig,
    df: pd.DataFrame,
    available_names: list[str],
) -> None:
    chart_name = st.selectbox("Chart type", available_names, key="dash_add_chart")
    viz        = registry.get(chart_name)()
    title      = st.text_input("Title", value=chart_name, key="dash_add_title")

    all_cols = list(df.columns)
    col_map: dict[str, str | None] = {}
    for spec in viz.required_columns():
        choices = ["(none)"] + all_cols if not spec.required else all_cols
        chosen = st.selectbox(spec.label, choices, key=f"dash_add_col_{spec.role}")
        col_map[spec.role] = None if chosen == "(none)" else chosen

    if st.button("Add to Dashboard", key="dash_add_confirm", type="primary"):
        params = {s.key: s.default for s in viz.assumptions()}
        dashboard.add(SlotConfig(
            chart_name=chart_name,
            columns=col_map,
            params=params,
            title=title,
        ))
        st.rerun()


def _export_dashboard_html(
    dashboard: DashboardConfig,
    df: pd.DataFrame,
    theme: Theme,
) -> None:
    import io
    import plotly.io as pio

    figures_html: list[str] = []
    for slot in dashboard.slots:
        try:
            viz    = registry.get(slot.chart_name)()
            result = viz.build(df, slot.columns, slot.params)
            apply_theme(result.figure, theme)
            result.figure.update_layout(height=400)
            fig_html = pio.to_html(result.figure, full_html=False, include_plotlyjs="cdn")
            title_html = f"<h3>{slot.title or slot.chart_name}</h3>"
            figures_html.append(f"<div class='chart-card'>{title_html}{fig_html}</div>")
        except Exception as exc:
            figures_html.append(f"<div class='chart-card'><p>Error: {exc}</p></div>")

    grid_style = "display:grid;grid-template-columns:repeat(2,1fr);gap:20px;"
    body = f"<div style='{grid_style}'>{''.join(figures_html)}</div>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CERP Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: {theme.paper_bgcolor}; color: {theme.font_color}; }}
  h1 {{ border-bottom: 2px solid #ccc; padding-bottom: 8px; }}
  .chart-card {{ background: white; border-radius: 8px; padding: 12px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .chart-card h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #333; }}
</style>
</head>
<body>
<h1>📊 CERP Dashboard</h1>
{body}
</body>
</html>"""

    st.download_button(
        label="⬇️ Download Dashboard HTML",
        data=html.encode("utf-8"),
        file_name="cerp_dashboard.html",
        mime="text/html",
        key="dashboard_dl_html",
    )
    st.success("Dashboard ready to download.")
