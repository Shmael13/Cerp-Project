from typing import Any

import streamlit as st

from cerp_viz.core.models import AssumptionSpec


def render(specs: list[AssumptionSpec]) -> dict[str, Any]:
    """
    Groups AssumptionSpecs by category, renders the correct Streamlit widget
    for each, and returns a flat dict of key → current value.
    Always appends universal title/subtitle fields at the bottom.
    """
    params: dict[str, Any] = {}

    categories: dict[str, list[AssumptionSpec]] = {}
    for spec in specs:
        categories.setdefault(spec.category, []).append(spec)

    for category, items in categories.items():
        st.sidebar.markdown(f"**── {category} ──**")
        for spec in items:
            params[spec.key] = _render_widget(spec)

    # Universal title / subtitle — appended after chart-specific assumptions
    st.sidebar.markdown("**── Labels ──**")
    params["_chart_title"]    = st.sidebar.text_input("Chart Title (leave blank = auto)", "",
                                                       key="assumption__chart_title")
    params["_chart_subtitle"] = st.sidebar.text_input("Subtitle (optional)", "",
                                                       key="assumption__chart_subtitle")

    return params


def apply_title_subtitle(figure: Any, params: dict[str, Any]) -> Any:
    """Apply user-supplied title/subtitle to a Plotly figure. Returns the figure."""
    title    = (params.get("_chart_title")    or "").strip()
    subtitle = (params.get("_chart_subtitle") or "").strip()

    if not title and not subtitle:
        return figure

    layout_kw: dict[str, Any] = {}
    if title:
        layout_kw["title_text"] = title
        layout_kw["title_font_size"] = 18

    figure.update_layout(**layout_kw)

    if subtitle:
        figure.add_annotation(
            text=f"<i>{subtitle}</i>",
            xref="paper", yref="paper",
            x=0, y=1.04,
            xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=12, color="#666666"),
        )

    return figure


def _render_widget(spec: AssumptionSpec) -> Any:
    opts = spec.options
    key  = f"assumption_{spec.key}"

    if spec.widget == "slider":
        return st.sidebar.slider(
            spec.label,
            min_value=opts.get("min", 0.0),
            max_value=opts.get("max", 100.0),
            value=spec.default,
            step=opts.get("step", 1.0),
            key=key,
        )

    if spec.widget == "selectbox":
        choices = opts.get("choices", [])
        idx = choices.index(spec.default) if spec.default in choices else 0
        return st.sidebar.selectbox(spec.label, choices, index=idx, key=key)

    if spec.widget == "number_input":
        return st.sidebar.number_input(
            spec.label,
            min_value=opts.get("min", None),
            max_value=opts.get("max", None),
            value=spec.default,
            step=opts.get("step", 1),
            key=key,
        )

    if spec.widget == "multiselect":
        return st.sidebar.multiselect(
            spec.label,
            options=opts.get("choices", []),
            default=spec.default,
            key=key,
        )

    if spec.widget == "toggle":
        return st.sidebar.toggle(spec.label, value=spec.default, key=key)

    return spec.default
