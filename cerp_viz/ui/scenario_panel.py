"""
Sidebar UI for saving, listing, and deleting named scenarios.
Only this file knows about Streamlit; ScenarioStore is pure Python.
"""
from __future__ import annotations
from typing import Any

import streamlit as st

from cerp_viz.core.scenarios import ScenarioStore


def render(store: ScenarioStore, current_params: dict[str, Any]) -> None:
    """Render the Scenarios section in the sidebar. Mutates store in place."""
    st.sidebar.markdown("**── Scenarios ──**")

    name = st.sidebar.text_input("Scenario name", "Base Case", key="scenario_name_input")

    if st.sidebar.button("💾 Save current assumptions", use_container_width=True):
        if name.strip():
            store.save(name.strip(), current_params)
        else:
            st.sidebar.warning("Enter a name before saving.")

    if store.count() == 0:
        st.sidebar.caption("No scenarios saved yet.")
        return

    st.sidebar.caption(f"{store.count()} scenario(s) saved:")
    for scenario_name in list(store.names()):
        col_label, col_del = st.sidebar.columns([5, 1])
        col_label.markdown(f"• **{scenario_name}**")
        if col_del.button("🗑", key=f"del_scenario_{scenario_name}", help=f"Delete '{scenario_name}'"):
            store.delete(scenario_name)
            st.rerun()
