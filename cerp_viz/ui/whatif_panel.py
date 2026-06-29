from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from cerp_viz.core.simulate import SimRule, apply_rules, available_operations
from cerp_viz.core.theme import apply_theme
from cerp_viz.renderers.streamlit_renderer import StreamlitRenderer


@dataclass
class WhatIfResult:
    baseline_fig: object
    scenario_fig: object
    sim_df: pd.DataFrame


_OP_SHORT = {
    "scale":    "× factor",
    "add":      "+ constant",
    "replace":  "= constant",
    "clip_min": "floor ≥",
    "clip_max": "cap ≤",
    "round":    "round (decimals)",
}

_OP_DEFAULTS = {
    "scale": 1.1,
    "add": 0.0,
    "replace": 0.0,
    "clip_min": 0.0,
    "clip_max": 1000.0,
    "round": 0.0,
}


def _build_rules(n: int, numeric_cols: list[str]) -> list[SimRule]:
    rules: list[SimRule] = []
    for i in range(n):
        col = st.session_state.get(f"wi_col_{i}", numeric_cols[0])
        op  = st.session_state.get(f"wi_op_{i}", "scale")
        val = st.session_state.get(f"wi_val_{i}", _OP_DEFAULTS.get(op, 1.0))
        rules.append(SimRule(column=col, operation=op, value=float(val)))
    return rules


def _render_rule_row(i: int, numeric_cols: list[str], op_ids: list[str]) -> None:
    c1, c2, c3 = st.columns([3, 2, 2])
    c1.selectbox(
        "Column", numeric_cols,
        key=f"wi_col_{i}", label_visibility="collapsed",
    )
    c2.selectbox(
        "Operation", op_ids,
        format_func=lambda x: _OP_SHORT.get(x, x),
        key=f"wi_op_{i}", label_visibility="collapsed",
    )
    default_val = _OP_DEFAULTS.get(st.session_state.get(f"wi_op_{i}", "scale"), 1.0)
    c3.number_input(
        "Value", value=default_val, step=0.1,
        key=f"wi_val_{i}", label_visibility="collapsed",
    )


def _run_simulation(df: pd.DataFrame, rules: list[SimRule], viz, col_mapping: dict, params: dict, theme) -> WhatIfResult:
    sim_df   = apply_rules(df, rules)
    base_res = viz.build(df, col_mapping, params)
    sim_res  = viz.build(sim_df, col_mapping, params)
    apply_theme(base_res.figure, theme)
    apply_theme(sim_res.figure, theme)
    base_res.figure.update_layout(title_text="Baseline")
    sim_res.figure.update_layout(title_text="What-If Scenario")
    return WhatIfResult(base_res.figure, sim_res.figure, sim_df)


def _render_impact_summary(df: pd.DataFrame, sim_df: pd.DataFrame, col_mapping: dict) -> None:
    numeric_mapped = [
        c for c in col_mapping.values()
        if c and c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_mapped:
        return
    st.markdown("**Impact Summary**")
    cols = st.columns(min(len(numeric_mapped), 5))
    for i, c in enumerate(numeric_mapped[:5]):
        base_sum = float(df[c].sum())
        sim_sum  = float(sim_df[c].sum())
        delta    = sim_sum - base_sum
        pct      = (delta / base_sum * 100) if base_sum else 0
        cols[i].metric(f"∑ {c}", f"{sim_sum:,.2f}", delta=f"{delta:+,.2f} ({pct:+.1f}%)")


def render_whatif_panel(df: pd.DataFrame, viz, col_mapping: dict, params: dict, theme) -> None:
    st.subheader("🔮 What-If Simulator")
    st.caption(
        "Apply numeric rules to columns and see how your chart changes versus the baseline. "
        "Rules execute sequentially on top of any active data transforms."
    )

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.info("No numeric columns found in this dataset.")
        return

    ops    = available_operations()
    op_ids = [o["id"] for o in ops]

    if "wi_n_rules" not in st.session_state:
        st.session_state["wi_n_rules"] = 1

    n = st.session_state["wi_n_rules"]

    hdr = st.columns([3, 2, 2])
    hdr[0].caption("Column")
    hdr[1].caption("Operation")
    hdr[2].caption("Value")

    for i in range(n):
        _render_rule_row(i, numeric_cols, op_ids)

    btn_add, btn_rem, btn_run, _ = st.columns([1, 1, 2, 3])

    if btn_add.button("➕ Rule", use_container_width=True, key="wi_add"):
        st.session_state["wi_n_rules"] = min(8, n + 1)
        st.rerun()

    if btn_rem.button("✕ Last", use_container_width=True, key="wi_rem", disabled=(n <= 1)):
        st.session_state["wi_n_rules"] = max(1, n - 1)
        st.rerun()

    if btn_run.button("▶  Run What-If", type="primary", use_container_width=True, key="wi_run"):
        rules = _build_rules(n, numeric_cols)
        try:
            result = _run_simulation(df, rules, viz, col_mapping, params, theme)
            st.session_state["wi_result"] = result
            st.session_state["wi_error"]  = None
        except Exception as exc:
            st.session_state["wi_result"] = None
            st.session_state["wi_error"]  = str(exc)

    if err := st.session_state.get("wi_error"):
        st.error(f"Simulation failed: {err}")

    result: WhatIfResult | None = st.session_state.get("wi_result")
    if result is None:
        return

    col_b, col_s = st.columns(2)
    with col_b:
        st.markdown("##### Baseline")
        StreamlitRenderer().render(result.baseline_fig)
    with col_s:
        st.markdown("##### What-If Scenario")
        StreamlitRenderer().render(result.scenario_fig)

    st.divider()
    _render_impact_summary(df, result.sim_df, col_mapping)
