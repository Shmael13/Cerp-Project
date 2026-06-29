from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

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

_OP_PREVIEW = {
    "scale":    lambda col, v: f"**{col}** × {v} ({'+' if v >= 1 else ''}{(v-1)*100:.0f}%)",
    "add":      lambda col, v: f"**{col}** {'+' if v >= 0 else ''}{v:g}  (shift by {v:g})",
    "replace":  lambda col, v: f"**{col}** = {v:g}  (all values set to {v:g})",
    "clip_min": lambda col, v: f"**{col}** ≥ {v:g}  (floor at {v:g})",
    "clip_max": lambda col, v: f"**{col}** ≤ {v:g}  (cap at {v:g})",
    "round":    lambda col, v: f"**{col}** rounded to {int(v)} decimal place(s)",
}


def _build_rules(n: int, numeric_cols: list[str]) -> list[SimRule]:
    rules: list[SimRule] = []
    for i in range(n):
        col = st.session_state.get(f"wi_col_{i}", numeric_cols[0])
        op  = st.session_state.get(f"wi_op_{i}", "scale")
        val = st.session_state.get(f"wi_val_{i}", _OP_DEFAULTS.get(op, 1.0))
        rules.append(SimRule(column=col, operation=op, value=float(val)))
    return rules


def _rule_preview(i: int) -> str:
    col = st.session_state.get(f"wi_col_{i}", "")
    op  = st.session_state.get(f"wi_op_{i}", "scale")
    val = float(st.session_state.get(f"wi_val_{i}", _OP_DEFAULTS.get(op, 1.0)))
    fn  = _OP_PREVIEW.get(op)
    return fn(col, val) if fn and col else ""


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
    preview = _rule_preview(i)
    if preview:
        st.caption(f"↳ {preview}")


def _apply_preset(preset: str, n: int) -> None:
    presets: dict[str, tuple | None] = {
        "+10%":       ("scale", 1.1),
        "+20%":       ("scale", 1.2),
        "-10%":       ("scale", 0.9),
        "-20%":       ("scale", 0.8),
        "Zero floor": ("clip_min", 0.0),
        "Reset":      None,
    }
    setting = presets.get(preset)
    if setting is None:
        for i in range(n):
            st.session_state[f"wi_op_{i}"]  = "scale"
            st.session_state[f"wi_val_{i}"] = 1.0
        return
    op, val = setting
    for i in range(n):
        st.session_state[f"wi_op_{i}"]  = op
        st.session_state[f"wi_val_{i}"] = val


def _run_simulation(
    df: pd.DataFrame, rules: list[SimRule], viz,
    col_mapping: dict, params: dict, theme, scenario_label: str,
) -> WhatIfResult:
    sim_df   = apply_rules(df, rules)
    base_res = viz.build(df, col_mapping, params)
    sim_res  = viz.build(sim_df, col_mapping, params)
    apply_theme(base_res.figure, theme)
    apply_theme(sim_res.figure, theme)
    base_res.figure.update_layout(title_text="Baseline")
    sim_res.figure.update_layout(title_text=scenario_label)
    return WhatIfResult(base_res.figure, sim_res.figure, sim_df)


def _build_delta_df(df: pd.DataFrame, sim_df: pd.DataFrame, changed_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in changed_cols:
        b = df[col].dropna()
        s = sim_df[col].dropna()
        base_sum  = float(b.sum())
        sim_sum   = float(s.sum())
        base_mean = float(b.mean())
        sim_mean  = float(s.mean())
        delta_sum  = sim_sum - base_sum
        delta_mean = sim_mean - base_mean
        pct_sum    = (delta_sum / base_sum * 100) if base_sum else 0.0
        rows.append({
            "Column":    col,
            "Base Total": round(base_sum, 2),
            "Sim Total":  round(sim_sum, 2),
            "Δ Total":    round(delta_sum, 2),
            "Δ Total %":  round(pct_sum, 2),
            "Base Mean":  round(base_mean, 4),
            "Sim Mean":   round(sim_mean, 4),
            "Δ Mean":     round(delta_mean, 4),
        })
    return pd.DataFrame(rows).set_index("Column") if rows else pd.DataFrame()


def _render_impact_summary(df: pd.DataFrame, sim_df: pd.DataFrame, col_mapping: dict) -> None:
    numeric_mapped = [
        c for c in col_mapping.values()
        if c and c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_mapped:
        return
    st.markdown("##### Impact on Charted Columns")
    cols = st.columns(min(len(numeric_mapped), 5))
    for i, c in enumerate(numeric_mapped[:5]):
        base_sum = float(df[c].sum())
        sim_sum  = float(sim_df[c].sum())
        delta    = sim_sum - base_sum
        pct      = (delta / base_sum * 100) if base_sum else 0
        cols[i].metric(f"∑ {c}", f"{sim_sum:,.2f}", delta=f"{delta:+,.2f} ({pct:+.1f}%)")


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf)
    return buf.getvalue()


def render_whatif_panel(df: pd.DataFrame, viz, col_mapping: dict, params: dict, theme) -> None:
    st.subheader("🔮 What-If Simulator")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.info("No numeric columns found in this dataset.")
        return

    ops    = available_operations()
    op_ids = [o["id"] for o in ops]

    if "wi_n_rules" not in st.session_state:
        st.session_state["wi_n_rules"] = 1

    n = st.session_state["wi_n_rules"]

    # ── Scenario name ─────────────────────────────────────────────────────────
    name_col, _ = st.columns([2, 3])
    with name_col:
        st.text_input(
            "Scenario name",
            value="Scenario A",
            key="wi_scenario_name",
            help="Label shown above the scenario chart.",
        )

    # ── Quick presets ─────────────────────────────────────────────────────────
    with st.expander("⚡ Quick presets — apply to all rules", expanded=False):
        st.caption("Instantly set all rule values, then fine-tune individual rows below.")
        preset_labels = ["+10%", "+20%", "-10%", "-20%", "Zero floor", "Reset"]
        pcols = st.columns(len(preset_labels))
        for idx, label in enumerate(preset_labels):
            if pcols[idx].button(label, key=f"wi_preset_{label}", use_container_width=True):
                _apply_preset(label, n)
                st.rerun()

    # ── Rule builder ──────────────────────────────────────────────────────────
    st.markdown("**Rules** — applied sequentially to the data before re-rendering the chart")

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

    if btn_run.button("▶  Run Simulation", type="primary", use_container_width=True, key="wi_run"):
        rules = _build_rules(n, numeric_cols)
        label = st.session_state.get("wi_scenario_name", "Scenario A") or "Scenario A"
        try:
            result = _run_simulation(df, rules, viz, col_mapping, params, theme, label)
            st.session_state["wi_result"] = result
            st.session_state["wi_rules"]  = rules
            st.session_state["wi_error"]  = None
        except Exception as exc:
            st.session_state["wi_result"] = None
            st.session_state["wi_error"]  = str(exc)

    if err := st.session_state.get("wi_error"):
        st.error(f"Simulation failed: {err}")

    result: WhatIfResult | None = st.session_state.get("wi_result")
    if result is None:
        st.divider()
        st.caption(
            "Configure one or more rules above, then click **▶ Run Simulation**. "
            "The chart will appear side-by-side with the baseline so you can spot differences immediately."
        )
        return

    st.divider()

    # ── Output tabs ───────────────────────────────────────────────────────────
    tab_charts, tab_impact, tab_data = st.tabs(["📊 Chart Comparison", "📈 Impact Summary", "📋 Data Table"])

    with tab_charts:
        col_b, col_s = st.columns(2)
        with col_b:
            st.markdown("##### Baseline")
            StreamlitRenderer().render(result.baseline_fig)
        with col_s:
            label = st.session_state.get("wi_scenario_name", "Scenario") or "Scenario"
            st.markdown(f"##### {label}")
            StreamlitRenderer().render(result.scenario_fig)

    with tab_impact:
        _render_impact_summary(df, result.sim_df, col_mapping)

        changed = [
            c for c in df.select_dtypes(include="number").columns
            if not df[c].equals(result.sim_df[c])
        ]
        if changed:
            st.markdown("##### All Modified Columns")
            delta_df = _build_delta_df(df, result.sim_df, changed)
            if not delta_df.empty:
                def _color_delta(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return "color: #2ecc71"
                        elif val < 0:
                            return "color: #e74c3c"
                    return ""

                styled = delta_df.style.applymap(
                    _color_delta, subset=["Δ Total", "Δ Total %", "Δ Mean"]
                )
                st.dataframe(styled, use_container_width=True)
        else:
            st.info("No numeric columns changed — check your rules.")

    with tab_data:
        st.caption(
            "Simulated data after applying all rules. "
            "Download as CSV for offline analysis."
        )
        st.dataframe(result.sim_df.head(200), use_container_width=True)
        if len(result.sim_df) > 200:
            st.caption(f"Showing 200 of {len(result.sim_df)} rows in preview.")
        st.download_button(
            "⬇️ Download simulated data (CSV)",
            data=_to_csv_bytes(result.sim_df),
            file_name="cerp_whatif_simulation.csv",
            mime="text/csv",
            key="wi_download",
        )
