from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

from cerp_viz.core.scenarios import ScenarioStore
from cerp_viz.core.theme import apply_theme
from cerp_viz.renderers.streamlit_renderer import StreamlitRenderer


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf)
    return buf.getvalue()


def _render_workflow_guide(store: ScenarioStore) -> None:
    saved = store.count()
    steps = [
        ("1", "Configure chart", "Set up your chart in the sidebar (columns, assumptions, theme)."),
        ("2", "Save a scenario", "Give it a name in **Scenarios** section of the sidebar and click 💾 Save."),
        ("3", "Adjust & save again", "Change one or more Data assumptions, then save under a different name."),
        ("4", "Run comparison", "Click **▶ Run Comparison** below to overlay all scenarios on one chart."),
    ]
    with st.expander("ℹ️ How scenario comparison works", expanded=(saved < 2)):
        cols = st.columns(4)
        for col, (num, title, desc) in zip(cols, steps):
            done = (num == "1") or (saved >= 1 and num in ("1", "2")) or (saved >= 2 and num in ("1", "2", "3"))
            icon = "✅" if done else "○"
            col.markdown(f"**{icon} Step {num}**")
            col.markdown(f"**{title}**")
            col.caption(desc)


def _render_scenario_table(store: ScenarioStore, assumption_specs) -> None:
    scenarios = store.all()
    if not scenarios:
        return

    label_map = {spec.key: spec.label for spec in assumption_specs}
    names = list(scenarios.keys())

    all_keys: set[str] = set()
    for s in scenarios.values():
        all_keys.update(s.params.keys())

    rows_all, rows_diff = [], []
    for key in sorted(all_keys):
        values = {n: scenarios[n].params.get(key, "—") for n in names}
        row = {"Parameter": label_map.get(key, key), **values}
        rows_all.append(row)
        if len({str(v) for v in values.values()}) > 1:
            rows_diff.append(row)

    diff_tab, full_tab = st.tabs(["🔍 Differences only", "📋 All parameters"])

    with diff_tab:
        if rows_diff:
            st.dataframe(
                pd.DataFrame(rows_diff).set_index("Parameter"),
                use_container_width=True,
            )
        else:
            st.info("Scenarios are identical — modify Data assumptions and save under a new name.")

    with full_tab:
        if rows_all:
            st.dataframe(
                pd.DataFrame(rows_all).set_index("Parameter"),
                use_container_width=True,
            )


def _render_delta_metrics(result) -> None:
    if not hasattr(result, "delta") or not result.delta:
        return
    st.markdown("##### Delta between first two scenarios")
    cols = st.columns(min(len(result.delta), 5))
    for i, (label, delta_val) in enumerate(list(result.delta.items())[:5]):
        cols[i].metric(label, f"{delta_val:+.2f}")


def render_compare_panel(
    df: pd.DataFrame,
    viz,
    col_mapping: dict,
    store: ScenarioStore,
    theme,
) -> None:
    st.subheader("⚖️ Compare Scenarios")

    _render_workflow_guide(store)

    if not viz.supports_comparison:
        supported = (
            "Bar Chart, Line Chart, Area Chart, Scatter Plot, "
            "Waterfall, Distribution, Funnel Chart, Tornado Chart"
        )
        st.warning(
            f"**{viz.name}** does not support scenario comparison.\n\n"
            f"Supported chart types: {supported}"
        )
        return

    saved = store.count()

    # ── Quick-save from inside the compare tab ────────────────────────────────
    with st.expander("💾 Save current configuration as a scenario", expanded=(saved == 0)):
        from cerp_viz.core.scenarios import ScenarioStore as _S  # noqa: F401
        current_params_key = "compare_quick_save_name"
        save_col, btn_col = st.columns([3, 1])
        save_name = save_col.text_input(
            "Scenario name", value=f"Scenario {saved + 1}",
            key=current_params_key, label_visibility="collapsed",
        )
        if btn_col.button("💾 Save", use_container_width=True, key="cmp_quick_save"):
            name = save_name.strip() or f"Scenario {saved + 1}"
            from cerp_viz.ui.assumption_panel import render as _render_assumptions  # noqa: F401
            current_params = st.session_state.get("_last_params", {})
            if current_params:
                store.save(name, current_params)
                st.success(f"Saved **{name}**")
                st.rerun()
            else:
                st.warning("No chart parameters found — click **▶ Apply** in the sidebar first.")

    # ── Saved scenario list ────────────────────────────────────────────────────
    if saved == 0:
        st.info("No scenarios saved yet. Use the sidebar or the form above to save your first scenario.")
        return

    names = store.names()
    st.markdown(f"**{saved} scenario(s) saved:** " + " · ".join(f"`{n}`" for n in names))

    # Delete controls
    with st.expander("🗑 Manage saved scenarios", expanded=False):
        for sname in list(store.names()):
            dcol, btnc = st.columns([5, 1])
            dcol.markdown(f"• **{sname}**")
            if btnc.button("🗑", key=f"cmp_del_{sname}", help=f"Delete '{sname}'"):
                store.delete(sname)
                st.rerun()

    if saved < 2:
        st.info(
            f"You have **{saved}** scenario saved. "
            "Save at least **2** to run a comparison."
        )
        return

    # ── Parameter diff table ──────────────────────────────────────────────────
    st.markdown("##### Scenario parameters")
    _render_scenario_table(store, viz.assumptions())

    # ── Run comparison ────────────────────────────────────────────────────────
    st.divider()
    if st.button("▶  Run Comparison", type="primary", use_container_width=False, key="cmp_run"):
        try:
            scenarios = {n: s.params for n, s in store.all().items()}
            result = viz.compare(df, col_mapping, scenarios)
            apply_theme(result.figure, theme)
            st.session_state["compare_result"] = result
            st.session_state["compare_error"]  = None
        except NotImplementedError as exc:
            st.session_state["compare_result"] = None
            st.session_state["compare_error"]  = str(exc)
        except Exception as exc:
            st.session_state["compare_result"] = None
            st.session_state["compare_error"]  = f"Comparison failed: {exc}"

    if err := st.session_state.get("compare_error"):
        st.error(err)

    compare_result = st.session_state.get("compare_result")
    if compare_result is None:
        return

    st.divider()

    # ── Result tabs ────────────────────────────────────────────────────────────
    tab_chart, tab_export = st.tabs(["📊 Comparison Chart", "⬇️ Export"])

    with tab_chart:
        if compare_result.warnings:
            for w in compare_result.warnings:
                st.warning(w)
        StreamlitRenderer().render(compare_result.figure)
        _render_delta_metrics(compare_result)

    with tab_export:
        st.caption("Export the comparison chart or the underlying scenario data.")
        try:
            from cerp_viz.renderers.image_renderer import ImageRenderer
            e1, e2, _ = st.columns([1, 1, 4])
            for col, fmt in [(e1, "png"), (e2, "svg")]:
                try:
                    img_bytes = ImageRenderer(fmt=fmt).to_bytes(compare_result.figure)
                    col.download_button(
                        f"⬇️ {fmt.upper()}",
                        data=img_bytes,
                        file_name=f"cerp_comparison.{fmt}",
                        mime=f"image/{fmt}",
                        key=f"cmp_export_{fmt}",
                    )
                except ImportError:
                    col.caption(f"_{fmt.upper()} needs `pip install kaleido`_")
        except Exception:
            pass

        scenarios_df_rows = []
        for sname, sc in store.all().items():
            row: dict[str, Any] = {"Scenario": sname}
            row.update(sc.params)
            scenarios_df_rows.append(row)
        if scenarios_df_rows:
            scenarios_df = pd.DataFrame(scenarios_df_rows).set_index("Scenario")
            st.download_button(
                "⬇️ Download scenario parameters (CSV)",
                data=_to_csv_bytes(scenarios_df),
                file_name="cerp_scenarios.csv",
                mime="text/csv",
                key="cmp_export_csv",
            )
