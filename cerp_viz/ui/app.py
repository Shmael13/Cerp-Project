import cerp_viz.charts       # noqa: F401 — triggers self-registration of all chart types
import cerp_viz.suggestions   # noqa: F401 — triggers self-registration of all suggesters
import streamlit as st

from cerp_viz.core.compatibility import compatible_visualizations
from cerp_viz.core.config import ChartConfig
from cerp_viz.core.dashboard import DashboardConfig, SlotConfig
from cerp_viz.core.registry import registry
from cerp_viz.core.scenarios import ScenarioStore
from cerp_viz.core.suggestions import SuggestionResult
from cerp_viz.core.theme import THEMES, apply_theme
from cerp_viz.renderers.streamlit_renderer import StreamlitRenderer
from cerp_viz.suggestions import suggester_registry
from cerp_viz.ui import assumption_panel, scenario_panel, sidebar
from cerp_viz.ui.assumption_panel import apply_title_subtitle
from cerp_viz.ui.transform_panel import render_transform_panel
from cerp_viz.ui.whatif_panel import render_whatif_panel
from cerp_viz.ui.compare_panel import render_compare_panel
from cerp_viz.ui.multifile_panel import render_multifile_panel
from cerp_viz.ui.data_tab import render_data_tab
from cerp_viz.ui.welcome_panel import render_welcome
from cerp_viz.core.transform import apply_transforms


@st.cache_data(show_spinner=False)
def _cached_compatibility(df) -> dict:
    return compatible_visualizations(df)


@st.cache_data(show_spinner="Analysing your data for suggestions…")
def _cached_suggestions(df, suggester_name: str) -> list[SuggestionResult]:
    return suggester_registry.get(suggester_name).suggest(df)


@st.cache_data(show_spinner=False)
def _cached_transforms(df, transform_cfg):
    return apply_transforms(df, transform_cfg)


def main() -> None:
    # ── Consume pending suggestion keys BEFORE any widget is instantiated ─────
    if "_pending_chart" in st.session_state:
        st.session_state["chart_picker"] = st.session_state.pop("_pending_chart")
    if "_pending_cols" in st.session_state:
        for role, col in st.session_state.pop("_pending_cols").items():
            if col is not None:
                st.session_state[f"col_{role}"] = col
    if "_pending_params" in st.session_state:
        for key, val in st.session_state.pop("_pending_params").items():
            st.session_state[f"assumption_{key}"] = val
    if "_pending_transforms" in st.session_state:
        _consume_pending_transforms(st.session_state.pop("_pending_transforms"))

    st.set_page_config(page_title="CERP Visualizer", layout="wide", page_icon="📊")
    st.title("📊 CERP Data Visualizer")
    st.caption("Upload a spreadsheet, configure your chart in the sidebar, then click **▶ Apply**.")

    # ── Persistent stores ─────────────────────────────────────────────────────
    if "scenario_store" not in st.session_state:
        st.session_state["scenario_store"] = ScenarioStore()
    if "dashboard" not in st.session_state:
        st.session_state["dashboard"] = DashboardConfig()
    store:     ScenarioStore  = st.session_state["scenario_store"]
    dashboard: DashboardConfig = st.session_state["dashboard"]

    # ── File upload ───────────────────────────────────────────────────────────
    st.sidebar.title("Configuration")
    sheets, sheet_name = sidebar.render_upload()

    if sheets is None:
        render_welcome()
        st.stop()

    raw_df = sheets[sheet_name]

    # ── Multi-file join (optional, overrides raw_df when active) ─────────────
    st.sidebar.divider()
    render_multifile_panel(raw_df)
    raw_df = st.session_state.get("merged_df", raw_df)

    # ── Data transforms (filters, derived cols, date parts) ───────────────────
    st.sidebar.divider()
    transform_cfg = render_transform_panel(raw_df)
    df, transform_warnings = _cached_transforms(raw_df, transform_cfg)

    # ── Compatibility check (cached) ──────────────────────────────────────────
    compat      = _cached_compatibility(df)
    available   = [n for n, r in compat.items() if r.compatible]
    unavailable = {n: r for n, r in compat.items() if not r.compatible}

    if not available:
        st.error("No chart types are compatible with this sheet. Try a different sheet.")
        st.stop()

    st.sidebar.divider()
    st.sidebar.caption(f"✅ {len(available)} of {len(compat)} chart types available")

    # ── Chart picker ──────────────────────────────────────────────────────────
    viz_name = sidebar.render_chart_picker(available)
    viz      = registry.get(viz_name)()

    config_key = f"{sheet_name}::{viz_name}"
    if st.session_state.get("_config_key") != config_key:
        st.session_state.pop("build_result",   None)
        st.session_state.pop("compare_result", None)
        st.session_state.pop("figure_error",   None)
        store.clear()
        st.session_state["_config_key"] = config_key

    # ── Column mapping ────────────────────────────────────────────────────────
    st.sidebar.divider()
    col_mapping = sidebar.render_column_mapping(df, viz.required_columns())

    # ── Assumptions ───────────────────────────────────────────────────────────
    st.sidebar.divider()
    params = assumption_panel.render(viz.assumptions())

    # ── Theme selector ────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("**── Theme ──**")
    theme_name = st.sidebar.selectbox(
        "Visual theme", list(THEMES.keys()), index=0, key="global_theme"
    )
    theme = THEMES[theme_name]

    # ── Apply button ──────────────────────────────────────────────────────────
    st.sidebar.divider()
    if st.sidebar.button("▶  Apply", type="primary", use_container_width=True):
        try:
            result = viz.build(df, col_mapping, params)
            apply_theme(result.figure, theme)
            apply_title_subtitle(result.figure, params)
            st.session_state["build_result"] = result
            st.session_state["figure_error"] = None
            st.session_state["_last_params"]  = dict(params)
            _precompute_exports(result.figure)
        except Exception as exc:
            st.session_state["build_result"] = None
            st.session_state["figure_error"] = str(exc)

    # ── Scenario panel ────────────────────────────────────────────────────────
    st.sidebar.divider()
    scenario_panel.render(store, params)

    # ── Config import / export ────────────────────────────────────────────────
    st.sidebar.divider()
    _render_config_io(st, viz_name, col_mapping, params, store)

    # ── Retrieve last results ─────────────────────────────────────────────────
    build_result = st.session_state.get("build_result")
    figure_error = st.session_state.get("figure_error")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_qs, tab_chart, tab_compare, tab_whatif, tab_dash, tab_data = st.tabs(
        ["💡 Quick Start", "📊 Chart", "⚖️ Compare Scenarios", "🔮 What-If", "📋 Dashboard", "🗂 Data"]
    )

    # ── Tab 1: Quick Start (most useful entry point) ──────────────────────────
    with tab_qs:
        _render_suggestions_tab(st, df, col_mapping, available, dashboard, theme)

    # ── Tab 2: Chart ──────────────────────────────────────────────────────────
    with tab_chart:
        col_title, col_export = st.columns([4, 1])
        col_title.subheader(viz.name)
        col_title.caption(viz.description)

        if transform_warnings:
            _render_warnings(st, [f"[Transform] {w}" for w in transform_warnings])

        if build_result is not None:
            _render_warnings(st, build_result.warnings)
            StreamlitRenderer().render(build_result.figure)
            _render_export_buttons(st, build_result)
            _render_stats_bar(st, df, col_mapping)
            _render_story_panel(st, df, viz, col_mapping, params, build_result.warnings)
            _render_ai_interpretation(st, viz, col_mapping, params, df, build_result.warnings)
        elif figure_error:
            st.error(f"Could not render chart: {figure_error}")
        else:
            st.info("Configure your settings in the sidebar and click **▶ Apply**.")

        if unavailable:
            with st.expander(f"ℹ️ {len(unavailable)} chart type(s) unavailable for this sheet"):
                for name, result in unavailable.items():
                    st.markdown(f"- **{name}** — {result.reason}")

    # ── Tab 3 (slot 2): Scenario comparison ──────────────────────────────────
    with tab_compare:
        render_compare_panel(df, viz, col_mapping, store, theme)

    # ── Tab 4: What-If Simulator ──────────────────────────────────────────────
    with tab_whatif:
        render_whatif_panel(df, viz, col_mapping, params, theme)

    # ── Tab 5: Dashboard ─────────────────────────────────────────────────────
    with tab_dash:
        from cerp_viz.ui.dashboard_tab import render_dashboard
        render_dashboard(df, dashboard, theme, available)

    # ── Tab 6: Data ───────────────────────────────────────────────────────────
    with tab_data:
        render_data_tab(df, sheet_name, build_result)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _precompute_exports(figure) -> None:
    """Generate PNG and SVG bytes once when Apply is clicked; skip if kaleido absent."""
    from cerp_viz.renderers.image_renderer import ImageRenderer
    for fmt, key in [("png", "_export_png"), ("svg", "_export_svg")]:
        try:
            st.session_state[key] = ImageRenderer(fmt=fmt).to_bytes(figure)
        except Exception:
            st.session_state[key] = None


def _render_export_buttons(st, build_result) -> None:
    """PNG / SVG download buttons — bytes are pre-computed at Apply time, not on every rerun."""
    c1, c2, _ = st.columns([1, 1, 4])
    for col, fmt, label, key in [
        (c1, "png", "⬇️ PNG", "_export_png"),
        (c2, "svg", "⬇️ SVG", "_export_svg"),
    ]:
        data = st.session_state.get(key)
        if data is not None:
            col.download_button(
                label=label,
                data=data,
                file_name=f"cerp_chart.{fmt}",
                mime=f"image/{fmt}",
                key=f"export_{fmt}",
            )
        else:
            col.caption(f"_{fmt.upper()} needs `pip install kaleido`_")


def _render_suggestions_tab(
    st, df, col_mapping, available_names: list[str],
    dashboard: DashboardConfig, theme
) -> None:
    """Render the Quick Start tab: engine picker, ranked suggestion cards."""
    st.subheader("💡 Quick Start")

    engine_names = suggester_registry.names()
    default_idx  = (
        engine_names.index("Smart (Statistical + Rule-Based)")
        if "Smart (Statistical + Rule-Based)" in engine_names else 0
    )
    engine_col, _ = st.columns([2, 3])
    with engine_col:
        chosen_engine = st.selectbox(
            "Suggestion engine",
            options=engine_names,
            index=default_idx,
            key="qs_engine_picker",
            help=(
                "**Smart** — statistical + heuristic combined (best results).\n\n"
                "**Statistical** — correlation, OLS trend, Pareto, distribution shape.\n\n"
                "**Rule-Based** — column type and name heuristics.\n\n"
                "**AI (Claude)** — natural-language analysis (requires API key)."
            ),
        )

    st.caption(
        f"Suggestions from **{chosen_engine}** — ranked by relevance. "
        "**Apply** pre-fills the sidebar. **➕ Dashboard** adds to your dashboard grid."
    )

    suggestions = _cached_suggestions(df, chosen_engine)
    if not suggestions:
        st.info("No suggestions for this sheet. Try a different engine or upload a richer file.")
        return

    valid  = [s for s in suggestions if s.chart_name in available_names]
    hidden = len(suggestions) - len(valid)

    for i, s in enumerate(valid):
        score_pct = int(s.score * 100)
        score_bar = "█" * (score_pct // 10) + "░" * (10 - score_pct // 10)

        with st.container(border=True):
            col_info, col_apply, col_dash = st.columns([5, 1, 1])
            with col_info:
                st.markdown(f"**{s.title}**")
                st.caption(f"Chart type: `{s.chart_name}`")
                st.markdown(f"_{s.rationale}_")
                st.markdown(
                    f"<small>Confidence: `{score_bar}` {score_pct}%</small>",
                    unsafe_allow_html=True,
                )
                mapped = ", ".join(
                    f"`{role}` → **{col}**"
                    for role, col in s.columns.items()
                    if col is not None
                )
                if mapped:
                    st.markdown(f"<small>{mapped}</small>", unsafe_allow_html=True)
                if s.transforms:
                    hints = " &nbsp;·&nbsp; ".join(
                        f"✦ {t['label']}" for t in s.transforms[:3] if "label" in t
                    )
                    st.markdown(
                        f"<small style='color:#6c9;'>Suggested transforms: {hints}</small>",
                        unsafe_allow_html=True,
                    )

            with col_apply:
                if st.button("Apply", key=f"qs_apply_{i}", type="primary"):
                    _apply_suggestion(st, s, df, theme)

            with col_dash:
                if st.button("➕", key=f"qs_dash_{i}", help="Add to Dashboard"):
                    dashboard.add(SlotConfig(
                        chart_name=s.chart_name,
                        columns=s.columns,
                        params=s.params,
                        title=s.title,
                    ))
                    st.toast(f"Added '{s.title}' to dashboard", icon="📋")

    if hidden:
        st.caption(f"_{hidden} suggestion(s) hidden — chart type not compatible with this sheet._")


def _transforms_to_config(transforms: list[dict]) -> "object":
    """Convert list-of-dicts transform hints into a TransformConfig."""
    from cerp_viz.core.transform import TransformConfig, FilterRule, DerivedColumn, DatePart
    filters, derived_cols, date_parts = [], [], []
    for t in transforms:
        if t.get("type") == "filter":
            filters.append(FilterRule(column=t["column"], operator=t["operator"], value=t["value"]))
        elif t.get("type") == "date_part":
            date_parts.append(DatePart(source_column=t["source_column"], part=t["part"]))
        elif t.get("type") == "derived":
            derived_cols.append(DerivedColumn(name=t["name"], expression=t["expression"]))
    return TransformConfig(filters=filters, derived_cols=derived_cols, date_parts=date_parts)


def _consume_pending_transforms(transforms: list[dict]) -> None:
    """Pre-populate transform panel session-state keys from suggestion hints."""
    filters    = [t for t in transforms if t.get("type") == "filter"]
    date_parts = [t for t in transforms if t.get("type") == "date_part"]
    derived    = [t for t in transforms if t.get("type") == "derived"]

    if filters:
        st.session_state["tf_n_filters"] = min(len(filters), 4)
        for i, f in enumerate(filters[:4]):
            st.session_state[f"tf_f_col_{i}"] = f["column"]
            st.session_state[f"tf_f_op_{i}"]  = f["operator"]
            st.session_state[f"tf_f_val_{i}"]  = f["value"]

    if date_parts:
        st.session_state["tf_n_dp"] = min(len(date_parts), 4)
        for i, dp in enumerate(date_parts[:4]):
            st.session_state[f"tf_dp_col_{i}"]  = dp["source_column"]
            st.session_state[f"tf_dp_part_{i}"] = dp["part"]

    if derived:
        st.session_state["tf_n_derived"] = min(len(derived), 3)
        for i, d in enumerate(derived[:3]):
            st.session_state[f"tf_d_name_{i}"] = d["name"]
            st.session_state[f"tf_d_expr_{i}"] = d["expression"]


def _apply_suggestion(st, s: "SuggestionResult", df, theme) -> None:
    """Two-rerun pattern: stage pending keys then rerun."""
    st.session_state["_pending_chart"]  = s.chart_name
    st.session_state["_pending_cols"]   = {
        role: col for role, col in s.columns.items()
        if col is not None and col in df.columns
    }
    st.session_state["_pending_params"] = s.params

    if s.transforms:
        st.session_state["_pending_transforms"] = s.transforms

    try:
        viz = registry.get(s.chart_name)()
        build_df = df
        if s.transforms:
            from cerp_viz.core.transform import apply_transforms as _apply_transforms
            build_df, _ = _apply_transforms(df, _transforms_to_config(s.transforms))
        result = viz.build(build_df, s.columns, s.params)
        apply_theme(result.figure, theme)
        st.session_state["build_result"] = result
        st.session_state["figure_error"] = None
    except Exception as exc:
        st.session_state["build_result"] = None
        st.session_state["figure_error"] = str(exc)

    st.rerun()


def _render_warnings(st, warnings: list[str]) -> None:
    if warnings:
        with st.expander("⚠️ Data adjustments applied before rendering", expanded=True):
            for w in warnings:
                st.warning(w)


def _render_stats_bar(st, df, col_mapping: dict) -> None:
    numeric_mapped = [
        col for col in col_mapping.values()
        if col and col in df.columns and df[col].dtype.kind in "biufc"
    ]
    if not numeric_mapped:
        return
    with st.expander("📐 Column statistics", expanded=False):
        stats = df[numeric_mapped].describe().T[["mean", "std", "min", "50%", "max"]]
        stats.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
        st.dataframe(stats.round(3), use_container_width=True)



def _render_story_panel(st, df, viz, col_mapping, params, warnings) -> None:
    """Key Observations expander — stat bullets always shown; AI story on demand."""
    from cerp_viz.core.story import generate_story_bullets

    bullets = generate_story_bullets(df, viz.name, col_mapping, params)
    if not bullets:
        return

    with st.expander("📖 Key Observations", expanded=False):
        for b in bullets:
            st.markdown(f"- {b}")

        # AI story section (only if API key present)
        from cerp_viz.ai.story_writer import is_available, write_story
        if is_available():
            st.divider()
            if st.button("🤖 Generate AI Data Story", key="ai_story"):
                numeric_stats = {
                    col: {
                        "min":  float(df[col].min()),
                        "max":  float(df[col].max()),
                        "mean": float(df[col].mean()),
                    }
                    for col in col_mapping.values()
                    if col and col in df.columns and df[col].dtype.kind in "biufc"
                }
                with st.spinner("Crafting data story…"):
                    try:
                        story = write_story(
                            viz_name=viz.name,
                            columns=col_mapping,
                            params=params,
                            df_stats=numeric_stats,
                            stat_bullets=bullets,
                            warnings=warnings,
                        )
                        st.session_state["_ai_story"] = story
                    except Exception as exc:
                        st.error(f"Story generation failed: {exc}")

        if "_ai_story" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["_ai_story"])


def _render_ai_interpretation(st, viz, col_mapping, params, df, warnings) -> None:
    from cerp_viz.ai.interpreter import interpret_chart, is_available

    if not is_available():
        return

    if st.button("🤖 Explain this chart", key="ai_explain"):
        numeric_stats = {
            col: {
                "min":  float(df[col].min()),
                "max":  float(df[col].max()),
                "mean": float(df[col].mean()),
            }
            for col in col_mapping.values()
            if col and col in df.columns and df[col].dtype.kind in "biufc"
        }
        with st.spinner("Interpreting chart…"):
            try:
                interpretation = interpret_chart(
                    viz_name=viz.name,
                    viz_description=viz.description,
                    columns=col_mapping,
                    params=params,
                    df_stats=numeric_stats,
                    warnings=warnings,
                )
                st.info(f"**AI Interpretation**\n\n{interpretation}")
            except Exception as exc:
                st.error(f"AI interpretation failed: {exc}")


def _render_config_io(st, viz_name, col_mapping, params, store) -> None:
    st.sidebar.markdown("**── Config ──**")

    config = ChartConfig(
        chart_name=viz_name,
        columns=col_mapping,
        params=params,
        scenarios={n: s.params for n, s in store.all().items()},
    )
    st.sidebar.download_button(
        "⬇️ Export config",
        data=config.to_json(),
        file_name="cerp_config.json",
        mime="application/json",
        use_container_width=True,
    )

    st.sidebar.markdown("⬆️ Import config")
    uploaded = st.sidebar.file_uploader(
        "Import config file", type=["json"],
        key="config_uploader",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        try:
            cfg = ChartConfig.from_json(uploaded.read().decode())
            for role, col in cfg.columns.items():
                if col:
                    st.session_state[f"col_{role}"] = col
            for key, val in cfg.params.items():
                st.session_state[f"assumption_{key}"] = val
            store.clear()
            for name, sparams in cfg.scenarios.items():
                store.save(name, sparams)
            st.sidebar.success(f"Config loaded: {cfg.chart_name}")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Failed to load config: {exc}")
