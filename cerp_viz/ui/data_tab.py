from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _sparkline(series: pd.Series) -> go.Figure:
    vals = series.dropna().reset_index(drop=True)
    fig = go.Figure(go.Scatter(
        y=vals,
        mode="lines",
        line=dict(color="#2563EB", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=2, b=2),
        height=60,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _render_metric_cards(df: pd.DataFrame) -> None:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return

    st.markdown("#### Column Summaries")
    per_row = 4
    for start in range(0, len(numeric_cols), per_row):
        batch = numeric_cols[start : start + per_row]
        cols  = st.columns(len(batch))
        for col_widget, col_name in zip(cols, batch):
            s = df[col_name].dropna()
            if s.empty:
                continue
            mean_v  = s.mean()
            min_v   = s.min()
            max_v   = s.max()
            missing = int(df[col_name].isna().sum())
            with col_widget:
                with st.container(border=True):
                    st.metric(
                        label=col_name,
                        value=f"{mean_v:,.2f}",
                        delta=f"{min_v:,.1f} – {max_v:,.1f}",
                        delta_color="off",
                        help=f"mean={mean_v:.3f}  std={s.std():.3f}  n={len(s):,}",
                    )
                    if missing:
                        st.caption(f"⚠ {missing} missing ({missing/len(df)*100:.1f}%)")
                    st.plotly_chart(
                        _sparkline(s),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"spark_{col_name}",
                    )


def _render_excel_download(df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        label="⬇️ Export as Excel",
        data=buf.getvalue(),
        file_name="cerp_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="data_excel_export",
    )


def _render_column_manager(df: pd.DataFrame) -> list[str]:
    """Column hide/show manager. Returns list of visible column names."""
    with st.expander("🔧 Manage Columns", expanded=False):
        hidden_key = "data_hidden_cols"
        if hidden_key not in st.session_state:
            st.session_state[hidden_key] = []

        all_cols = list(df.columns)
        hidden   = st.session_state[hidden_key]

        selected = st.multiselect(
            "Visible columns",
            options=all_cols,
            default=[c for c in all_cols if c not in hidden],
            key="data_col_visibility",
            help="Deselect columns to hide them from the grid below.",
        )

        col_reset, col_hide = st.columns([1, 3])
        if col_reset.button("Reset", key="data_col_reset", use_container_width=True):
            st.session_state[hidden_key] = []
            st.rerun()

        st.session_state[hidden_key] = [c for c in all_cols if c not in selected]
        visible = selected or all_cols
        col_hide.caption(
            f"{len(visible)} / {len(all_cols)} columns visible"
            + (f" · {len(all_cols) - len(visible)} hidden" if hidden else "")
        )

    return [c for c in df.columns if c not in st.session_state.get("data_hidden_cols", [])]


def _render_all_sheets(sheets: dict[str, pd.DataFrame]) -> None:
    """Browse all loaded sheets without switching the active analysis view."""
    if len(sheets) <= 1:
        st.info("Upload a multi-sheet Excel file to browse all sheets here.")
        return

    for sname, sdf in sheets.items():
        n_rows, n_cols = sdf.shape
        with st.expander(f"**{sname}** — {n_rows:,} rows × {n_cols} columns", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows",    f"{n_rows:,}")
            c2.metric("Columns", n_cols)
            c3.metric("Missing", f"{sdf.isnull().sum().sum():,}")

            switch_key = f"switch_sheet_{sname}"
            if st.button(f"Switch to '{sname}'", key=switch_key):
                st.session_state["_active_sheet_override"] = sname
                st.rerun()

            st.dataframe(sdf.head(100), use_container_width=True, hide_index=True)

            buf = io.BytesIO()
            sdf.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                f"⬇️ Export '{sname}'",
                data=buf.getvalue(),
                file_name=f"cerp_{sname}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_sheet_{sname}",
            )


def render_data_tab(
    df: pd.DataFrame,
    sheet_name: str,
    build_result,
    sheets: dict[str, pd.DataFrame] | None = None,
) -> None:
    from cerp_viz.ui.profile_panel import render_profile_panel

    tab_preview, tab_edit, tab_sheets, tab_profile = st.tabs(
        ["🗂 Preview", "✏️ Edit Data", "📑 All Sheets", "🔍 Profile Report"]
    )

    # ── Preview ───────────────────────────────────────────────────────────────
    with tab_preview:
        st.markdown(f"#### {sheet_name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",          f"{len(df):,}")
        c2.metric("Columns",       len(df.columns))
        c3.metric("Missing cells", f"{df.isnull().sum().sum():,}")

        st.divider()
        _render_metric_cards(df)
        st.divider()

        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            _render_excel_download(df)
        with btn_col2:
            if build_result is not None:
                if st.button("⬇️ Chart as HTML", key="data_html_export"):
                    from cerp_viz.renderers.html_renderer import HTMLRenderer
                    HTMLRenderer("cerp_output.html").render(build_result.figure)
                    st.success("Saved to cerp_output.html")

        st.divider()

        # Column manager
        visible_cols = _render_column_manager(df)
        st.dataframe(
            df[visible_cols],
            use_container_width=True,
            hide_index=True,
        )

    # ── Edit Data ─────────────────────────────────────────────────────────────
    with tab_edit:
        st.markdown("#### Edit Data")
        st.caption(
            "Changes made here are applied to your active dataset for charting. "
            "Click **Reset all edits** to revert."
        )

        if st.button("🔄 Reset all edits", key="data_reset_edits"):
            st.session_state.pop("user_edited_df", None)
            st.rerun()

        has_edits = "user_edited_df" in st.session_state
        if has_edits:
            st.info("Custom edits active — charts use the edited data below.")

        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="data_editor_main",
            hide_index=True,
            column_config={
                col: st.column_config.NumberColumn(format="%.4g")
                if pd.api.types.is_numeric_dtype(df[col]) else None
                for col in df.columns
            },
        )

        if not edited.equals(df):
            st.session_state["user_edited_df"] = edited
            st.success(f"Edits captured — {len(edited):,} rows · {len(edited.columns)} columns")

        csv_buf = edited.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download edited data (CSV)",
            data=csv_buf,
            file_name="cerp_edited.csv",
            mime="text/csv",
            key="data_edited_csv",
        )

    # ── All Sheets ────────────────────────────────────────────────────────────
    with tab_sheets:
        st.markdown("#### All Sheets")
        _render_all_sheets(sheets or {sheet_name: df})

    # ── Profile ───────────────────────────────────────────────────────────────
    with tab_profile:
        render_profile_panel(df)
