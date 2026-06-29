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
        line=dict(color="#636EFA", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(99,110,250,0.12)",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=2, b=2),
        height=70,
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

    st.markdown("**Column Summaries**")
    per_row = 4
    for start in range(0, len(numeric_cols), per_row):
        batch = numeric_cols[start : start + per_row]
        cols  = st.columns(len(batch))
        for col_widget, col_name in zip(cols, batch):
            s = df[col_name].dropna()
            if s.empty:
                continue
            mean_v   = s.mean()
            min_v    = s.min()
            max_v    = s.max()
            missing  = int(df[col_name].isna().sum())
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
        label="⬇️ Export data as Excel",
        data=buf.getvalue(),
        file_name="cerp_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="data_excel_export",
    )


def render_data_tab(df: pd.DataFrame, sheet_name: str, build_result) -> None:
    from cerp_viz.ui.profile_panel import render_profile_panel

    tab_preview, tab_profile = st.tabs(["🗂 Preview", "🔍 Profile Report"])

    with tab_preview:
        st.subheader(f"Data Preview — {sheet_name}")

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
        st.dataframe(df, use_container_width=True)

    with tab_profile:
        render_profile_panel(df)
