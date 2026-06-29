from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Pure computation helpers ──────────────────────────────────────────────────

def _shannon_entropy(series: pd.Series) -> float:
    counts = series.dropna().value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log2(counts + 1e-12)).sum())


def _iqr_outlier_rate(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 4:
        return 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())


def _column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        fill_pct   = round((1 - null_count / n) * 100, 1) if n else 0.0
        n_unique   = int(s.nunique(dropna=True))
        entropy    = round(_shannon_entropy(s), 3)
        dtype_str  = str(s.dtype)

        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            row = {
                "Column":        col,
                "Type":          dtype_str,
                "Fill %":        fill_pct,
                "Unique":        n_unique,
                "Entropy":       entropy,
                "Mean":          round(float(non_null.mean()), 4) if len(non_null) else None,
                "Std":           round(float(non_null.std()), 4)  if len(non_null) else None,
                "Skewness":      round(float(non_null.skew()), 3) if len(non_null) >= 3 else None,
                "Outlier %":     round(_iqr_outlier_rate(s) * 100, 1),
                "Top value":     None,
            }
        else:
            top = s.dropna().mode()
            top_val = str(top.iloc[0]) if len(top) else "—"
            row = {
                "Column":        col,
                "Type":          dtype_str,
                "Fill %":        fill_pct,
                "Unique":        n_unique,
                "Entropy":       entropy,
                "Mean":          None,
                "Std":           None,
                "Skewness":      None,
                "Outlier %":     None,
                "Top value":     top_val,
            }
        rows.append(row)
    return pd.DataFrame(rows).set_index("Column")


def _null_heatmap(df: pd.DataFrame) -> go.Figure:
    null_matrix = df.isnull().astype(int)
    if null_matrix.values.max() == 0:
        return None

    sample = null_matrix.head(200) if len(null_matrix) > 200 else null_matrix
    fig = go.Figure(go.Heatmap(
        z=sample.T.values,
        x=list(range(len(sample))),
        y=list(sample.columns),
        colorscale=[[0, "#eaf2fb"], [1, "#e74c3c"]],
        showscale=False,
        hoverongaps=False,
    ))
    fig.update_layout(
        height=max(120, 22 * len(df.columns)),
        margin=dict(l=0, r=0, t=30, b=10),
        xaxis=dict(title="Row index", showticklabels=False),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        title=dict(text="Missing-value map (red = null)", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _top_values_chart(series: pd.Series, col: str, n: int = 8) -> go.Figure:
    counts = series.dropna().value_counts().head(n)
    fig = go.Figure(go.Bar(
        x=counts.values[::-1],
        y=[str(v) for v in counts.index[::-1]],
        orientation="h",
        marker_color="#636EFA",
        text=[f"{v:,}" for v in counts.values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(180, 30 * len(counts)),
        margin=dict(l=0, r=10, t=20, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _distribution_chart(series: pd.Series, col: str) -> go.Figure:
    vals = series.dropna()
    if len(vals) < 3:
        return None
    fig = go.Figure(go.Histogram(
        x=vals,
        nbinsx=min(40, vals.nunique()),
        marker_color="#636EFA",
        opacity=0.8,
    ))
    mean_v = float(vals.mean())
    fig.add_vline(x=mean_v, line_dash="dash", line_color="#EF553B",
                  annotation_text=f"mean={mean_v:.2f}", annotation_position="top right")
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=20, b=10),
        xaxis=dict(title=None, tickfont=dict(size=10)),
        yaxis=dict(title="count", tickfont=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _render_overview(df: pd.DataFrame) -> None:
    n_rows, n_cols = df.shape
    n_missing      = int(df.isnull().sum().sum())
    n_dupes        = int(df.duplicated().sum())
    pct_missing    = round(n_missing / (n_rows * n_cols) * 100, 1) if n_rows * n_cols else 0
    n_numeric      = len(df.select_dtypes("number").columns)
    n_cat          = n_cols - n_numeric

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows",         f"{n_rows:,}")
    c2.metric("Columns",      n_cols)
    c3.metric("Numeric cols", n_numeric)
    c4.metric("Categorical",  n_cat)
    c5.metric("Missing cells", f"{n_missing:,}", delta=f"{pct_missing}%", delta_color="inverse")
    c6.metric("Duplicate rows", f"{n_dupes:,}", delta_color="inverse")

    if n_dupes > 0:
        st.warning(f"{n_dupes} duplicate row(s) detected — consider deduplicating before analysis.")
    if pct_missing > 20:
        st.warning(f"{pct_missing}% of all cells are missing — imputation or filtering recommended.")


def _render_column_table(profile: pd.DataFrame) -> None:
    st.markdown("#### Column Quality Summary")
    st.caption("Fill % = rows with a value ÷ total rows. Entropy = Shannon information (higher = more diverse). Outlier % = IQR rule (numeric cols only).")

    def _color_fill(val):
        if pd.isna(val):
            return ""
        if val < 50:
            return "color: #e74c3c; font-weight: bold"
        if val < 80:
            return "color: #e67e22"
        return "color: #27ae60"

    def _color_outlier(val):
        if pd.isna(val):
            return ""
        if val > 15:
            return "color: #e74c3c; font-weight: bold"
        if val > 5:
            return "color: #e67e22"
        return ""

    styled = (
        profile.style
        .applymap(_color_fill,    subset=["Fill %"])
        .applymap(_color_outlier, subset=["Outlier %"])
        .format({
            "Fill %":    "{:.1f}",
            "Entropy":   "{:.3f}",
            "Mean":      lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "Std":       lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "Skewness":  lambda v: f"{v:.2f}"  if pd.notna(v) else "—",
            "Outlier %": lambda v: f"{v:.1f}"  if pd.notna(v) else "—",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True)


def _render_column_detail(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes("number").columns.tolist()
    cat_cols     = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        st.markdown("#### Numeric Columns — Distributions")
        per_row = 3
        for start in range(0, len(numeric_cols), per_row):
            batch = numeric_cols[start : start + per_row]
            cols  = st.columns(len(batch))
            for widget_col, col_name in zip(cols, batch):
                with widget_col:
                    st.markdown(f"**{col_name}**")
                    fig = _distribution_chart(df[col_name], col_name)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True,
                                        config={"displayModeBar": False},
                                        key=f"prof_dist_{col_name}")

    if cat_cols:
        st.markdown("#### Categorical Columns — Top Values")
        per_row = 3
        for start in range(0, len(cat_cols), per_row):
            batch = cat_cols[start : start + per_row]
            cols  = st.columns(len(batch))
            for widget_col, col_name in zip(cols, batch):
                with widget_col:
                    n_unique = df[col_name].nunique()
                    st.markdown(f"**{col_name}** ({n_unique} unique)")
                    fig = _top_values_chart(df[col_name], col_name)
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"prof_top_{col_name}")


def _render_null_map(df: pd.DataFrame) -> None:
    fig = _null_heatmap(df)
    if fig is None:
        st.success("No missing values detected in this dataset.")
    else:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False},
                        key="prof_null_heatmap")


def _export_profile_csv(profile: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    profile.reset_index().to_csv(buf, index=False)
    return buf.getvalue()


# ── Public entry point ────────────────────────────────────────────────────────

def render_profile_panel(df: pd.DataFrame) -> None:
    st.markdown("### Data Profile Report")
    st.caption("Auto-generated quality report — no configuration needed.")

    _render_overview(df)
    st.divider()

    profile = _column_profile(df)
    _render_column_table(profile)

    st.download_button(
        "⬇️ Export profile as CSV",
        data=_export_profile_csv(profile),
        file_name="cerp_profile.csv",
        mime="text/csv",
        key="profile_csv_export",
    )

    st.divider()

    tab_dist, tab_missing = st.tabs(["📊 Column Distributions", "🔴 Missing Value Map"])
    with tab_dist:
        _render_column_detail(df)
    with tab_missing:
        _render_null_map(df)
