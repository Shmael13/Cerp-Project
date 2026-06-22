"""
Transform panel — sidebar expander for row filters, derived columns, date parts.
Returns a TransformConfig; the caller applies it to the DataFrame.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from cerp_viz.core.transform import (
    TransformConfig, FilterRule, DerivedColumn, DatePart,
    available_operators, available_date_parts,
)

_MAX_FILTERS      = 4
_MAX_DERIVED      = 3
_MAX_DATE_PARTS   = 4


def render_transform_panel(df: pd.DataFrame) -> TransformConfig:
    """Render the '── Transforms ──' expander in the sidebar. Returns a TransformConfig."""
    cfg = TransformConfig()

    with st.sidebar.expander("🔧 Data Transforms", expanded=False):
        all_cols     = list(df.columns)
        numeric_cols = list(df.select_dtypes(include="number").columns)
        dt_cols      = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        # Also detect object columns that look like dates
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(5)
            try:
                pd.to_datetime(sample, errors="raise")
                if col not in dt_cols:
                    dt_cols.append(col)
            except Exception:
                pass

        # ── Row Filters ───────────────────────────────────────────────────────
        st.markdown("**Row Filters**")
        n_filters = st.number_input(
            "Number of filters", min_value=0, max_value=_MAX_FILTERS,
            value=0, step=1, key="tf_n_filters"
        )
        ops = available_operators()

        for i in range(int(n_filters)):
            col_sel = st.selectbox(
                f"Filter {i+1} — column", all_cols, key=f"tf_f_col_{i}"
            )
            op_sel = st.selectbox(
                f"Filter {i+1} — operator", ops, key=f"tf_f_op_{i}"
            )
            val = "" if op_sel == "is blank" else st.text_input(
                f"Filter {i+1} — value", key=f"tf_f_val_{i}"
            )
            cfg.filters.append(FilterRule(column=col_sel, operator=op_sel, value=val))

        st.divider()

        # ── Derived Columns ───────────────────────────────────────────────────
        st.markdown("**Derived Columns**")
        st.caption("Use column names directly, e.g. `Margin = Revenue - Cost`")
        n_derived = st.number_input(
            "Number of derived columns", min_value=0, max_value=_MAX_DERIVED,
            value=0, step=1, key="tf_n_derived"
        )

        for i in range(int(n_derived)):
            name = st.text_input(f"Column {i+1} — name", key=f"tf_d_name_{i}")
            expr = st.text_input(f"Column {i+1} — formula", key=f"tf_d_expr_{i}")
            if name.strip() and expr.strip():
                cfg.derived_cols.append(DerivedColumn(name=name.strip(), expression=expr.strip()))

        # Show available column names as a hint
        if int(n_derived) > 0 and all_cols:
            st.caption(f"Available columns: `{'`, `'.join(all_cols)}`")

        st.divider()

        # ── Date Part Extraction ───────────────────────────────────────────────
        if dt_cols:
            st.markdown("**Date Part Extraction**")
            st.caption("Extract Year, Month, Quarter, etc. as new columns.")
            n_dp = st.number_input(
                "Number of date parts", min_value=0, max_value=_MAX_DATE_PARTS,
                value=0, step=1, key="tf_n_dp"
            )
            dp_options = available_date_parts()

            for i in range(int(n_dp)):
                src = st.selectbox(
                    f"Date part {i+1} — source column", dt_cols, key=f"tf_dp_col_{i}"
                )
                part = st.selectbox(
                    f"Date part {i+1} — extract", dp_options, key=f"tf_dp_part_{i}"
                )
                cfg.date_parts.append(DatePart(source_column=src, part=part))
        else:
            st.caption("_No datetime columns detected — date part extraction unavailable._")

    return cfg
