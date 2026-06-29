from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from cerp_viz.loaders.csv import CSVLoader
from cerp_viz.loaders.excel import ExcelLoader


@st.cache_data(show_spinner=False)
def _load_bytes(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    loader = CSVLoader() if file_name.lower().endswith(".csv") else ExcelLoader()
    return loader.load(io.BytesIO(file_bytes))


def _select_sheet(sheets: dict[str, pd.DataFrame], label: str, key: str) -> pd.DataFrame:
    if len(sheets) > 1:
        name = st.selectbox(label, list(sheets.keys()), key=key)
    else:
        name = list(sheets.keys())[0]
        st.caption(f"Sheet: **{name}**")
    return sheets[name]


def _merge_all(
    primary: pd.DataFrame,
    extras: list[tuple[str, pd.DataFrame]],
    join_key: str,
    join_type: str,
) -> pd.DataFrame:
    result = primary
    for i, (_, df) in enumerate(extras):
        result = result.merge(df, on=join_key, how=join_type, suffixes=("", f"_{i + 2}"))
    return result


def render_multifile_panel(primary_df: pd.DataFrame) -> None:
    with st.sidebar.expander("🔗 Multi-file Join", expanded=False):
        if "merged_df" in st.session_state:
            mdf = st.session_state["merged_df"]
            st.success(f"✓ Active: {len(mdf):,} rows × {len(mdf.columns)} cols")
            if st.button("✕ Clear merge", key="mf_clear", use_container_width=True):
                del st.session_state["merged_df"]
                st.session_state.pop("mf_error", None)
                st.rerun()
            st.divider()

        st.caption("Upload additional Excel/CSV files to join with the primary data.")
        extra_files = st.file_uploader(
            "Additional files",
            type=["xlsx", "xls", "xlsm", "csv"],
            accept_multiple_files=True,
            key="mf_uploads",
            label_visibility="collapsed",
        )

        if not extra_files:
            return

        extra_dfs: list[tuple[str, pd.DataFrame]] = []
        for i, f in enumerate(extra_files):
            try:
                sheets = _load_bytes(f.getvalue(), f.name)
                df = _select_sheet(sheets, f.name, key=f"mf_sheet_{i}")
                extra_dfs.append((f.name, df))
            except Exception as exc:
                st.warning(f"{f.name}: {exc}")

        if not extra_dfs:
            return

        all_col_sets = [set(primary_df.columns)] + [set(df.columns) for _, df in extra_dfs]
        common_cols  = sorted(set.intersection(*all_col_sets))

        if not common_cols:
            st.warning("No common columns found across files — cannot join.")
            return

        join_key  = st.selectbox("Join key",  common_cols, key="mf_join_key")
        join_type = st.selectbox("Join type", ["left", "inner", "right", "outer"],
                                 key="mf_join_type")

        if st.button("🔗 Merge files", type="primary", use_container_width=True, key="mf_merge"):
            try:
                merged = _merge_all(primary_df, extra_dfs, join_key, join_type)
                st.session_state["merged_df"] = merged
                st.session_state.pop("mf_error", None)
                st.rerun()
            except Exception as exc:
                st.session_state["mf_error"] = str(exc)

        if err := st.session_state.get("mf_error"):
            st.error(f"Merge failed: {err}")
