from typing import Any
import io

import pandas as pd
import streamlit as st

from cerp_viz.core.models import ColumnSpec
from cerp_viz.core.registry import registry
from cerp_viz.loaders.csv import CSVLoader
from cerp_viz.loaders.excel import ExcelLoader


@st.cache_data(show_spinner="Reading file…")
def _load_bytes(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    """Parse file bytes into sheets. Cached by content so re-parses only on new upload."""
    loader = CSVLoader() if file_name.lower().endswith(".csv") else ExcelLoader()
    return loader.load(io.BytesIO(file_bytes))


def render_upload() -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    """File upload + sheet picker. Returns (sheets_dict, selected_sheet)."""
    uploaded = st.sidebar.file_uploader(
        "Upload file", type=["xlsx", "xls", "xlsm", "csv"]
    )
    if uploaded is None:
        return None, None

    try:
        sheets = _load_bytes(uploaded.read(), uploaded.name)
    except Exception as exc:
        st.sidebar.error(f"Could not read file: {exc}")
        return None, None

    if len(sheets) > 1:
        sheet_name = st.sidebar.selectbox("Sheet", list(sheets.keys()))
    else:
        sheet_name = list(sheets.keys())[0]

    return sheets, sheet_name


def render_chart_picker(available_names: list[str]) -> str:
    # key="chart_picker" lets suggestions drive the selection via session_state
    return st.sidebar.selectbox("Visualization", available_names, key="chart_picker")


def render_column_mapping(
    df: pd.DataFrame, column_specs: list[ColumnSpec]
) -> dict[str, str | None]:
    """
    Renders a selectbox per ColumnSpec. Optional columns include a '(none)' choice.
    Returns a mapping of role → column name (or None if unselected).
    """
    st.sidebar.markdown("**── Column Mapping ──**")

    import re as _re
    _DATE_HINTS = _re.compile(r"date|time|day|month|year|start|end|due|deadline|creat|open|clos|finish", _re.I)

    all_cols      = list(df.columns)
    numeric_cols  = list(df.select_dtypes(include="number").columns)
    cat_cols      = list(df.select_dtypes(exclude="number").columns)
    dt_cols       = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    for _col in df.select_dtypes(include="object").columns:
        if _col not in dt_cols and _DATE_HINTS.search(_col):
            _sample = df[_col].dropna().head(5)
            if len(_sample) > 0:
                try:
                    pd.to_datetime(_sample, errors="raise")
                    dt_cols.append(_col)
                except Exception:
                    pass

    dtype_map: dict[str, list[str]] = {
        "numeric":     numeric_cols  or all_cols,
        "categorical": cat_cols      or all_cols,
        "datetime":    dt_cols       or all_cols,
        "any":         all_cols,
    }

    mapping: dict[str, str | None] = {}
    used_required: set[str] = set()

    for spec in column_specs:
        choices = list(dtype_map.get(spec.dtype, all_cols))

        if spec.required:
            # Remove columns already committed to another required role so
            # two required pickers can never default to the same column.
            choices = [c for c in choices if c not in used_required] or choices
        else:
            choices = ["(none)"] + choices

        selected = st.sidebar.selectbox(
            spec.label,
            choices,
            key=f"col_{spec.role}",
        )
        mapping[spec.role] = None if selected == "(none)" else selected

        if spec.required and selected and selected != "(none)":
            used_required.add(selected)

    return mapping
