"""
In-memory session store.

Designed so the store itself can be replaced with a Redis or disk adapter
without touching any other file — just swap _store for an appropriate backend
that satisfies the same interface.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from fastapi import HTTPException


# ── Domain models ──────────────────────────────────────────────────────────────

@dataclass
class FileEntry:
    """One uploaded file's parsed content."""
    file_id: str
    name:    str
    sheets:  dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class PanelConfig:
    """Configuration for one chart panel inside a dashboard."""
    panel_id:   str
    file_id:    str
    sheet:      str
    chart_name: str
    columns:    dict[str, Any]
    params:     dict[str, Any]
    transforms: dict[str, Any]  = field(default_factory=dict)
    theme:      str              = "Light"
    title:      str              = ""


@dataclass
class DashboardState:
    """A named collection of panels arranged in a layout grid."""
    dashboard_id: str
    name:         str
    layout:       str                 = "2-col"   # 1-col | 2-col | 3-col | 2x2
    panels:       list[PanelConfig]   = field(default_factory=list)


@dataclass
class SessionData:
    """All mutable state for one browser session."""
    session_id: str
    files:      dict[str, FileEntry]      = field(default_factory=dict)
    dashboards: dict[str, DashboardState] = field(default_factory=dict)


# ── Store ──────────────────────────────────────────────────────────────────────

_store: dict[str, SessionData] = {}


def create(session_id: str | None = None) -> SessionData:
    sid = session_id or str(uuid.uuid4())
    _store[sid] = SessionData(session_id=sid)
    return _store[sid]


def get(session_id: str) -> SessionData | None:
    return _store.get(session_id)


def require(session_id: str) -> SessionData:
    """Return session or raise HTTP 404."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(404, "Session expired — please re-upload your file.")
    return s


# ── File helpers ───────────────────────────────────────────────────────────────

def add_file(
    session: SessionData,
    name: str,
    sheets: dict[str, pd.DataFrame],
) -> FileEntry:
    fid = str(uuid.uuid4())
    fe  = FileEntry(file_id=fid, name=name, sheets=sheets)
    session.files[fid] = fe
    return fe


def require_file(session: SessionData, file_id: str) -> FileEntry:
    fe = session.files.get(file_id)
    if fe is None:
        raise HTTPException(404, f"File '{file_id}' not found in session.")
    return fe


def require_sheet(fe: FileEntry, sheet: str) -> pd.DataFrame:
    df = fe.sheets.get(sheet)
    if df is None:
        raise HTTPException(404, f"Sheet '{sheet}' not found in file '{fe.name}'.")
    return df


def resolve_df(
    session: SessionData,
    file_id: str | None,
    sheet: str,
) -> pd.DataFrame:
    """Resolve (file_id, sheet) → DataFrame.

    If file_id is None, searches all files for the sheet name so that
    existing callers that only know the sheet name continue to work.
    """
    if file_id:
        fe = require_file(session, file_id)
        return require_sheet(fe, sheet)

    # Backward-compat: search every file in insertion order
    for fe in session.files.values():
        if sheet in fe.sheets:
            return fe.sheets[sheet]
    raise HTTPException(404, f"Sheet '{sheet}' not found in any uploaded file.")


# ── Dashboard helpers ──────────────────────────────────────────────────────────

def require_dashboard(session: SessionData, dashboard_id: str) -> DashboardState:
    d = session.dashboards.get(dashboard_id)
    if d is None:
        raise HTTPException(404, f"Dashboard '{dashboard_id}' not found.")
    return d


def require_panel(dashboard: DashboardState, panel_id: str) -> PanelConfig:
    for p in dashboard.panels:
        if p.panel_id == panel_id:
            return p
    raise HTTPException(404, f"Panel '{panel_id}' not found.")
