"""
Dashboard CRUD + build router.

Endpoints:
  POST /api/dashboards/create        — create a new dashboard
  GET  /api/dashboards               — list all dashboards in a session
  POST /api/dashboards/{id}/panels   — add a panel
  PUT  /api/dashboards/{id}/panels/{pid} — update a panel
  DELETE /api/dashboards/{id}/panels/{pid} — remove a panel
  PUT  /api/dashboards/{id}/layout   — change layout
  POST /api/dashboards/{id}/build    — render all panels → list of figures
  DELETE /api/dashboards/{id}        — delete dashboard
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from cerp_viz.api import session as store
from cerp_viz.api.build_helpers import build_one

router = APIRouter(prefix="/api/dashboards", tags=["dashboards"])


# ── Request models ─────────────────────────────────────────────────────────────

class CreateDashboardReq(BaseModel):
    session_id: str
    name: str
    layout: str = "2-col"


class AddPanelReq(BaseModel):
    session_id: str
    file_id: str
    sheet: str
    chart_name: str
    columns: dict[str, Any]
    params: dict[str, Any] = {}
    transforms: dict[str, Any] = {}
    theme: str = "Light"
    title: str = ""


class UpdatePanelReq(BaseModel):
    session_id: str
    file_id: str | None = None
    sheet: str | None = None
    chart_name: str | None = None
    columns: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    transforms: dict[str, Any] | None = None
    theme: str | None = None
    title: str | None = None


class SetLayoutReq(BaseModel):
    session_id: str
    layout: str


class BuildDashboardReq(BaseModel):
    session_id: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/create")
async def create_dashboard(req: CreateDashboardReq):
    sess = store.require(req.session_id)
    did  = str(uuid.uuid4())
    dash = store.DashboardState(dashboard_id=did, name=req.name, layout=req.layout)
    sess.dashboards[did] = dash
    return {"dashboard_id": did, "name": dash.name, "layout": dash.layout}


@router.get("")
async def list_dashboards(session_id: str):
    sess = store.require(session_id)
    return {"dashboards": [
        {"dashboard_id": d.dashboard_id, "name": d.name, "layout": d.layout,
         "panel_count": len(d.panels)}
        for d in sess.dashboards.values()
    ]}


@router.delete("/{dashboard_id}")
async def delete_dashboard(dashboard_id: str, session_id: str):
    sess = store.require(session_id)
    store.require_dashboard(sess, dashboard_id)
    del sess.dashboards[dashboard_id]
    return {"deleted": dashboard_id}


@router.put("/{dashboard_id}/layout")
async def set_layout(dashboard_id: str, req: SetLayoutReq):
    sess = store.require(req.session_id)
    dash = store.require_dashboard(sess, dashboard_id)
    dash.layout = req.layout
    return {"dashboard_id": dashboard_id, "layout": dash.layout}


@router.post("/{dashboard_id}/panels")
async def add_panel(dashboard_id: str, req: AddPanelReq):
    sess = store.require(req.session_id)
    dash = store.require_dashboard(sess, dashboard_id)
    # Validate file + sheet exist
    store.resolve_df(sess, req.file_id, req.sheet)

    pid   = str(uuid.uuid4())
    panel = store.PanelConfig(
        panel_id=pid,
        file_id=req.file_id,
        sheet=req.sheet,
        chart_name=req.chart_name,
        columns=req.columns,
        params=req.params,
        transforms=req.transforms,
        theme=req.theme,
        title=req.title,
    )
    dash.panels.append(panel)
    return {"panel_id": pid}


@router.put("/{dashboard_id}/panels/{panel_id}")
async def update_panel(dashboard_id: str, panel_id: str, req: UpdatePanelReq):
    sess  = store.require(req.session_id)
    dash  = store.require_dashboard(sess, dashboard_id)
    panel = store.require_panel(dash, panel_id)

    if req.file_id   is not None: panel.file_id    = req.file_id
    if req.sheet     is not None: panel.sheet       = req.sheet
    if req.chart_name is not None: panel.chart_name = req.chart_name
    if req.columns   is not None: panel.columns     = req.columns
    if req.params    is not None: panel.params      = req.params
    if req.transforms is not None: panel.transforms = req.transforms
    if req.theme     is not None: panel.theme       = req.theme
    if req.title     is not None: panel.title       = req.title

    return {"panel_id": panel_id, "updated": True}


@router.delete("/{dashboard_id}/panels/{panel_id}")
async def remove_panel(dashboard_id: str, panel_id: str, session_id: str):
    sess  = store.require(session_id)
    dash  = store.require_dashboard(sess, dashboard_id)
    store.require_panel(dash, panel_id)  # validates existence
    dash.panels = [p for p in dash.panels if p.panel_id != panel_id]
    return {"deleted": panel_id}


@router.post("/{dashboard_id}/build")
async def build_dashboard(dashboard_id: str, req: BuildDashboardReq):
    sess = store.require(req.session_id)
    dash = store.require_dashboard(sess, dashboard_id)

    panels_out = []
    for panel in dash.panels:
        df = store.resolve_df(sess, panel.file_id, panel.sheet)
        fig, warnings = build_one(
            df=df,
            chart_name=panel.chart_name,
            columns=panel.columns,
            raw_params=panel.params,
            theme_name=panel.theme,
            title=panel.title,
            transforms=panel.transforms or None,
        )
        panels_out.append({
            "panel_id":   panel.panel_id,
            "title":      panel.title,
            "chart_name": panel.chart_name,
            "file_id":    panel.file_id,
            "sheet":      panel.sheet,
            "figure":     fig,
            "warnings":   warnings,
        })

    return {
        "dashboard_id": dashboard_id,
        "name":         dash.name,
        "layout":       dash.layout,
        "panels":       panels_out,
    }
