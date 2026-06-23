"""
CERP Visualizer — FastAPI backend.
Exposes REST endpoints consumed by the single-page frontend.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import cerp_viz.charts        # noqa: F401 — triggers chart registration
import cerp_viz.suggestions   # noqa: F401 — triggers suggester registration
from cerp_viz.api import session as sess_store
from cerp_viz.api.build_helpers import build_one, col_meta, to_transform_cfg
from cerp_viz.api.dashboard import router as dashboard_router
from cerp_viz.api.compare   import router as compare_router
from cerp_viz.api.simulate  import router as simulate_router
from cerp_viz.core.compatibility import compatible_visualizations
from cerp_viz.core.registry import registry
from cerp_viz.core.theme import THEMES
from cerp_viz.core.transform import (
    apply_transforms, available_operators, available_date_parts,
)
from cerp_viz.suggestions import suggester_registry

app = FastAPI(title="CERP Visualizer")

# ── Register feature routers ───────────────────────────────────────────────────
app.include_router(dashboard_router)
app.include_router(compare_router)
app.include_router(simulate_router)


# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data  = await file.read()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".csv"):
            sheets = {"Sheet1": pd.read_csv(io.BytesIO(data))}
        else:
            xf     = pd.ExcelFile(io.BytesIO(data))
            sheets = {n: xf.parse(n) for n in xf.sheet_names}
    except Exception as exc:
        raise HTTPException(400, f"Cannot read file: {exc}")

    # Ensure a session exists — create a fresh one per upload
    sess = sess_store.create()
    fe   = sess_store.add_file(sess, file.filename or "upload", sheets)

    first = list(sheets)[0]
    return {
        "session_id":  sess.session_id,
        "file_id":     fe.file_id,
        "file_name":   fe.name,
        "sheet_names": list(sheets.keys()),
        "active_sheet": first,
        "cols":        col_meta(sheets[first]),
        "rows":        len(sheets[first]),
    }


@app.post("/api/upload_additional")
async def upload_additional(
    session_id: str,
    file: UploadFile = File(...),
):
    """Add another file to an existing session (multi-file support)."""
    sess  = sess_store.require(session_id)
    data  = await file.read()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".csv"):
            sheets = {"Sheet1": pd.read_csv(io.BytesIO(data))}
        else:
            xf     = pd.ExcelFile(io.BytesIO(data))
            sheets = {n: xf.parse(n) for n in xf.sheet_names}
    except Exception as exc:
        raise HTTPException(400, f"Cannot read file: {exc}")

    fe    = sess_store.add_file(sess, file.filename or "upload", sheets)
    first = list(sheets)[0]
    return {
        "file_id":     fe.file_id,
        "file_name":   fe.name,
        "sheet_names": list(sheets.keys()),
        "active_sheet": first,
        "cols":        col_meta(sheets[first]),
        "rows":        len(sheets[first]),
    }


@app.get("/api/session/{session_id}/files")
async def list_files(session_id: str):
    """Return all files currently loaded in this session."""
    sess = sess_store.require(session_id)
    return {"files": [
        {
            "file_id":    fe.file_id,
            "name":       fe.name,
            "sheet_names": list(fe.sheets.keys()),
        }
        for fe in sess.files.values()
    ]}


@app.get("/api/sheet/{sid}/{sheet:path}")
async def get_sheet(sid: str, sheet: str, file_id: str | None = None):
    sess = sess_store.require(sid)
    df   = sess_store.resolve_df(sess, file_id, sheet)
    return {"cols": col_meta(df), "rows": len(df)}


# ── Catalogue ──────────────────────────────────────────────────────────────────

@app.get("/api/charts")
async def list_charts():
    return {"charts": [
        {"name": n, "description": registry.get(n)().description}
        for n in registry.names()
    ]}


@app.get("/api/specs/{chart_name:path}")
async def get_specs(chart_name: str):
    try:
        viz = registry.get(chart_name)()
    except KeyError:
        raise HTTPException(404, "Unknown chart type")
    return {
        "columns": [
            {"role": c.role, "dtype": c.dtype, "label": c.label, "required": c.required}
            for c in viz.required_columns()
        ],
        "assumptions": [
            {"key": a.key, "widget": a.widget, "label": a.label,
             "default": a.default, "options": a.options, "category": a.category}
            for a in viz.assumptions()
        ],
    }


# ── Build (single chart) ───────────────────────────────────────────────────────

class BuildRequest(BaseModel):
    session_id: str
    sheet_name: str
    chart_name: str
    columns: dict[str, Any]
    params: dict[str, Any]
    theme: str = "Light"
    transforms: dict[str, Any] = {}
    file_id: str | None = None   # optional; backward compat: None → search by sheet name


@app.post("/api/build")
async def build(req: BuildRequest):
    sess   = sess_store.require(req.session_id)
    raw_df = sess_store.resolve_df(sess, req.file_id, req.sheet_name)

    fig, warnings = build_one(
        df=raw_df,
        chart_name=req.chart_name,
        columns=req.columns,
        raw_params=req.params,
        theme_name=req.theme,
        transforms=req.transforms or None,
    )
    return {"figure": fig, "warnings": warnings}


# ── Suggest ────────────────────────────────────────────────────────────────────

class SuggestRequest(BaseModel):
    session_id: str
    sheet_name: str
    engine: str = "Smart (Statistical + Rule-Based)"
    transforms: dict[str, Any] = {}
    file_id: str | None = None


@app.post("/api/suggest")
async def suggest(req: SuggestRequest):
    sess   = sess_store.require(req.session_id)
    raw_df = sess_store.resolve_df(sess, req.file_id, req.sheet_name)
    df, _  = apply_transforms(raw_df, to_transform_cfg(req.transforms))

    eng = suggester_registry.get(req.engine)
    if eng is None:
        raise HTTPException(400, "Unknown engine")

    compat    = compatible_visualizations(df)
    available = {n for n, r in compat.items() if r.compatible}
    suggestions = eng.suggest(df)

    return {"suggestions": [
        {
            "chart_name": s.chart_name,
            "title":      s.title,
            "rationale":  s.rationale,
            "score":      round(s.score, 3),
            "columns":    s.columns,
            "params":     s.params,
            "transforms": s.transforms,
            "available":  s.chart_name in available,
        }
        for s in suggestions
    ]}


# ── AI interpret ──────────────────────────────────────────────────────────────

class InterpretRequest(BaseModel):
    session_id: str
    sheet_name: str
    chart_name: str
    columns: dict[str, Any]
    params: dict[str, Any] = {}
    warnings: list[str] = []
    file_id: str | None = None


@app.post("/api/interpret")
async def interpret(req: InterpretRequest):
    from cerp_viz.ai.groq_client import is_available
    if not is_available():
        raise HTTPException(503, "AI interpretation unavailable — GROQ_API_KEY not configured.")

    sess   = sess_store.require(req.session_id)
    df     = sess_store.resolve_df(sess, req.file_id, req.sheet_name)
    num_df = df.select_dtypes(include="number")

    df_stats: dict = {}
    for col in req.columns.values():
        if col and col in num_df.columns:
            s = num_df[col].describe()
            df_stats[col] = {"min": float(s["min"]), "max": float(s["max"]), "mean": float(s["mean"])}

    try:
        viz = registry.get(req.chart_name)()
    except KeyError:
        raise HTTPException(400, f"Unknown chart: {req.chart_name}")

    from cerp_viz.ai.groq_interpreter import interpret_chart
    try:
        text = interpret_chart(
            viz_name=req.chart_name,
            viz_description=viz.description,
            columns=req.columns,
            params=req.params,
            df_stats=df_stats,
            warnings=req.warnings,
        )
    except Exception as exc:
        raise HTTPException(500, f"AI interpretation failed: {exc}")

    return {"interpretation": text}


@app.get("/api/ai/status")
async def ai_status():
    from cerp_viz.ai.groq_client import is_available
    return {"available": is_available()}


# ── Meta ───────────────────────────────────────────────────────────────────────

@app.get("/api/engines")
async def list_engines():
    return {"engines": suggester_registry.names()}


@app.get("/api/themes")
async def list_themes():
    return {"themes": list(THEMES.keys())}


@app.get("/api/operators")
async def list_operators():
    return {"operators": available_operators(), "date_parts": available_date_parts()}


# ── SPA entry point ────────────────────────────────────────────────────────────

@app.get("/")
async def spa():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
