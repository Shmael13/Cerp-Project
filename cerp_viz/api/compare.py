"""
Compare endpoint: build the same chart config against two different
(file_id, sheet) sources and return both figures side-by-side.

Supports:
  - Within-file comparison: same file, different sheets
  - Cross-file comparison: different files, same or different sheets

POST /api/compare
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from cerp_viz.api import session as store
from cerp_viz.api.build_helpers import build_one

router = APIRouter(prefix="/api", tags=["compare"])


class CompareSource(BaseModel):
    file_id: str | None = None
    sheet: str
    label: str = ""


class CompareRequest(BaseModel):
    session_id: str
    source_a: CompareSource
    source_b: CompareSource
    chart_name: str
    columns: dict[str, Any]
    params: dict[str, Any] = {}
    transforms: dict[str, Any] = {}
    theme: str = "Light"


@router.post("/compare")
async def compare(req: CompareRequest):
    sess = store.require(req.session_id)

    df_a = store.resolve_df(sess, req.source_a.file_id, req.source_a.sheet)
    df_b = store.resolve_df(sess, req.source_b.file_id, req.source_b.sheet)

    label_a = req.source_a.label or req.source_a.sheet
    label_b = req.source_b.label or req.source_b.sheet

    fig_a, warn_a = build_one(
        df=df_a,
        chart_name=req.chart_name,
        columns=req.columns,
        raw_params=req.params,
        theme_name=req.theme,
        title=label_a,
        transforms=req.transforms or None,
    )
    fig_b, warn_b = build_one(
        df=df_b,
        chart_name=req.chart_name,
        columns=req.columns,
        raw_params=req.params,
        theme_name=req.theme,
        title=label_b,
        transforms=req.transforms or None,
    )

    return {
        "a": {"label": label_a, "figure": fig_a, "warnings": warn_a,
              "file_id": req.source_a.file_id, "sheet": req.source_a.sheet},
        "b": {"label": label_b, "figure": fig_b, "warnings": warn_b,
              "file_id": req.source_b.file_id, "sheet": req.source_b.sheet},
    }
