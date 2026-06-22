"""
Simulation endpoint: apply numeric what-if rules to a dataset and
return the chart built from the modified data alongside the baseline.

POST /api/simulate        — baseline + simulated chart
GET  /api/simulate/ops    — list available operations
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from cerp_viz.api import session as store
from cerp_viz.api.build_helpers import build_one
from cerp_viz.core.simulate import SimRule, apply_rules, available_operations

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


class RuleIn(BaseModel):
    column: str
    operation: str
    value: float


class SimulateRequest(BaseModel):
    session_id: str
    file_id: str | None = None
    sheet: str
    chart_name: str
    columns: dict[str, Any]
    params: dict[str, Any] = {}
    transforms: dict[str, Any] = {}
    theme: str = "Light"
    rules: list[RuleIn] = []
    baseline_label: str = "Baseline"
    scenario_label: str = "Scenario"


@router.get("/ops")
async def list_ops():
    return {"operations": available_operations()}


@router.post("")
async def simulate(req: SimulateRequest):
    sess   = store.require(req.session_id)
    raw_df = store.resolve_df(sess, req.file_id, req.sheet)

    rules = [SimRule(column=r.column, operation=r.operation, value=r.value)
             for r in req.rules]
    sim_df = apply_rules(raw_df, rules)

    fig_base, warn_base = build_one(
        df=raw_df,
        chart_name=req.chart_name,
        columns=req.columns,
        raw_params=req.params,
        theme_name=req.theme,
        title=req.baseline_label,
        transforms=req.transforms or None,
    )
    fig_sim, warn_sim = build_one(
        df=sim_df,
        chart_name=req.chart_name,
        columns=req.columns,
        raw_params=req.params,
        theme_name=req.theme,
        title=req.scenario_label,
        transforms=req.transforms or None,
    )

    return {
        "baseline": {
            "label":    req.baseline_label,
            "figure":   fig_base,
            "warnings": warn_base,
        },
        "scenario": {
            "label":    req.scenario_label,
            "figure":   fig_sim,
            "warnings": warn_sim,
        },
        "rules_applied": len(rules),
    }
