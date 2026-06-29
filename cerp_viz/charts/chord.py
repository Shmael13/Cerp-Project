from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

_GAP = 0.03  # radians between node arcs


# ── Pure geometry helpers ──────────────────────────────────────────────────────

def _arc_pts(a1: float, a2: float, r: float = 1.0, n: int = 30) -> tuple[list, list]:
    if abs(a2 - a1) < 1e-6:
        return [math.cos(a1) * r], [math.sin(a1) * r]
    t = np.linspace(a1, a2, n)
    return (np.cos(t) * r).tolist(), (np.sin(t) * r).tolist()


def _bezier_quad(p0: tuple, ctrl: tuple, p2: tuple, n: int = 40) -> tuple[list, list]:
    t = np.linspace(0, 1, n)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p2[1]
    return x.tolist(), y.tolist()


def _chord_outline(sa1: float, sa2: float, ta1: float, ta2: float, r: float = 0.97) -> tuple[list, list]:
    ctrl = (0.0, 0.0)
    pt = lambda a: (math.cos(a) * r, math.sin(a) * r)

    xa, ya   = _arc_pts(sa1, sa2, r)
    bx1, by1 = _bezier_quad(pt(sa2), ctrl, pt(ta1))
    xb, yb   = _arc_pts(ta1, ta2, r)
    bx2, by2 = _bezier_quad(pt(ta2), ctrl, pt(sa1))

    x = xa + bx1 + xb + bx2 + [xa[0]]
    y = ya + by1 + yb + by2 + [ya[0]]
    return x, y


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Data helpers ───────────────────────────────────────────────────────────────

def _aggregate(
    df: pd.DataFrame, src_col: str, tgt_col: str, val_col: str
) -> dict[tuple[str, str], float]:
    work = df[[src_col, tgt_col, val_col]].copy()
    work[val_col] = pd.to_numeric(work[val_col], errors="coerce")
    work = work.dropna()
    work[src_col] = work[src_col].astype(str)
    work[tgt_col] = work[tgt_col].astype(str)
    work = work[work[src_col] != work[tgt_col]]
    agg = work.groupby([src_col, tgt_col])[val_col].sum().reset_index()
    agg = agg[agg[val_col] > 0]
    return {(row[src_col], row[tgt_col]): float(row[val_col]) for _, row in agg.iterrows()}


def _node_totals(flows: dict, nodes: list[str]) -> dict[str, float]:
    totals: dict[str, float] = {n: 0.0 for n in nodes}
    for (s, t), v in flows.items():
        totals[s] += v
        totals[t] += v
    return totals


def _compute_layout(
    flows: dict, nodes: list[str]
) -> tuple[dict[str, tuple[float, float]], dict[str, dict]]:
    totals = _node_totals(flows, nodes)
    grand  = sum(totals.values()) or 1.0
    n      = len(nodes)
    avail  = 2 * math.pi - _GAP * n

    node_arcs: dict[str, tuple[float, float]] = {}
    angle = 0.0
    for node in nodes:
        size = (totals[node] / grand) * avail
        node_arcs[node] = (angle, angle + size)
        angle += size + _GAP

    # Per-node, per-chord sub-arc allocation
    node_flow_lists: dict[str, list] = {n: [] for n in nodes}
    for key, v in sorted(flows.items()):
        s, t = key
        node_flow_lists[s].append((key, v))
        node_flow_lists[t].append((key, v))

    sub_arcs: dict[str, dict] = {}
    for node in nodes:
        start, end = node_arcs[node]
        total = totals[node] or 1.0
        sub_arcs[node] = {}
        offset = start
        for key, v in node_flow_lists[node]:
            span = (v / total) * (end - start)
            sub_arcs[node][key] = (offset, offset + span)
            offset += span

    return node_arcs, sub_arcs


def _top_n_filter(
    flows: dict, n: int
) -> tuple[dict, list[str]]:
    totals: dict[str, float] = {}
    for (s, t), v in flows.items():
        totals[s] = totals.get(s, 0) + v
        totals[t] = totals.get(t, 0) + v
    top = set(sorted(totals, key=totals.get, reverse=True)[:n])
    filtered = {(s, t): v for (s, t), v in flows.items() if s in top and t in top}
    nodes = sorted(set(s for s, _ in filtered) | set(t for _, t in filtered))
    return filtered, nodes


# ── Plotly trace builders ──────────────────────────────────────────────────────

def _ribbon_traces(
    flows: dict,
    sub_arcs: dict,
    color_map: dict[str, str],
    opacity: float,
) -> list[go.Scatter]:
    traces = []
    for (src, tgt), v in flows.items():
        if src not in sub_arcs or tgt not in sub_arcs:
            continue
        key = (src, tgt)
        sa1, sa2 = sub_arcs[src].get(key, (0.0, 0.0))
        ta1, ta2 = sub_arcs[tgt].get(key, (0.0, 0.0))
        x, y = _chord_outline(sa1, sa2, ta1, ta2)
        fill_color = _hex_to_rgba(color_map[src], opacity)
        traces.append(go.Scatter(
            x=x, y=y,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=fill_color, width=0.5),
            mode="lines",
            showlegend=False,
            hovertemplate=f"<b>{src} → {tgt}</b><br>Value: {v:,.2f}<extra></extra>",
        ))
    return traces


def _arc_traces(
    nodes: list[str],
    node_arcs: dict,
    totals: dict,
    color_map: dict[str, str],
) -> list[go.Scatter]:
    traces = []
    for node in nodes:
        a1, a2 = node_arcs[node]
        xs, ys = _arc_pts(a1, a2, r=1.0, n=60)
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(width=18, color=color_map[node]),
            name=node,
            showlegend=True,
            hovertemplate=f"<b>{node}</b><br>Total: {totals.get(node, 0):,.2f}<extra></extra>",
        ))
    return traces


def _label_annotations(nodes: list[str], node_arcs: dict) -> list[dict]:
    annotations = []
    for node in nodes:
        a1, a2 = node_arcs[node]
        mid = (a1 + a2) / 2
        r = 1.22
        annotations.append(dict(
            x=math.cos(mid) * r,
            y=math.sin(mid) * r,
            text=f"<b>{node}</b>",
            showarrow=False,
            font=dict(size=11),
            xanchor="center",
            yanchor="middle",
        ))
    return annotations


# ── Chart class ────────────────────────────────────────────────────────────────

class ChordDiagram(BaseVisualization):
    name: ClassVar[str] = "Chord Diagram"
    description: ClassVar[str] = "Flow ribbons between categories — size encodes magnitude, color encodes source."

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("source", "categorical", "Source"),
            ColumnSpec("target", "categorical", "Target"),
            ColumnSpec("value",  "numeric",     "Flow value"),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("top_n",         "number_input", "Max nodes (0 = all)", 0,
                           {"min": 0, "max": 30, "step": 1},     category="Data"),
            AssumptionSpec("chord_opacity",  "slider",       "Chord opacity",      0.45,
                           {"min": 0.05, "max": 0.9, "step": 0.05}, category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        src_col = columns["source"]
        tgt_col = columns["target"]
        val_col = columns["value"]
        warnings: list[str] = []

        flows = _aggregate(df, src_col, tgt_col, val_col)
        if not flows:
            return BuildResult(figure=go.Figure(), warnings=["No valid flows found after filtering."])

        nodes = sorted(set(s for s, _ in flows) | set(t for _, t in flows))

        top_n = int(params["top_n"])
        if top_n > 0 and len(nodes) > top_n:
            flows, nodes = _top_n_filter(flows, top_n)
            warnings.append(f"Showing top {top_n} nodes by total flow.")

        if not nodes:
            return BuildResult(figure=go.Figure(), warnings=["No nodes remain after filtering."])

        color_map = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(nodes)}
        node_arcs, sub_arcs = _compute_layout(flows, nodes)
        totals = _node_totals(flows, nodes)
        opacity = float(params["chord_opacity"])

        traces = (
            _ribbon_traces(flows, sub_arcs, color_map, opacity)
            + _arc_traces(nodes, node_arcs, totals, color_map)
        )

        fig = go.Figure(data=traces)
        fig.update_layout(
            template="plotly_white",
            annotations=_label_annotations(nodes, node_arcs),
            xaxis=dict(range=[-1.55, 1.55], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-1.55, 1.55], showgrid=False, zeroline=False, showticklabels=False,
                       scaleanchor="x", scaleratio=1),
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="v", x=1.02, y=0.5),
        )
        return BuildResult(figure=fig, warnings=warnings)


registry.register(ChordDiagram)
