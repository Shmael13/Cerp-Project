from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False


def _spring_layout(nodes: list, edges: list[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    if not _NX:
        rng = np.random.default_rng(42)
        return {n: (float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))) for n in nodes}
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    return {n: (float(pos[n][0]), float(pos[n][1])) for n in nodes}


def _build_graph_data(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str | None,
    max_nodes: int,
) -> tuple[list[str], dict[tuple[str, str], float], list[str]]:
    work = df[[c for c in [source_col, target_col, weight_col] if c]].dropna()

    if weight_col:
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(1.0)
        agg = work.groupby([source_col, target_col])[weight_col].sum().reset_index()
    else:
        agg = work.groupby([source_col, target_col]).size().reset_index(name="_w")
        weight_col = "_w"

    all_nodes = sorted(set(agg[source_col]) | set(agg[target_col]))
    warnings: list[str] = []

    if len(all_nodes) > max_nodes:
        total_node_count = len(all_nodes)
        top = (
            agg.groupby(source_col)[weight_col].sum()
            .nlargest(max_nodes)
            .index.tolist()
        )
        agg = agg[agg[source_col].isin(top) & agg[target_col].isin(top)]
        all_nodes = sorted(set(agg[source_col]) | set(agg[target_col]))
        warnings.append(f"Showing top {max_nodes} of {total_node_count} nodes by total weight.")

    edge_weights = {
        (row[source_col], row[target_col]): float(row[weight_col])
        for _, row in agg.iterrows()
    }
    return all_nodes, edge_weights, warnings


def _edge_traces(
    edge_weights: dict[tuple[str, str], float],
    pos: dict[str, tuple[float, float]],
    n_buckets: int = 5,
) -> list[go.Scatter]:
    if not edge_weights:
        return []

    weights = list(edge_weights.values())
    w_min, w_max = min(weights), max(weights)
    w_range = w_max - w_min or 1.0

    def _bucket(w: float) -> int:
        return min(int((w - w_min) / w_range * n_buckets), n_buckets - 1)

    buckets: dict[int, tuple[list, list, list]] = {i: ([], [], []) for i in range(n_buckets)}
    for (u, v), w in edge_weights.items():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        b = _bucket(w)
        buckets[b][0].extend([x0, x1, None])
        buckets[b][1].extend([y0, y1, None])
        buckets[b][2].append(f"{u} → {v}: {w:.2f}")

    traces = []
    for b, (xs, ys, labels) in buckets.items():
        if not xs:
            continue
        width = 1 + b * (4 / max(n_buckets - 1, 1))
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(width=width, color="rgba(120,120,120,0.45)"),
            hoverinfo="none",
            showlegend=False,
        ))
    return traces


def _node_trace(
    nodes: list[str],
    pos: dict[str, tuple[float, float]],
    edge_weights: dict[tuple[str, str], float],
    node_color: str,
) -> go.Scatter:
    degree: dict[str, float] = {n: 0.0 for n in nodes}
    for (u, v), w in edge_weights.items():
        degree[u] = degree.get(u, 0.0) + w
        degree[v] = degree.get(v, 0.0) + w

    max_deg = max(degree.values()) or 1.0
    sizes = [10 + 30 * (degree[n] / max_deg) for n in nodes]

    return go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=node_color,
            line=dict(width=1.5, color="white"),
        ),
        text=nodes,
        textposition="top center",
        textfont=dict(size=11),
        hovertemplate="<b>%{text}</b><br>Weight: %{customdata:.2f}<extra></extra>",
        customdata=[degree[n] for n in nodes],
        showlegend=False,
    )


_NODE_COLORS = {
    "Blue":   "rgba(31,119,180,0.85)",
    "Green":  "rgba(44,160,44,0.85)",
    "Orange": "rgba(255,127,14,0.85)",
    "Purple": "rgba(148,103,189,0.85)",
    "Red":    "rgba(214,39,40,0.85)",
    "Teal":   "rgba(23,190,207,0.85)",
}


class NetworkGraph(BaseVisualization):
    name: ClassVar[str] = "Network Graph"
    description: ClassVar[str] = "Visualize relationships between entities as a force-directed node-link diagram."

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("source", "categorical", "Source node"),
            ColumnSpec("target", "categorical", "Target node"),
            ColumnSpec("weight", "numeric",     "Edge weight (optional)", required=False),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("max_nodes",   "slider",    "Max nodes to display", 60,
                           {"min": 5, "max": 200, "step": 5},   category="Data"),
            AssumptionSpec("node_color",  "selectbox", "Node color", "Blue",
                           {"choices": list(_NODE_COLORS)},      category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        source_col = columns["source"]
        target_col = columns["target"]
        weight_col = columns.get("weight")
        warnings: list[str] = []

        nodes, edge_weights, build_warnings = _build_graph_data(
            df, source_col, target_col, weight_col, int(params["max_nodes"])
        )
        warnings.extend(build_warnings)

        if not nodes:
            return BuildResult(figure=go.Figure(), warnings=["No valid node data found."])

        edges = list(edge_weights.keys())
        pos = _spring_layout(nodes, edges)

        fig = go.Figure(
            data=_edge_traces(edge_weights, pos) + [_node_trace(nodes, pos, edge_weights, _NODE_COLORS[params["node_color"]])],
        )
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        if not _NX:
            warnings.append("networkx not installed — using random node placement. Install networkx for spring layout.")

        return BuildResult(figure=fig, warnings=warnings)


registry.register(NetworkGraph)
