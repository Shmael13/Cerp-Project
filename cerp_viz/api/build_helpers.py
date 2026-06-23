"""
Shared chart-build logic used by the single-chart, dashboard, compare, and
simulate endpoints.  Centralised here so that post-processing behaviour stays
consistent across all surfaces.
"""
from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from fastapi import HTTPException

import cerp_viz.charts        # noqa: F401 — side-effect: registers all chart types
from cerp_viz.core.registry import registry
from cerp_viz.core.theme import THEMES, apply_theme
from cerp_viz.core.transform import (
    DatePart, DerivedColumn, FilterRule, TopNRule, TransformConfig, apply_transforms,
)


# ── Legend / axis post-processing constants ────────────────────────────────────
_TRUNCATE_AT     = 22   # always truncate legend text beyond this length
_LEGEND_ABOVE_AT = 8    # switch to above-chart horizontal legend beyond this
_MAX_LEGEND_SHOW = 15   # cap visible entries; extras still plotted but unlabelled


# ── Column metadata ────────────────────────────────────────────────────────────

def col_meta(df: pd.DataFrame) -> list[dict]:
    """Describe every column's dtype and a representative sample value."""
    result = []
    for c in df.columns:
        k = df[c].dtype.kind
        if k in "biufc":
            t = "numeric"
        elif k in "Mm":
            t = "datetime"
        else:
            sample = df[c].dropna().head(5)
            try:
                pd.to_datetime(sample, errors="raise")
                t = "datetime"
            except Exception:
                t = "categorical"
        first = df[c].dropna()
        result.append({
            "name":   str(c),
            "type":   t,
            "sample": str(first.iloc[0]) if len(first) else "",
        })
    return result


# ── Transform config deserialisation ──────────────────────────────────────────

def to_transform_cfg(d: dict) -> TransformConfig:
    return TransformConfig(
        filters=[
            FilterRule(column=f["column"], operator=f["operator"], value=f.get("value", ""))
            for f in d.get("filters", []) if f.get("column")
        ],
        top_n=[
            TopNRule(
                cat_column=t["cat_column"],
                num_column=t["num_column"],
                n=int(t.get("n", 10)),
                agg=t.get("agg", "sum"),
            )
            for t in d.get("top_n", []) if t.get("cat_column") and t.get("num_column")
        ],
        date_parts=[
            DatePart(source_column=dp["source_column"], part=dp["part"])
            for dp in d.get("date_parts", []) if dp.get("source_column")
        ],
        derived_cols=[
            DerivedColumn(name=dc["name"], expression=dc["expression"])
            for dc in d.get("derived_cols", []) if dc.get("name") and dc.get("expression")
        ],
    )


# ── Figure post-processing ─────────────────────────────────────────────────────

def post_process_figure(fig: dict) -> tuple[dict, list[str]]:
    """Fix legend overflow and long axis label issues. Returns (fig, extra_warnings)."""
    layout         = fig.setdefault("layout", {})
    data           = fig.get("data", [])
    extra_warnings: list[str] = []

    named        = [t for t in data if t.get("name") and t.get("showlegend") is not False]
    n_named      = len(named)
    max_name_pre = max((len(str(t["name"])) for t in named), default=0)

    # Always truncate long names for a compact legend
    for t in named:
        name = str(t.get("name", ""))
        if len(name) > _TRUNCATE_AT:
            t["name"] = name[: _TRUNCATE_AT - 1] + "…"

    if n_named > _LEGEND_ABOVE_AT or (n_named > 4 and max_name_pre > _TRUNCATE_AT):
        # Place legend ABOVE the chart in horizontal orientation.
        # y=1.02 + yanchor="bottom" → entries grow upward into margin.t,
        # which is always within the SVG (no overflow:hidden clipping risk).
        if n_named > _MAX_LEGEND_SHOW:
            for t in named[_MAX_LEGEND_SHOW:]:
                t["showlegend"] = False
            extra_warnings.append(
                f"Legend shows {_MAX_LEGEND_SHOW} of {n_named} categories. "
                "Hover over chart elements to identify unlabelled series."
            )

        visible = min(n_named, _MAX_LEGEND_SHOW)
        n_rows  = max(1, math.ceil(visible / 5))

        layout["legend"] = {
            **layout.get("legend", {}),
            "orientation": "h",
            "yanchor":     "bottom",
            "y":           1.02,
            "xanchor":     "left",
            "x":           0,
            "font":        {"size": 10},
            "tracegroupgap": 0,
        }
        layout.setdefault("margin", {})["r"] = 20

        margin    = layout.setdefault("margin", {})
        has_title = bool((layout.get("title") or {}).get("text"))
        title_px  = 35 if has_title else 10
        legend_px = min(n_rows * 22, 130)
        margin["t"] = max(margin.get("t", 40), title_px + legend_px + 5)

    # Enable automargin on both axes
    for ax_key in ("xaxis", "yaxis"):
        layout.setdefault(ax_key, {})["automargin"] = True

    # Tilt long categorical X-axis labels
    xax = layout.get("xaxis", {})
    if "tickangle" not in xax:
        x_strings: list[str] = []
        for t in data:
            x_vals = t.get("x")
            if not isinstance(x_vals, list):
                continue
            for v in x_vals[:30]:
                if isinstance(v, str):
                    x_strings.append(v)
        if x_strings and max(len(s) for s in x_strings) > 12:
            xax["tickangle"] = -45

    return fig, extra_warnings


# ── Param coercion ─────────────────────────────────────────────────────────────

def coerce_params(viz: Any, raw_params: dict) -> dict:
    """Coerce raw (string) params to the types declared in the chart's assumptions."""
    spec_map = {a.key: a for a in viz.assumptions()}
    coerced: dict[str, Any] = {}
    for k, v in raw_params.items():
        if k.startswith("_"):
            coerced[k] = v
            continue
        if k in spec_map:
            d = spec_map[k].default
            try:
                if isinstance(d, bool):
                    coerced[k] = bool(v) if not isinstance(v, str) else v.lower() == "true"
                elif isinstance(d, int):
                    coerced[k] = int(float(v))
                elif isinstance(d, float):
                    coerced[k] = float(v)
                elif isinstance(d, list):
                    coerced[k] = v if isinstance(v, list) else [v]
                else:
                    coerced[k] = v
            except (ValueError, TypeError):
                coerced[k] = v
        else:
            coerced[k] = v
    return coerced


# ── Core build function ────────────────────────────────────────────────────────

def build_one(
    df: pd.DataFrame,
    chart_name: str,
    columns: dict[str, Any],
    raw_params: dict[str, Any],
    theme_name: str = "Light",
    title: str = "",
    subtitle: str = "",
    transforms: dict | None = None,
) -> tuple[dict, list[str]]:
    """Build one chart and return (fig_dict, warnings).

    Raises HTTPException on unknown chart name or build failure so callers
    don't need to repeat error-handling boilerplate.
    """
    # Apply transforms
    if transforms:
        df, t_warnings = apply_transforms(df, to_transform_cfg(transforms))
    else:
        t_warnings = []

    # Resolve viz
    try:
        viz = registry.get(chart_name)()
    except KeyError:
        raise HTTPException(400, f"Unknown chart type: '{chart_name}'")

    # Merge defaults ← coerced params (every key present; chart.build() needs it)
    defaults    = {a.key: a.default for a in viz.assumptions()}
    coerced     = coerce_params(viz, raw_params)
    full_params = {**defaults, **coerced}

    # Build
    try:
        result = viz.build(df, columns, full_params)
    except Exception as exc:
        raise HTTPException(400, str(exc))

    # Theme
    apply_theme(result.figure, THEMES.get(theme_name, list(THEMES.values())[0]))

    # Title / subtitle
    title    = str(coerced.get("_chart_title",    title)).strip()
    subtitle = str(coerced.get("_chart_subtitle", subtitle)).strip()
    if title:
        result.figure.update_layout(title_text=title, title_font_size=18)
    if subtitle:
        result.figure.add_annotation(
            text=f"<i>{subtitle}</i>", xref="paper", yref="paper",
            x=0, y=1.04, xanchor="left", yanchor="bottom",
            showarrow=False, font=dict(size=12, color="#666"),
        )

    # Serialise — strip fixed dimensions so browser controls sizing
    fig_dict = json.loads(result.figure.to_json())
    fig_dict["layout"].pop("width",   None)
    fig_dict["layout"].pop("height",  None)
    fig_dict["layout"].pop("autosize", None)
    fig_dict, legend_warnings = post_process_figure(fig_dict)

    all_warnings = list(t_warnings) + list(result.warnings) + legend_warnings
    return fig_dict, all_warnings
