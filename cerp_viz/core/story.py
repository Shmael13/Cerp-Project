"""
Statistical story generator — derives plain-English observations from a DataFrame
and chart configuration.  No Streamlit, no Plotly, no external API.
Returns a list of bullet strings; callers decide how to display them.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_story_bullets(
    df: pd.DataFrame,
    chart_name: str,
    col_mapping: dict[str, str | None],
    params: dict[str, Any],
) -> list[str]:
    """
    Produce up to 6 plain-English observations relevant to the current chart type.
    Never raises — returns [] on any error.
    """
    try:
        bullets: list[str] = []
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # ── Generic numeric observations ───────────────────────────────────
        for role in ["y", "value", "actual"]:
            col = col_mapping.get(role)
            if col and col in df.columns and df[col].dtype.kind in "biufc":
                s = df[col].dropna()
                if len(s) < 2:
                    continue

                total   = s.sum()
                mean    = s.mean()
                median  = s.median()
                std     = s.std()
                skew    = float(s.skew())
                cv      = std / mean * 100 if mean != 0 else 0

                bullets.append(
                    f"**{col}**: total {total:,.0f}, mean {mean:,.1f}, "
                    f"median {median:,.1f}, SD {std:,.1f} (CV {cv:.0f}%)"
                )

                # Skew direction
                if abs(skew) > 0.5:
                    direction = "right-skewed (long upper tail)" if skew > 0 else "left-skewed (long lower tail)"
                    bullets.append(f"Distribution of **{col}** is {direction} (skewness {skew:+.2f}).")

                # Max/min categories
                cat_col = col_mapping.get("x") or col_mapping.get("names") or col_mapping.get("label")
                if cat_col and cat_col in df.columns:
                    agg = df.groupby(cat_col)[col].sum()
                    top_cat = agg.idxmax()
                    bot_cat = agg.idxmin()
                    pct_top = agg.max() / agg.sum() * 100 if agg.sum() != 0 else 0
                    bullets.append(
                        f"**{top_cat}** leads with {agg.max():,.1f} "
                        f"({pct_top:.0f}% of total {col})."
                    )
                    if agg.max() != agg.min():
                        bullets.append(
                            f"**{bot_cat}** is lowest at {agg.min():,.1f} — "
                            f"{agg.max() / max(agg.min(), 0.001):.1f}× below the leader."
                        )
                break  # only process first matched role

        # ── Pareto / concentration ──────────────────────────────────────────
        cat_col = col_mapping.get("x") or col_mapping.get("names")
        val_col = col_mapping.get("y") or col_mapping.get("value") or col_mapping.get("actual")
        if cat_col and val_col and cat_col in df.columns and val_col in df.columns:
            agg = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
            total = agg.sum()
            if total > 0 and len(agg) >= 3:
                n_top = max(1, int(len(agg) * 0.2))
                top_pct = agg.iloc[:n_top].sum() / total * 100
                if top_pct >= 60:
                    bullets.append(
                        f"**Pareto effect**: top {n_top} of {len(agg)} {cat_col} categories "
                        f"account for {top_pct:.0f}% of total {val_col}."
                    )

        # ── Trend direction (if datetime axis) ─────────────────────────────
        x_col = col_mapping.get("x")
        if x_col and x_col in df.columns and val_col and val_col in df.columns:
            try:
                x_parsed = pd.to_datetime(df[x_col], errors="coerce")
                if x_parsed.notna().sum() > 3:
                    ordered = df.assign(_x=x_parsed).sort_values("_x").dropna(subset=[val_col])
                    y_vals  = pd.to_numeric(ordered[val_col], errors="coerce").dropna().values
                    if len(y_vals) >= 4:
                        slope = np.polyfit(range(len(y_vals)), y_vals, 1)[0]
                        pct   = slope / abs(y_vals.mean()) * 100 if y_vals.mean() != 0 else 0
                        direction = "upward" if slope > 0 else "downward"
                        bullets.append(
                            f"**Trend**: {val_col} shows a {direction} trend "
                            f"({pct:+.1f}% per period on average)."
                        )
            except Exception:
                pass

        # ── Mixed-sign / waterfall alert ───────────────────────────────────
        if val_col and val_col in df.columns:
            s = pd.to_numeric(df[val_col], errors="coerce").dropna()
            if (s > 0).any() and (s < 0).any():
                pos = s[s > 0].sum()
                neg = s[s < 0].sum()
                net = pos + neg
                bullets.append(
                    f"**Mixed signs**: positive contributions sum to {pos:,.1f}, "
                    f"negatives to {neg:,.1f}, giving a net of {net:,.1f}."
                )

        # ── Missing data alert ─────────────────────────────────────────────
        total_cells  = df.shape[0] * df.shape[1]
        missing      = int(df.isnull().sum().sum())
        missing_pct  = missing / total_cells * 100 if total_cells > 0 else 0
        if missing_pct > 5:
            bullets.append(
                f"⚠️ **Data quality**: {missing:,} missing cells ({missing_pct:.0f}% of dataset) — "
                "interpret results with caution."
            )

        return bullets[:6]  # cap at 6 to avoid overwhelming

    except Exception:
        return []
