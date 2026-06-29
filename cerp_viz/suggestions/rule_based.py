"""
Rule-based chart suggester.

Heuristics based on column types, cardinalities, and name patterns.
Every builder explicitly sets ALL column roles (optional roles → None).
Results are deduplicated per chart type before returning.
Self-registers as "Rule-Based" in the suggester registry.
"""
from __future__ import annotations

import pandas as pd

from cerp_viz.core.suggestions import BaseSuggester, SuggestionResult
from cerp_viz.suggestions._utils import (
    NUMERIC_HINTS, STAGE_HINTS, FLOW_SRC_HINTS, FLOW_TGT_HINTS,
    GANTT_START_HINTS, GANTT_END_HINTS, GANTT_TASK_HINTS,
    numeric_cols, categorical_cols, datetime_cols,
    best_numeric, best_categorical, cardinality_score,
    has_mixed_sign, default_params, complete_columns, validate_and_complete,
    suggest_date_parts, suggest_outlier_filter, suggest_derived,
    suggest_top_n, suggest_ratio_derived, suggest_null_filter,
    filter_by_query,
)

KPI_HINTS    = __import__("re").compile(
    r"total|revenue|sales|profit|margin|count|amount|kpi|metric|volume|spend|budget", __import__("re").I
)
TARGET_HINTS = __import__("re").compile(r"target|goal|quota|budget|plan|forecast", __import__("re").I)


# ── Builders — each sets ALL roles for its chart type ────────────────────────

def _bar(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=8)
    if not num or not cat:
        return None
    n = df[cat].nunique()
    top_n = min(n, 10) if n > 10 else 0
    agg = "sum" if NUMERIC_HINTS.search(num) else "mean"
    # Bar chart has a built-in top_n param, so we don't add a separate top_n transform.
    # Only suggest outlier/null cleanup that the chart param cannot handle.
    transforms = suggest_outlier_filter(df, num) + suggest_null_filter(df, num)
    return SuggestionResult(
        chart_name="Bar Chart",
        columns=complete_columns("Bar Chart", x=cat, y=num, color=None),
        params={**default_params("Bar Chart"),
                "aggregation": agg, "sort_by": "Value (desc)", "top_n": top_n},
        title=f"{num} by {cat}",
        rationale=f"Ranks {n} {cat} categories by {agg} of {num} — shows which categories dominate.",
        score=0.60 + 0.25 * cardinality_score(n),
        transforms=transforms,
    )


def _line(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    if not num:
        return None
    dt = datetime_cols(df)
    if dt:
        x_col, score = dt[0], 0.88
        rationale = f"Tracks {num} over time — reveals trends, seasonality, and growth."
    else:
        others = [c for c in df.columns if c != num]
        if not others:
            return None
        x_col = max(others, key=lambda c: df[c].nunique())
        score, rationale = 0.50, f"Shows how {num} changes across {x_col}."
    window = 3 if len(df) > 30 else 1
    cats = categorical_cols(df)
    transforms = suggest_date_parts(df, x_col) + suggest_outlier_filter(df, num)
    return SuggestionResult(
        chart_name="Line Chart",
        columns=complete_columns("Line Chart", x=x_col, y=num,
                                 series=cats[0] if len(cats) > 0 else None),
        params={**default_params("Line Chart"),
                "rolling_window": window, "mode": "lines+markers"},
        title=f"{num} over {x_col}",
        rationale=rationale,
        score=score,
        transforms=transforms,
    )


def _scatter(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 2:
        return None
    cats = categorical_cols(df)
    transforms = (
        suggest_outlier_filter(df, nums[0])
        + suggest_outlier_filter(df, nums[1])
        + suggest_ratio_derived(df, nums[1], nums[0])
        + suggest_null_filter(df, nums[0])
    )
    return SuggestionResult(
        chart_name="Scatter Plot",
        columns=complete_columns("Scatter Plot",
                                 x=nums[0], y=nums[1],
                                 size=nums[2] if len(nums) > 2 else None,
                                 color=cats[0] if cats else None,
                                 label=None),
        params={**default_params("Scatter Plot"),
                "trendline": "ols" if len(df) >= 5 else "None"},
        title=f"{nums[1]} vs {nums[0]}",
        rationale=f"Explores the relationship between {nums[0]} and {nums[1]}.",
        score=0.58,
        transforms=transforms,
    )


def _heatmap(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    if len(cats) < 2 or not numeric_cols(df):
        return None
    scored = sorted(cats, key=lambda c: cardinality_score(df[c].nunique()), reverse=True)
    x_col, y_col = scored[0], scored[1]
    val_col = best_numeric(df)
    nx, ny = df[x_col].nunique(), df[y_col].nunique()
    transforms = (
        suggest_top_n(df, x_col, val_col, n=15)
        + suggest_top_n(df, y_col, val_col, n=15)
        + suggest_null_filter(df, val_col)
    )
    return SuggestionResult(
        chart_name="Heatmap",
        columns=complete_columns("Heatmap", x=x_col, y=y_col, value=val_col),
        params={**default_params("Heatmap"), "aggregation": "mean", "show_values": True},
        title=f"{val_col} by {x_col} × {y_col}",
        rationale=(f"Cross-tabulates {val_col} across {nx} {x_col} and {ny} {y_col} values "
                   "— quickly spots high/low combinations."),
        score=0.50 + 0.35 * cardinality_score(nx) * cardinality_score(ny),
        transforms=transforms,
    )


def _waterfall(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=8)
    if not num or not cat or not has_mixed_sign(df, num):
        return None
    return SuggestionResult(
        chart_name="Waterfall",
        columns=complete_columns("Waterfall", label=cat, value=num),
        params={**default_params("Waterfall"), "show_total": True},
        title=f"Cumulative {num} breakdown",
        rationale=(f"{num} has both positive and negative values — "
                   f"waterfall shows net cumulative impact across {cat}."),
        score=0.78,
    )


def _distribution(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if not nums:
        return None
    val_col = max(nums, key=lambda c: df[c].nunique())
    n_unique = df[val_col].nunique()
    if n_unique < 5:
        return None
    cats = categorical_cols(df)
    chart_type = "Box Plot" if len(df) < 30 else "Histogram"
    nbins = max(10, min(int(n_unique ** 0.5) * 5, 50))
    return SuggestionResult(
        chart_name="Distribution",
        columns=complete_columns("Distribution",
                                 value=val_col,
                                 group=cats[0] if cats else None),
        params={**default_params("Distribution"),
                "chart_type": chart_type, "nbins": nbins,
                "show_mean": True, "show_median": True},
        title=f"Distribution of {val_col}",
        rationale=f"{val_col} has {n_unique} unique values — {chart_type.lower()} reveals spread and outliers.",
        score=0.55 + 0.25 * min(1.0, n_unique / 50),
    )


def _tornado(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=8)
    if not num or not cat or not has_mixed_sign(df, num):
        return None
    n = df[cat].nunique()
    return SuggestionResult(
        chart_name="Tornado Chart",
        columns=complete_columns("Tornado Chart", label=cat, value=num),
        params={**default_params("Tornado Chart"),
                "top_n": min(n, 10), "show_baseline": True, "show_values": True},
        title=f"Sensitivity: {num} by {cat}",
        rationale=(f"Ranks {n} {cat} factors by signed impact on {num} "
                   "— shows which drive the most positive or negative change."),
        score=0.72,
    )


def _funnel(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cats = categorical_cols(df)
    stage_cols = [c for c in cats if STAGE_HINTS.search(c)]
    if not stage_cols or not num:
        return None
    stage_col = stage_cols[0]
    n = df[stage_col].nunique()
    return SuggestionResult(
        chart_name="Funnel Chart",
        columns=complete_columns("Funnel Chart", stage=stage_col, value=num),
        params={**default_params("Funnel Chart"),
                "aggregation": "sum", "sort_order": "By Value (desc)", "show_pct": True},
        title=f"{num} through {stage_col}",
        rationale=(f"'{stage_col}' suggests a multi-step process — "
                   f"funnel shows drop-off rates across {n} stages."),
        score=0.88,
    )


def _sankey(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    if len(cats) < 2 or not numeric_cols(df):
        return None
    src_cols = [c for c in cats if FLOW_SRC_HINTS.search(c)]
    tgt_cols = [c for c in cats if FLOW_TGT_HINTS.search(c)]
    if src_cols and tgt_cols and src_cols[0] != tgt_cols[0]:
        src, tgt, score = src_cols[0], tgt_cols[0], 0.88
    else:
        ordered = sorted(cats, key=lambda c: df[c].nunique())
        src, tgt, score = ordered[0], ordered[1], 0.38
    num = best_numeric(df)
    return SuggestionResult(
        chart_name="Sankey Diagram",
        columns=complete_columns("Sankey Diagram", source=src, target=tgt, value=num),
        params={**default_params("Sankey Diagram"), "top_n": 30},
        title=f"{num} flow: {src} → {tgt}",
        rationale=f"Shows how {num} flows from {src} to {tgt} — traces where value goes.",
        score=score,
    )


def _area(df: pd.DataFrame) -> SuggestionResult | None:
    dt = datetime_cols(df)
    num = best_numeric(df)
    if not num or not dt:
        return None
    cats = categorical_cols(df)
    window = 3 if len(df) > 30 else 1
    transforms = suggest_date_parts(df, dt[0])
    return SuggestionResult(
        chart_name="Area Chart",
        columns=complete_columns("Area Chart", x=dt[0], y=num,
                                 series=cats[0] if cats else None),
        params={**default_params("Area Chart"),
                "rolling_window": window, "fill_mode": "tozeroy", "groupnorm": "none"},
        title=f"Cumulative {num} over {dt[0]}",
        rationale=f"Area chart shows {num} volume accumulating over time — good for trend + magnitude together.",
        score=0.70,
        transforms=transforms,
    )


def _pie(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=6)
    if not num or not cat:
        return None
    n = df[cat].nunique()
    if n > 12:
        return None  # too many slices — not a pie
    return SuggestionResult(
        chart_name="Pie / Donut",
        columns=complete_columns("Pie / Donut", names=cat, values=num),
        params={**default_params("Pie / Donut"),
                "hole": 0.45, "show_pct": True, "top_n": min(n, 8)},
        title=f"Share of {num} by {cat}",
        rationale=f"{n} {cat} categories — donut shows each category's contribution to the whole.",
        score=0.55 + 0.25 * cardinality_score(n, lo=3, hi=8),
    )


def _treemap(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    nums = numeric_cols(df)
    if len(cats) < 2 or not nums:
        return None
    # Pick two categoricals: broadest and finest grained
    sorted_cats = sorted(cats, key=lambda c: df[c].nunique())
    l1, l2 = sorted_cats[0], sorted_cats[1]
    val_col = best_numeric(df)
    n_leaves = df[l2].nunique()
    return SuggestionResult(
        chart_name="Treemap",
        columns=complete_columns("Treemap", level1=l1, level2=l2, level3=None,
                                 value=val_col, color=None),
        params={**default_params("Treemap"), "show_values": True, "show_pct": True},
        title=f"{val_col} by {l1} → {l2}",
        rationale=(f"Treemap reveals {n_leaves} {l2} items nested inside {df[l1].nunique()} "
                   f"{l1} groups — size = {val_col}."),
        score=0.60 + 0.2 * cardinality_score(n_leaves, lo=4, hi=20),
    )


def _kpi(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if not nums:
        return None
    # Prefer KPI-named columns
    kpi_cols = [c for c in nums if KPI_HINTS.search(c)]
    pool = kpi_cols if kpi_cols else nums
    # Take up to 4
    metrics = pool[:4]
    # Look for a reference/target column
    ref = next((c for c in nums if TARGET_HINTS.search(c) and c not in metrics), None)
    col_map = {
        "value":     metrics[0],
        "value2":    metrics[1] if len(metrics) > 1 else None,
        "value3":    metrics[2] if len(metrics) > 2 else None,
        "value4":    metrics[3] if len(metrics) > 3 else None,
        "reference": ref,
    }
    n_tiles = len(metrics)
    score = 0.75 if kpi_cols else 0.45
    return SuggestionResult(
        chart_name="KPI Tiles",
        columns=complete_columns("KPI Tiles", **col_map),
        params={**default_params("KPI Tiles"), "delta_mode": "relative" if ref else "none"},
        title=f"KPI summary — {', '.join(metrics[:3])}{'…' if n_tiles > 3 else ''}",
        rationale=f"Shows {n_tiles} key metrics at a glance" + (f" with delta vs {ref}." if ref else "."),
        score=score,
    )


def _bullet(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    nums = numeric_cols(df)
    if not cats or len(nums) < 2:
        return None
    # Need at least one actual and one target column
    actual_col  = next((c for c in nums if not TARGET_HINTS.search(c)), None)
    target_col  = next((c for c in nums if TARGET_HINTS.search(c)), None)
    if not actual_col or not target_col:
        return None
    label_col = best_categorical(df, target_n=6)
    n = df[label_col].nunique()
    return SuggestionResult(
        chart_name="Bullet Chart",
        columns=complete_columns("Bullet Chart",
                                 label=label_col, actual=actual_col,
                                 target=target_col, low=None, high=None),
        params={**default_params("Bullet Chart"), "show_values": True},
        title=f"{actual_col} vs {target_col} by {label_col}",
        rationale=(f"Compares actual {actual_col} against target {target_col} "
                   f"across {n} {label_col} categories."),
        score=0.82,
    )


def _boxplot(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=6)
    if not num or not cat:
        return None
    n = df[cat].nunique()
    if n < 2 or n > 20:
        return None
    return SuggestionResult(
        chart_name="Box Plot",
        columns=complete_columns("Box Plot", x=cat, y=num, color=None),
        params={**default_params("Box Plot")},
        title=f"Distribution of {num} by {cat}",
        rationale=(
            f"Compares {num} across {n} {cat} groups — "
            f"shows medians, IQR, and outliers side-by-side."
        ),
        score=0.62 + 0.20 * cardinality_score(n, lo=3, hi=10),
        transforms=suggest_outlier_filter(df, num) + suggest_null_filter(df, num),
    )


def _violin(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=5)
    if not num or not cat:
        return None
    n = df[cat].nunique()
    if n < 2 or n > 15:
        return None
    if (len(df) / n) < 5:
        return None
    return SuggestionResult(
        chart_name="Violin Plot",
        columns=complete_columns("Violin Plot", x=cat, y=num, color=None),
        params={**default_params("Violin Plot"), "show_box": True},
        title=f"Shape of {num} across {cat}",
        rationale=(
            f"Violin reveals full density distribution of {num} within each {cat} group — "
            f"shows multimodality that box plots hide."
        ),
        score=0.58 + 0.22 * cardinality_score(n, lo=2, hi=8),
        transforms=suggest_outlier_filter(df, num),
    )


def _strip_plot(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=5)
    if not num or not cat:
        return None
    n = df[cat].nunique()
    if n < 2 or n > 20:
        return None
    if len(df) > 2000:
        return None
    return SuggestionResult(
        chart_name="Strip Plot",
        columns=complete_columns("Strip Plot", x=cat, y=num, color=None),
        params={**default_params("Strip Plot"), "jitter": 0.3, "show_box": True},
        title=f"All {num} values by {cat}",
        rationale=(
            f"{len(df)} rows across {n} {cat} groups — strip plot shows every data point "
            f"so outliers, clusters, and gaps are immediately visible."
        ),
        score=0.60 + 0.15 * cardinality_score(n, lo=2, hi=8),
    )


def _calendar_heatmap(df: pd.DataFrame) -> SuggestionResult | None:
    dt = datetime_cols(df)
    num = best_numeric(df)
    if not dt or not num:
        return None
    date_col = dt[0]
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
        n_days = parsed.dt.date.nunique()
        if n_days < 7:
            return None
    except Exception:
        return None
    return SuggestionResult(
        chart_name="Calendar Heatmap",
        columns=complete_columns("Calendar Heatmap", date=date_col, value=num),
        params={**default_params("Calendar Heatmap"), "aggregation": "Sum"},
        title=f"{num} calendar",
        rationale=(
            f"Data spans {n_days} unique dates — calendar heatmap reveals daily patterns, "
            f"weekday/weekend cycles, and seasonal spikes in {num}."
        ),
        score=0.72,
        transforms=suggest_date_parts(df, date_col),
    )


def _forecast(df: pd.DataFrame) -> SuggestionResult | None:
    dt = datetime_cols(df)
    num = best_numeric(df)
    if not dt or not num or len(df) < 5:
        return None
    date_col = dt[0]
    return SuggestionResult(
        chart_name="Forecast",
        columns=complete_columns("Forecast", date=date_col, value=num),
        params={**default_params("Forecast")},
        title=f"{num} forecast",
        rationale=(
            f"Projects future {num} values from {len(df)} historical data points — "
            f"visualizes trend with polynomial fit and confidence intervals."
        ),
        score=0.68,
        transforms=suggest_date_parts(df, date_col),
    )


def _network_graph(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    if len(cats) < 2:
        return None
    src_cols = [c for c in cats if FLOW_SRC_HINTS.search(c)]
    tgt_cols = [c for c in cats if FLOW_TGT_HINTS.search(c)]
    if src_cols and tgt_cols and src_cols[0] != tgt_cols[0]:
        src, tgt, score = src_cols[0], tgt_cols[0], 0.85
    else:
        ordered = sorted(cats, key=lambda c: df[c].nunique())
        src, tgt, score = ordered[0], ordered[-1], 0.40
    num = best_numeric(df)
    return SuggestionResult(
        chart_name="Network Graph",
        columns=complete_columns("Network Graph", source=src, target=tgt, weight=num),
        params={**default_params("Network Graph")},
        title=f"Network: {src} → {tgt}",
        rationale=(
            f"Visualizes relationships between {src} and {tgt} as a force-directed graph — "
            f"node size and edge width reflect {num or 'frequency'}."
        ),
        score=score,
    )


def _chord_diagram(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    if len(cats) < 2:
        return None
    src_cols = [c for c in cats if FLOW_SRC_HINTS.search(c)]
    tgt_cols = [c for c in cats if FLOW_TGT_HINTS.search(c)]
    if src_cols and tgt_cols and src_cols[0] != tgt_cols[0]:
        src, tgt, score = src_cols[0], tgt_cols[0], 0.82
    else:
        ordered = sorted(cats, key=lambda c: df[c].nunique())
        src, tgt = ordered[0], ordered[-1]
        if df[src].nunique() > 15 or df[tgt].nunique() > 15:
            return None
        score = 0.42
    num = best_numeric(df)
    return SuggestionResult(
        chart_name="Chord Diagram",
        columns=complete_columns("Chord Diagram", source=src, target=tgt, value=num),
        params={**default_params("Chord Diagram")},
        title=f"Flow: {src} ↔ {tgt}",
        rationale=(
            f"Chord diagram shows bidirectional flow between {src} and {tgt} — "
            f"arc width encodes {num or 'frequency'}, revealing dominant flows."
        ),
        score=score,
    )


def _correlation_matrix_rb(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 3:
        return None
    return SuggestionResult(
        chart_name="Correlation Matrix",
        columns=complete_columns("Correlation Matrix", _a=nums[0], _b=nums[1]),
        params={**default_params("Correlation Matrix")},
        title=f"Correlations across {len(nums)} numeric columns",
        rationale=(
            f"With {len(nums)} numeric columns, a correlation matrix instantly shows "
            f"which pairs move together — essential for feature selection and hypothesis generation."
        ),
        score=0.65 + 0.10 * min(1.0, (len(nums) - 3) / 5),
    )


def _scatter_matrix_rb(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 4:
        return None
    cats = categorical_cols(df)
    return SuggestionResult(
        chart_name="Scatter Matrix",
        columns=complete_columns("Scatter Matrix", _a=nums[0], _b=nums[1],
                                 color=cats[0] if cats else None),
        params={**default_params("Scatter Matrix"), "max_cols": min(len(nums), 6)},
        title=f"SPLOM: all pairwise scatter plots ({len(nums)} columns)",
        rationale=(
            f"{len(nums)} numeric columns → {len(nums)**2} pairwise views — "
            f"SPLOM spots clusters, correlations, and outliers across all pairs at once."
        ),
        score=0.58 + 0.12 * min(1.0, (len(nums) - 4) / 6),
    )


def _density_heatmap_rb(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 2 or len(df) < 100:
        return None
    z_col = nums[2] if len(nums) > 2 else None
    return SuggestionResult(
        chart_name="Density Heatmap",
        columns=complete_columns("Density Heatmap", x=nums[0], y=nums[1], z=z_col),
        params={**default_params("Density Heatmap")},
        title=f"Density: {nums[1]} vs {nums[0]}",
        rationale=(
            f"{len(df):,} rows — scatter would overplot. "
            f"Density heatmap bins data into a 2-D grid revealing where points concentrate."
        ),
        score=0.55 + 0.25 * min(1.0, len(df) / 2000),
    )


def _marginal_scatter_rb(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 2:
        return None
    cats = categorical_cols(df)
    size_col  = nums[2] if len(nums) > 2 else None
    color_col = cats[0]  if cats          else None
    return SuggestionResult(
        chart_name="Marginal Scatter",
        columns=complete_columns("Marginal Scatter",
                                 x=nums[0], y=nums[1],
                                 color=color_col, size=size_col),
        params={**default_params("Marginal Scatter")},
        title=f"{nums[1]} vs {nums[0]} with distributions",
        rationale=(
            f"Scatter of {nums[0]} × {nums[1]} with histogram and box marginals — "
            f"shows bivariate correlation and each variable's distribution in one chart."
        ),
        score=0.62 + 0.15 * min(1.0, len(nums) / 5),
        transforms=suggest_outlier_filter(df, nums[0]) + suggest_outlier_filter(df, nums[1]),
    )


def _lollipop_rb(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=12)
    if not num or not cat:
        return None
    n = df[cat].nunique()
    if n < 3:
        return None
    agg = "sum" if NUMERIC_HINTS.search(num) else "mean"
    return SuggestionResult(
        chart_name="Lollipop Chart",
        columns=complete_columns("Lollipop Chart", x=cat, y=num, color=None),
        params={**default_params("Lollipop Chart"),
                "aggregation": agg,
                "sort_by": "Value (desc)",
                "orientation": "h" if n > 10 else "v",
                "top_n": min(n, 15) if n > 15 else 0},
        title=f"{agg.capitalize()} of {num} by {cat}",
        rationale=(
            f"Lollipop ranks {n} {cat} categories by {agg} of {num} — "
            f"cleaner than a bar chart, easier to compare magnitudes at a glance."
        ),
        score=0.58 + 0.22 * cardinality_score(n, lo=5, hi=20),
        transforms=suggest_outlier_filter(df, num) + suggest_null_filter(df, num),
    )


def _gantt_rb(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    dts  = datetime_cols(df)

    task_col  = next((c for c in cats if GANTT_TASK_HINTS.search(c)), None)
    start_col = next((c for c in dts  if GANTT_START_HINTS.search(c)), None)
    end_col   = next((c for c in dts  if GANTT_END_HINTS.search(c) and c != start_col), None)

    if not task_col or not start_col or not end_col:
        if len(dts) >= 2 and len(cats) >= 1:
            task_col  = cats[0]
            start_col = dts[0]
            end_col   = dts[1]
        else:
            return None

    color_col = next((c for c in cats if c != task_col), None)
    return SuggestionResult(
        chart_name="Gantt Chart",
        columns=complete_columns("Gantt Chart",
                                 task=task_col, start=start_col,
                                 end=end_col, color=color_col),
        params={**default_params("Gantt Chart"), "sort_by": "Start", "show_today": True},
        title=f"Timeline of {task_col}",
        rationale=(
            f"Date columns '{start_col}' and '{end_col}' define task durations — "
            f"a Gantt chart visualises scheduling and progress at a glance."
        ),
        score=0.72,
        transforms=[],
    )


_BUILDERS = [_bar, _line, _scatter, _heatmap, _waterfall,
             _distribution, _tornado, _funnel, _sankey,
             _area, _pie, _treemap, _kpi, _bullet,
             _boxplot, _violin, _strip_plot,
             _calendar_heatmap, _forecast,
             _network_graph, _chord_diagram,
             _correlation_matrix_rb, _scatter_matrix_rb,
             _density_heatmap_rb, _marginal_scatter_rb,
             _lollipop_rb, _gantt_rb]


# ── Suggester class ───────────────────────────────────────────────────────────

class RuleBasedSuggester(BaseSuggester):
    """
    Heuristic suggester — column types, cardinalities, and name patterns.
    Reliable baseline, always produces results with no external calls.
    """

    def suggest(self, df: pd.DataFrame, query: str = "") -> list[SuggestionResult]:
        raw: list[SuggestionResult] = []
        for builder in _BUILDERS:
            try:
                r = builder(df)
                if r is not None:
                    v = validate_and_complete(r, df)
                    if v is not None:
                        raw.append(v)
            except Exception:
                pass

        # Deduplicate: keep best per chart type
        best: dict[str, SuggestionResult] = {}
        for r in raw:
            if r.chart_name not in best or r.score > best[r.chart_name].score:
                best[r.chart_name] = r

        results = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return filter_by_query(results, query)


# ── Self-registration ─────────────────────────────────────────────────────────

from cerp_viz.suggestions.registry import register  # noqa: E402
register("Rule-Based", RuleBasedSuggester)
