"""
Statistical chart suggester.

Computes real statistics (correlation, OLS R², Pareto concentration,
skewness/kurtosis, pivot CoV, IQR outlier rate) to find the most
analytically interesting chart and settings for each chart type.

Every builder sets ALL column roles. Results deduplicated per chart type.
Self-registers as "Statistical" in the suggester registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cerp_viz.core.suggestions import BaseSuggester, SuggestionResult
from cerp_viz.suggestions._utils import (
    numeric_cols, categorical_cols, datetime_cols,
    best_numeric, best_categorical, has_mixed_sign,
    ols_r2, pearson_r, default_params,
    complete_columns, validate_and_complete,
    STAGE_HINTS, FLOW_SRC_HINTS, FLOW_TGT_HINTS,
    suggest_date_parts, suggest_outlier_filter, filter_by_query,
)


# ── 1. Correlation scan → Scatter Plot ────────────────────────────────────────

def _correlation_scatter(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 2:
        return None

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            r = pearson_r(df, nums[i], nums[j])
            if np.isfinite(r):
                pairs.append((nums[i], nums[j], r))
    if not pairs:
        return None

    x, y, r = max(pairs, key=lambda p: abs(p[2]))
    abs_r = abs(r)
    cats = categorical_cols(df)

    if abs_r >= 0.8:
        trendline = "ols"
        direction = "positively" if r > 0 else "negatively"
        rationale = (f"Strong {direction} correlated pair (r={r:.2f}). "
                     f"OLS trendline quantifies the linear relationship.")
        score = 0.88
    elif abs_r >= 0.5:
        trendline = "lowess"
        rationale = (f"Moderate correlation (r={r:.2f}) between {x} and {y}. "
                     f"LOWESS curve reveals any non-linear pattern.")
        score = 0.72
    else:
        trendline = "None"
        rationale = (f"Weak correlation (r={r:.2f}) between {x} and {y} — "
                     f"the absence of a linear relationship is itself informative.")
        score = 0.52

    # Pick a size column (third numeric if available) to add bubble encoding
    size_col = nums[2] if len(nums) > 2 else None

    return SuggestionResult(
        chart_name="Scatter Plot",
        columns=complete_columns("Scatter Plot",
                                 x=x, y=y,
                                 size=size_col,
                                 color=cats[0] if cats else None,
                                 label=None),
        params={**default_params("Scatter Plot"),
                "trendline": trendline, "opacity": 0.75},
        title=f"{y} vs {x}  (r={r:.2f})",
        rationale=rationale,
        score=score,
        transforms=suggest_outlier_filter(df, x) + suggest_outlier_filter(df, y),
    )


# ── 2. Trend detection → Line Chart ──────────────────────────────────────────

def _trend_line(df: pd.DataFrame) -> SuggestionResult | None:
    dt = datetime_cols(df)
    nums = numeric_cols(df)
    if not dt or not nums:
        return None

    x_col = dt[0]
    best_num, best_r2 = None, -1.0

    for num in nums:
        col = df[num].dropna()
        if len(col) < 4:
            continue
        r2 = ols_r2(np.arange(len(col)), col.values)
        if r2 > best_r2:
            best_r2, best_num = r2, num

    if best_num is None:
        return None

    # Noise ratio → choose smoothing window
    col = df[best_num].dropna().values.astype(float)
    try:
        coeffs = np.polyfit(np.arange(len(col)), col, 1)
        residuals = col - np.polyval(coeffs, np.arange(len(col)))
        noise_ratio = float(residuals.std() / (col.std() + 1e-9))
    except Exception:
        noise_ratio = 0.5

    n = len(df)
    if n > 100 and noise_ratio > 0.5:
        window = min(14, n // 10)
    elif n > 30 and noise_ratio > 0.3:
        window = 5
    else:
        window = 1

    # Pick a secondary series column if available
    cats = categorical_cols(df)
    series_col = cats[0] if cats else None

    if best_r2 >= 0.75:
        rationale = (f"Strong trend in {best_num} (R²={best_r2:.2f}). "
                     f"Rolling window={window} smooths noise to reveal the trajectory.")
        score = 0.94
    elif best_r2 >= 0.40:
        rationale = (f"Moderate trend in {best_num} (R²={best_r2:.2f}). "
                     f"Smoothing (window={window}) highlights directional movement.")
        score = 0.80
    else:
        rationale = (f"No strong trend in {best_num} (R²={best_r2:.2f}) — "
                     f"the line chart reveals volatility and short-term patterns.")
        score = 0.62

    return SuggestionResult(
        chart_name="Line Chart",
        columns=complete_columns("Line Chart", x=x_col, y=best_num, series=series_col),
        params={**default_params("Line Chart"),
                "rolling_window": window, "mode": "lines+markers"},
        title=f"{best_num} trend  (R²={best_r2:.2f})",
        rationale=rationale,
        score=score,
        transforms=suggest_date_parts(df, x_col) + suggest_outlier_filter(df, best_num),
    )


# ── 3. Pareto concentration → Bar Chart ──────────────────────────────────────

def _pareto_bar(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    nums = numeric_cols(df)
    if not cats or not nums:
        return None

    best_cat, best_num, best_conc, n_cats = None, None, 0.0, 0

    for cat in cats:
        n = df[cat].nunique()
        if n < 3 or n > 60:
            continue
        for num in nums:
            try:
                grouped = df.groupby(cat)[num].sum().abs().sort_values(ascending=False)
                total = grouped.sum()
                if total == 0:
                    continue
                top_k = max(1, round(n * 0.2))
                conc = float(grouped.head(top_k).sum() / total)
                if conc > best_conc:
                    best_conc, best_cat, best_num, n_cats = conc, cat, num, n
            except Exception:
                pass

    if best_cat is None or best_conc < 0.50:
        return None

    top_k = max(1, round(n_cats * 0.2))
    top_n_param = min(n_cats, 10) if n_cats > 10 else 0

    # Look for a grouping column other than best_cat
    other_cats = [c for c in cats if c != best_cat]

    return SuggestionResult(
        chart_name="Bar Chart",
        columns=complete_columns("Bar Chart",
                                 x=best_cat, y=best_num,
                                 color=other_cats[0] if other_cats else None),
        params={**default_params("Bar Chart"),
                "aggregation": "sum", "sort_by": "Value (desc)",
                "top_n": top_n_param},
        title=f"Pareto: top {top_k} {best_cat} drive {best_conc*100:.0f}% of {best_num}",
        rationale=(f"Top {top_k} of {n_cats} {best_cat} categories account for "
                   f"{best_conc*100:.0f}% of total {best_num} — classic Pareto pattern."),
        score=min(0.96, 0.65 + 0.60 * (best_conc - 0.50)),
    )


# ── 4. Distribution shape → Histogram / Violin / Box ─────────────────────────

def _distribution_shape(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if not nums:
        return None

    best_num, best_score, best_skew, best_kurt = None, -1.0, 0.0, 0.0

    for num in nums:
        col = df[num].dropna()
        if len(col) < 10:
            continue
        try:
            skew = float(col.skew())
            kurt = float(col.kurtosis())
            interest = abs(skew) * 0.6 + abs(kurt) * 0.15
            if interest > best_score:
                best_score, best_num = interest, num
                best_skew, best_kurt = skew, kurt
        except Exception:
            pass

    if best_num is None or best_score < 0.3:
        return None

    cats = categorical_cols(df)
    n_unique = df[best_num].nunique()
    nbins = max(10, min(int(n_unique ** 0.5) * 5, 60))

    if abs(best_skew) >= 2.0:
        chart_type, score = "Histogram", 0.85
        rationale = (f"{best_num} is highly skewed (skewness={best_skew:.2f}) — "
                     f"histogram reveals the long tail and value concentration.")
    elif abs(best_kurt) >= 3.0:
        chart_type, score = "Violin", 0.80
        rationale = (f"{best_num} has heavy tails (kurtosis={best_kurt:.2f}) — "
                     f"violin plot shows peak density and tail behaviour.")
    else:
        chart_type, score = "Box Plot", 0.65
        rationale = (f"Distribution of {best_num} (skew={best_skew:.2f}, "
                     f"kurt={best_kurt:.2f}) — box plot shows median, IQR, outliers.")

    return SuggestionResult(
        chart_name="Distribution",
        columns=complete_columns("Distribution",
                                 value=best_num,
                                 group=cats[0] if cats else None),
        params={**default_params("Distribution"),
                "chart_type": chart_type, "nbins": nbins,
                "show_mean": True, "show_median": True},
        title=f"{'Skewed' if abs(best_skew) >= 1 else 'Distribution of'} {best_num}",
        rationale=rationale,
        score=score,
        transforms=suggest_outlier_filter(df, best_num),
    )


# ── 5. Cross-group variance → Heatmap ────────────────────────────────────────

def _variance_heatmap(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    nums = numeric_cols(df)
    if len(cats) < 2 or not nums:
        return None

    best_x, best_y, best_val, best_cv = None, None, None, 0.0

    for i, x_col in enumerate(cats):
        nx = df[x_col].nunique()
        if nx < 2 or nx > 25:
            continue
        for y_col in cats[i + 1:]:
            ny = df[y_col].nunique()
            if ny < 2 or ny > 25:
                continue
            for val_col in nums:
                try:
                    pivot = df.pivot_table(index=y_col, columns=x_col,
                                           values=val_col, aggfunc="mean")
                    flat = pivot.values.flatten()
                    flat = flat[np.isfinite(flat)]
                    if len(flat) < 4:
                        continue
                    cv = float(flat.std() / (abs(flat.mean()) + 1e-9))
                    if cv > best_cv:
                        best_cv, best_x, best_y, best_val = cv, x_col, y_col, val_col
                except Exception:
                    pass

    if best_x is None or best_cv < 0.2:
        return None

    return SuggestionResult(
        chart_name="Heatmap",
        columns=complete_columns("Heatmap", x=best_x, y=best_y, value=best_val),
        params={**default_params("Heatmap"),
                "aggregation": "mean", "show_values": True, "color_scale": "RdBu"},
        title=f"{best_val} by {best_x} × {best_y}  (CV={best_cv:.2f})",
        rationale=(f"High coefficient of variation (CV={best_cv:.2f}) in {best_val} "
                   f"across {best_x}×{best_y} — heatmap pinpoints hotspots and cold spots."),
        score=min(0.93, 0.55 + 0.30 * min(best_cv, 1.5)),
    )


# ── 6. IQR outlier density → Box Plot ────────────────────────────────────────

def _outlier_box(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if not nums:
        return None

    best_num, best_rate = None, 0.0
    for num in nums:
        col = df[num].dropna()
        if len(col) < 10:
            continue
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        rate = float(((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).sum() / len(col))
        if rate > best_rate:
            best_rate, best_num = rate, num

    if best_num is None or best_rate < 0.04:
        return None

    cats = categorical_cols(df)
    return SuggestionResult(
        chart_name="Distribution",
        columns=complete_columns("Distribution",
                                 value=best_num,
                                 group=cats[0] if cats else None),
        params={**default_params("Distribution"),
                "chart_type": "Box Plot", "remove_outliers": False},
        title=f"Outliers in {best_num}  ({best_rate*100:.1f}% flagged)",
        rationale=(f"{best_rate*100:.1f}% of {best_num} values are IQR outliers — "
                   f"box plot shows their magnitude and frequency."),
        score=min(0.90, 0.58 + 1.5 * best_rate),
        transforms=suggest_outlier_filter(df, best_num),
    )


# ── 7. Signed contribution sort → Waterfall ──────────────────────────────────

def _signed_waterfall(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=8)
    if not num or not cat or not has_mixed_sign(df, num):
        return None

    try:
        grouped = df.groupby(cat)[num].sum()
        n_pos = int((grouped > 0).sum())
        n_neg = int((grouped < 0).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        balance = 1.0 - abs(n_pos - n_neg) / (n_pos + n_neg)
    except Exception:
        balance = 0.3

    return SuggestionResult(
        chart_name="Waterfall",
        columns=complete_columns("Waterfall", label=cat, value=num),
        params={**default_params("Waterfall"),
                "sort_mode": "By Abs Value (desc)", "show_total": True},
        title=f"Net impact of {cat} on {num}",
        rationale=(f"{n_pos} positive and {n_neg} negative {cat} contributions. "
                   f"Waterfall sorted by magnitude shows what adds vs. subtracts value."),
        score=0.72 + 0.18 * balance,
    )


# ── 8. Funnel efficiency → Funnel Chart ──────────────────────────────────────

def _funnel_efficiency(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    nums = numeric_cols(df)
    stage_cols = [c for c in cats if STAGE_HINTS.search(c)]
    if not stage_cols or not nums:
        return None

    stage_col = stage_cols[0]
    num = best_numeric(df)
    n = df[stage_col].nunique()

    try:
        agg = df.groupby(stage_col)[num].sum().sort_values(ascending=False)
        top, bottom = float(agg.iloc[0]), float(agg.iloc[-1])
        conversion = bottom / top if top > 0 else 0.5
    except Exception:
        conversion = 0.5

    return SuggestionResult(
        chart_name="Funnel Chart",
        columns=complete_columns("Funnel Chart", stage=stage_col, value=num),
        params={**default_params("Funnel Chart"),
                "aggregation": "sum", "sort_order": "By Value (desc)",
                "show_pct": True, "min_stage_pct": 0.0},
        title=f"{num} funnel: {conversion*100:.1f}% end-to-end conversion",
        rationale=(f"Overall conversion across {n} {stage_col} stages: "
                   f"{conversion*100:.1f}%. Funnel shows exactly where volume is lost."),
        score=0.85 + 0.10 * (1.0 - conversion),
    )


# ── 9. Tornado sensitivity → Tornado Chart ───────────────────────────────────

def _sensitivity_tornado(df: pd.DataFrame) -> SuggestionResult | None:
    num = best_numeric(df)
    cat = best_categorical(df, target_n=8)
    if not num or not cat or not has_mixed_sign(df, num):
        return None

    try:
        grouped = df.groupby(cat)[num].sum()
        spread = float(grouped.max() - grouped.min())
        mean_abs = float(grouped.abs().mean())
        cv = spread / (mean_abs + 1e-9)
    except Exception:
        cv = 0.5

    n = df[cat].nunique()
    return SuggestionResult(
        chart_name="Tornado Chart",
        columns=complete_columns("Tornado Chart", label=cat, value=num),
        params={**default_params("Tornado Chart"),
                "top_n": min(n, 10), "show_baseline": True, "show_values": True,
                "direction_filter": "All"},
        title=f"Sensitivity analysis: {num} by {cat}  (spread CV={cv:.2f})",
        rationale=(f"Range of {num} across {n} {cat} values spans CV={cv:.2f}. "
                   f"Tornado ranks factors by absolute impact — shows key drivers."),
        score=min(0.88, 0.60 + 0.20 * min(cv, 1.4)),
    )


# ── N+1. Trend detection → Forecast ──────────────────────────────────────────

def _forecast_trend(df: pd.DataFrame) -> SuggestionResult | None:
    dt = datetime_cols(df)
    nums = numeric_cols(df)
    if not dt or not nums:
        return None

    date_col = dt[0]
    best_num, best_r2 = None, -1.0
    for num in nums:
        col = df[num].dropna()
        if len(col) < 5:
            continue
        r2 = ols_r2(np.arange(len(col)), col.values)
        if r2 > best_r2:
            best_r2, best_num = r2, num

    if best_num is None or best_r2 < 0.15:
        return None

    score = min(0.92, 0.55 + 0.45 * best_r2)
    return SuggestionResult(
        chart_name="Forecast",
        columns=complete_columns("Forecast", date=date_col, value=best_num),
        params={**default_params("Forecast")},
        title=f"{best_num} forecast  (trend R²={best_r2:.2f})",
        rationale=(
            f"Detectable trend in {best_num} (R²={best_r2:.2f}) across {len(df)} periods. "
            f"Forecast projects values forward with polynomial fit and confidence intervals."
        ),
        score=score,
        transforms=suggest_date_parts(df, date_col),
    )


# ── N+2. Inter-group variance → Box Plot ──────────────────────────────────────

def _box_by_category(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    nums = numeric_cols(df)
    if not cats or not nums:
        return None

    best_cat, best_num, best_cv = None, None, 0.0
    for cat in cats:
        n = df[cat].nunique()
        if n < 2 or n > 20:
            continue
        for num in nums:
            try:
                means = df.groupby(cat)[num].mean().dropna()
                if len(means) < 2:
                    continue
                cv = float(means.std() / (abs(means.mean()) + 1e-9))
                if cv > best_cv:
                    best_cv, best_cat, best_num = cv, cat, num
            except Exception:
                pass

    if best_cat is None or best_cv < 0.2:
        return None

    n = df[best_cat].nunique()
    return SuggestionResult(
        chart_name="Box Plot",
        columns=complete_columns("Box Plot", x=best_cat, y=best_num, color=None),
        params={**default_params("Box Plot")},
        title=f"{best_num} spread by {best_cat}  (group CV={best_cv:.2f})",
        rationale=(
            f"High variance across {n} {best_cat} groups (CV={best_cv:.2f}). "
            f"Box plot shows medians, IQR, and outliers per group — reveals which groups are most spread."
        ),
        score=min(0.88, 0.55 + 0.40 * min(best_cv, 0.8)),
    )


# ── N+3. High skewness with categories → Violin ───────────────────────────────

def _violin_skew(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if not nums:
        return None

    best_num, best_skew = None, 0.0
    for num in nums:
        col = df[num].dropna()
        if len(col) < 10:
            continue
        try:
            skew = abs(float(col.skew()))
            if skew > best_skew:
                best_skew, best_num = skew, num
        except Exception:
            pass

    if best_num is None or best_skew < 1.0:
        return None

    cats = categorical_cols(df)
    if not cats:
        return None
    cat = min(cats, key=lambda c: abs(df[c].nunique() - 5))
    n = df[cat].nunique()
    if n < 2 or n > 15 or (len(df) / n) < 5:
        return None

    return SuggestionResult(
        chart_name="Violin Plot",
        columns=complete_columns("Violin Plot", x=cat, y=best_num, color=None),
        params={**default_params("Violin Plot"), "show_box": True, "show_points": "outliers"},
        title=f"Shape of {best_num} by {cat}  (skew={best_skew:.1f})",
        rationale=(
            f"{best_num} is strongly skewed (|skew|={best_skew:.1f}) — violin reveals full "
            f"density shape per {cat} group, exposing asymmetry and heavy tails that box plots compress."
        ),
        score=min(0.87, 0.58 + 0.18 * min(best_skew, 2.5)),
        transforms=suggest_outlier_filter(df, best_num),
    )


# ── N+4. Moderate rows per group → Strip Plot ────────────────────────────────

def _strip_dense(df: pd.DataFrame) -> SuggestionResult | None:
    n_rows = len(df)
    if n_rows < 10 or n_rows > 1000:
        return None

    cats = categorical_cols(df)
    nums = numeric_cols(df)
    if not cats or not nums:
        return None

    cat = min(cats, key=lambda c: abs(df[c].nunique() - 5))
    num = best_numeric(df)
    n = df[cat].nunique()
    if n < 2 or n > 20:
        return None

    rpc = n_rows / n
    if rpc < 5 or rpc > 150:
        return None

    return SuggestionResult(
        chart_name="Strip Plot",
        columns=complete_columns("Strip Plot", x=cat, y=num, color=None),
        params={**default_params("Strip Plot"), "jitter": 0.3, "show_box": True},
        title=f"All {num} points by {cat}",
        rationale=(
            f"~{int(rpc)} rows per {cat} group — strip plot shows every individual "
            f"{num} value with jitter, making outliers and clusters visible without aggregation."
        ),
        score=min(0.80, 0.52 + 0.35 * min(rpc / 50, 1.0)),
    )


# ── N+5. Many mutually correlated columns → Correlation Matrix ───────────────

def _correlation_matrix_multi(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 3:
        return None

    pairs: list[float] = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            r = pearson_r(df, nums[i], nums[j])
            if np.isfinite(r):
                pairs.append(abs(r))

    if not pairs:
        return None

    avg_corr = float(np.mean(pairs))
    method = "pearson" if avg_corr >= 0.4 else "spearman"

    return SuggestionResult(
        chart_name="Correlation Matrix",
        columns=complete_columns("Correlation Matrix", _a=nums[0], _b=nums[1]),
        params={**default_params("Correlation Matrix"), "method": method},
        title=f"Correlation matrix — {len(nums)} variables  (avg |r|={avg_corr:.2f})",
        rationale=(
            f"{len(nums)} numeric columns with average |r|={avg_corr:.2f}. "
            f"Correlation matrix reveals collinear pairs, independent features, and hidden structure."
        ),
        score=min(0.90, 0.52 + 0.55 * avg_corr + 0.08 * min(len(nums) - 3, 5)),
    )


# ── N+6. 4+ numeric columns → Scatter Matrix (SPLOM) ─────────────────────────

def _scatter_matrix_splom(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 4:
        return None

    n_rows = len(df)
    cats = categorical_cols(df)
    sample_n = min(1000, n_rows) if n_rows > 500 else 0

    return SuggestionResult(
        chart_name="Scatter Matrix",
        columns=complete_columns("Scatter Matrix", _a=nums[0], _b=nums[1],
                                 color=cats[0] if cats else None),
        params={**default_params("Scatter Matrix"),
                "max_cols": min(len(nums), 6), "sample_n": sample_n},
        title=f"SPLOM: {len(nums)} × {len(nums)} scatter matrix",
        rationale=(
            f"{len(nums)} numeric columns → {len(nums)**2} pairwise panels. "
            f"SPLOM simultaneously identifies correlation pairs, clusters, and outlier dimensions."
        ),
        score=min(0.82, 0.60 + 0.06 * min(len(nums) - 4, 5)),
    )


# ── N+7. Recurring categorical pairs → Network Graph ─────────────────────────

def _network_pairs(df: pd.DataFrame) -> SuggestionResult | None:
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
        try:
            unique_pairs = df[[src, tgt]].drop_duplicates()
            pair_ratio = 1.0 - len(unique_pairs) / len(df)
        except Exception:
            pair_ratio = 0.0
        if pair_ratio < 0.3 or df[src].nunique() < 3:
            return None
        score = 0.45 + 0.25 * pair_ratio

    num = best_numeric(df)
    n_nodes = df[src].nunique() + df[tgt].nunique()
    return SuggestionResult(
        chart_name="Network Graph",
        columns=complete_columns("Network Graph", source=src, target=tgt, weight=num),
        params={**default_params("Network Graph")},
        title=f"Network: {src} ↔ {tgt}  ({n_nodes} nodes)",
        rationale=(
            f"Recurring {src}→{tgt} pairs form a graph of {n_nodes} nodes. "
            f"Spring layout reveals central hubs, clusters, and isolated pairs."
        ),
        score=min(0.88, score),
    )


# ── N+8. Bidirectional flow pattern → Chord Diagram ─────────────────────────

def _chord_flow(df: pd.DataFrame) -> SuggestionResult | None:
    cats = categorical_cols(df)
    if len(cats) < 2:
        return None

    src_cols = [c for c in cats if FLOW_SRC_HINTS.search(c)]
    tgt_cols = [c for c in cats if FLOW_TGT_HINTS.search(c)]
    if src_cols and tgt_cols and src_cols[0] != tgt_cols[0]:
        src, tgt, score = src_cols[0], tgt_cols[0], 0.85
    else:
        ordered = sorted(cats, key=lambda c: df[c].nunique())
        src, tgt = ordered[0], ordered[-1]
        n_s, n_t = df[src].nunique(), df[tgt].nunique()
        if n_s > 15 or n_t > 15 or n_s < 2:
            return None
        score = 0.42

    src_vals = set(df[src].dropna().unique())
    tgt_vals = set(df[tgt].dropna().unique())
    overlap = src_vals & tgt_vals
    if overlap:
        score = min(score + 0.15, 0.92)

    num = best_numeric(df)
    return SuggestionResult(
        chart_name="Chord Diagram",
        columns=complete_columns("Chord Diagram", source=src, target=tgt, value=num),
        params={**default_params("Chord Diagram")},
        title=f"Chord: {src} ↔ {tgt}" + (" (circular)" if overlap else ""),
        rationale=(
            f"{'Bidirectional' if overlap else 'Directional'} flow between {src} and {tgt} "
            f"shown as circular chord arcs — ribbon width ∝ {num or 'count'}, "
            + (f"{len(overlap)} nodes appear on both sides revealing circular flows." if overlap
               else "instantly highlights dominant flows.")
        ),
        score=score,
    )


# ── N+9. Two numeric cols, large dataset → Density Heatmap ──────────────────

def _density_large(df: pd.DataFrame) -> SuggestionResult | None:
    nums = numeric_cols(df)
    if len(nums) < 2 or len(df) < 200:
        return None

    best_x, best_y, best_r = None, None, 0.0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            r = abs(pearson_r(df, nums[i], nums[j]))
            if r > best_r:
                best_r, best_x, best_y = r, nums[i], nums[j]

    if best_x is None:
        return None

    n_rows = len(df)
    z = nums[2] if len(nums) > 2 else None
    log = n_rows > 5000
    score = min(0.82, 0.50 + 0.20 * min(n_rows / 2000, 1.0) + 0.12 * best_r)
    return SuggestionResult(
        chart_name="Density Heatmap",
        columns=complete_columns("Density Heatmap", x=best_x, y=best_y, z=z),
        params={**default_params("Density Heatmap"),
                "nbins_x": 30, "nbins_y": 30, "log_scale": log},
        title=f"Density: {best_y} vs {best_x}  (n={n_rows:,})",
        rationale=(
            f"{n_rows:,} data points — scatter would overplot. "
            f"Density heatmap (|r|={best_r:.2f}) reveals where the mass of data actually lies."
        ),
        score=score,
    )


_ANALYSES = [
    _correlation_scatter,
    _trend_line,
    _pareto_bar,
    _distribution_shape,
    _outlier_box,
    _variance_heatmap,
    _signed_waterfall,
    _funnel_efficiency,
    _sensitivity_tornado,
    # New chart type analyses
    _forecast_trend,
    _box_by_category,
    _violin_skew,
    _strip_dense,
    _correlation_matrix_multi,
    _scatter_matrix_splom,
    _network_pairs,
    _chord_flow,
    _density_large,
]


# ── Suggester class ───────────────────────────────────────────────────────────

class StatisticalSuggester(BaseSuggester):
    """
    Runs statistical analyses (correlation, OLS trend, Pareto, skewness,
    pivot CoV, outlier rate) to find the best chart + settings for each type.
    Uses only numpy + pandas — no external API calls.
    """

    def suggest(self, df: pd.DataFrame, query: str = "") -> list[SuggestionResult]:
        raw: list[SuggestionResult] = []
        for analysis in _ANALYSES:
            try:
                r = analysis(df)
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
register("Statistical", StatisticalSuggester)
