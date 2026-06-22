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
    complete_columns, validate_and_complete, STAGE_HINTS,
    suggest_date_parts, suggest_outlier_filter,
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


_ANALYSES = [
    _correlation_scatter,
    _trend_line,
    _pareto_bar,
    _distribution_shape,
    _outlier_box,          # also Distribution — will be deduplicated
    _variance_heatmap,
    _signed_waterfall,
    _funnel_efficiency,
    _sensitivity_tornado,
]


# ── Suggester class ───────────────────────────────────────────────────────────

class StatisticalSuggester(BaseSuggester):
    """
    Runs statistical analyses (correlation, OLS trend, Pareto, skewness,
    pivot CoV, outlier rate) to find the best chart + settings for each type.
    Uses only numpy + pandas — no external API calls.
    """

    def suggest(self, df: pd.DataFrame) -> list[SuggestionResult]:
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

        return sorted(best.values(), key=lambda r: r.score, reverse=True)


# ── Self-registration ─────────────────────────────────────────────────────────

from cerp_viz.suggestions.registry import register  # noqa: E402
register("Statistical", StatisticalSuggester)
