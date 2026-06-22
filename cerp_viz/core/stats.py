"""
Statistical overlay helpers — pure numpy/scipy, no Streamlit or Plotly imports.
All functions return plain dicts or arrays so chart layers can apply them however they choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── OLS (Ordinary Least Squares) ─────────────────────────────────────────────

def ols_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit y ~ x with OLS.  Returns dict with:
        slope, intercept, r2, p_value, se, n
    Returns None on degenerate input (< 3 valid pairs).
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 3:
            return {}

        x_m, y_m = x.mean(), y.mean()
        ss_xx = ((x - x_m) ** 2).sum()
        if ss_xx == 0:
            return {}

        slope     = ((x - x_m) * (y - y_m)).sum() / ss_xx
        intercept = y_m - slope * x_m
        y_pred    = slope * x + intercept
        ss_res    = ((y - y_pred) ** 2).sum()
        ss_tot    = ((y - y_m) ** 2).sum()
        r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard error and t-test for slope ≠ 0
        se     = np.sqrt(ss_res / max(n - 2, 1) / ss_xx) if ss_xx > 0 else 0.0
        t_stat = slope / se if se > 0 else 0.0
        # Two-tailed p-value approximation via Student-t CDF
        try:
            from scipy.special import betainc
            df_t    = n - 2
            x_t     = df_t / (df_t + t_stat ** 2)
            p_value = betainc(df_t / 2, 0.5, x_t)
        except ImportError:
            p_value = float("nan")

        return dict(slope=slope, intercept=intercept, r2=r2,
                    p_value=p_value, se=se, n=n, y_pred=y_pred, x=x, y=y)
    except Exception:
        return {}


# ── Confidence band ───────────────────────────────────────────────────────────

def ci_band(
    series: pd.Series,
    window: int = 1,
    n_sigma: float = 1.96,
) -> dict[str, pd.Series]:
    """
    Rolling mean ± n_sigma × rolling std.
    Returns {'upper': Series, 'lower': Series, 'mean': Series}.
    """
    s     = pd.to_numeric(series, errors="coerce")
    roll  = s.rolling(window, min_periods=1)
    mean  = roll.mean()
    std   = roll.std().fillna(0)
    return {"upper": mean + n_sigma * std, "lower": mean - n_sigma * std, "mean": mean}


# ── Anomaly detection ─────────────────────────────────────────────────────────

def anomaly_mask(series: pd.Series, n_sigma: float = 2.0) -> pd.Series:
    """Boolean Series — True where |value - mean| > n_sigma × std."""
    s  = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std()
    if sd == 0:
        return pd.Series(False, index=s.index)
    return (s - mu).abs() > n_sigma * sd


# ── Normality ─────────────────────────────────────────────────────────────────

def normality_stats(series: pd.Series) -> dict:
    """
    Returns skewness, kurtosis, and (if scipy available) Shapiro-Wilk p-value.
    Also returns a fitted normal curve as (x_fit, y_fit) arrays.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return {}

    skew = float(s.skew())
    kurt = float(s.kurtosis())

    result: dict = dict(skewness=skew, kurtosis=kurt, n=len(s),
                        mean=float(s.mean()), std=float(s.std()))

    # Fitted normal density for overlay
    mu, sigma = result["mean"], result["std"]
    if sigma > 0:
        x_fit = np.linspace(s.min(), s.max(), 200)
        y_fit = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mu) / sigma) ** 2)
        result["x_fit"] = x_fit
        result["y_fit"] = y_fit

    # Shapiro-Wilk (only reliable for n ≤ 5000)
    try:
        from scipy.stats import shapiro
        n_sample = min(len(s), 5000)
        _, p = shapiro(s.sample(n_sample, random_state=42) if len(s) > n_sample else s)
        result["shapiro_p"] = float(p)
    except ImportError:
        pass

    return result


# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson_annotate(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """Returns r, p_value, n for a Pearson correlation."""
    try:
        pair = df[[col_x, col_y]].dropna()
        pair = pair.apply(pd.to_numeric, errors="coerce").dropna()
        n = len(pair)
        if n < 3:
            return {}
        r = float(pair[col_x].corr(pair[col_y]))
        # t-statistic and p-value
        t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r ** 2, 1e-12))
        try:
            from scipy.special import betainc
            df_t = n - 2
            x_t  = df_t / (df_t + t ** 2)
            p    = float(betainc(df_t / 2, 0.5, x_t))
        except ImportError:
            p = float("nan")
        return dict(r=r, p_value=p, n=n)
    except Exception:
        return {}


# ── Significance stars ────────────────────────────────────────────────────────

def significance_stars(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "(ns)"
