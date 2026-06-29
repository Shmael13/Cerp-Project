from typing import Any, ClassVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cerp_viz.core.base import BaseVisualization
from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec
from cerp_viz.core.registry import registry

_SCALES = ["RdBu", "RdBu_r", "Viridis", "Plasma", "Spectral", "PiYG", "BrBG"]
_METHODS = ["pearson", "spearman", "kendall"]


def _correlation_matrix(df: pd.DataFrame, method: str, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr(method=method)


def _significance_mask(df: pd.DataFrame, method: str, cols: list[str], threshold: float) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    _FN = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}
    fn = _FN[method]
    n = len(cols)
    mask = pd.DataFrame(True, index=cols, columns=cols)
    data = df[cols].dropna()
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    _, p = fn(data[cols[i]], data[cols[j]])
                    mask.iloc[i, j] = p <= threshold
                except Exception:
                    pass
    return mask


class CorrelationMatrix(BaseVisualization):
    name: ClassVar[str] = "Correlation Matrix"
    description: ClassVar[str] = (
        "Heatmap of pairwise correlations across all numeric columns. "
        "Auto-discovers numeric columns — sidebar pickers just gate compatibility."
    )

    def required_columns(self) -> list[ColumnSpec]:
        return [
            ColumnSpec("_a", "numeric", "Numeric col (chart uses all numeric cols)", required=True),
            ColumnSpec("_b", "numeric", "Numeric col (chart uses all numeric cols)", required=True),
        ]

    def assumptions(self) -> list[AssumptionSpec]:
        return [
            AssumptionSpec("method",      "selectbox", "Correlation method", "pearson",
                           {"choices": _METHODS},                             category="Data"),
            AssumptionSpec("min_periods", "slider",    "Min non-null pairs",  10,
                           {"min": 2, "max": 200, "step": 1},                 category="Data"),
            AssumptionSpec("sig_filter",  "toggle",    "Hide non-significant (p>0.05)", False,
                           {},                                                 category="Data"),
            AssumptionSpec("color_scale", "selectbox", "Color scale",        "RdBu",
                           {"choices": _SCALES},                              category="Display"),
            AssumptionSpec("show_values", "toggle",    "Show values in cells", True,
                           {},                                                 category="Display"),
            AssumptionSpec("show_upper",  "toggle",    "Show upper triangle only", False,
                           {},                                                 category="Display"),
        ]

    def build(self, df: pd.DataFrame, columns: dict[str, str | None], params: dict[str, Any]) -> BuildResult:
        warnings: list[str] = []
        numeric_cols = list(df.select_dtypes("number").columns)

        if len(numeric_cols) < 2:
            return BuildResult(figure=go.Figure(), warnings=["Need at least 2 numeric columns."])

        min_p = int(params["min_periods"])
        work  = df[numeric_cols].dropna(thresh=min_p)
        if len(work) < min_p:
            warnings.append(f"Only {len(work)} rows have enough non-null values (min={min_p}).")

        method = params["method"]
        corr   = _correlation_matrix(work, method, numeric_cols)

        if params["show_upper"]:
            import numpy as np
            mask = ~np.triu(np.ones(corr.shape, dtype=bool), k=1)
            corr = corr.where(mask)

        if params["sig_filter"]:
            try:
                sig = _significance_mask(work, method, numeric_cols, 0.05)
                corr = corr.where(sig)
                warnings.append("Cells hidden where p > 0.05.")
            except ImportError:
                warnings.append("scipy not available — significance filter skipped.")

        text_fmt = ".2f" if params["show_values"] else False
        fig = px.imshow(
            corr,
            color_continuous_scale=params["color_scale"],
            zmin=-1, zmax=1,
            text_auto=text_fmt,
            aspect="auto",
            template="plotly_white",
            title=f"{method.capitalize()} correlation — {len(numeric_cols)} columns",
        )
        fig.update_coloraxes(colorbar_title="r")
        warnings.append(f"{method.capitalize()} correlations across {len(numeric_cols)} numeric columns.")
        return BuildResult(figure=fig, warnings=warnings)


registry.register(CorrelationMatrix)
