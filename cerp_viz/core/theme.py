from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Theme:
    name: str
    plotly_template: str
    paper_bgcolor: str
    plot_bgcolor: str
    font_color: str
    font_family: str
    accent_colors: list[str] = field(default_factory=list)
    gridcolor: str = "rgba(128,128,128,0.2)"


THEMES: dict[str, Theme] = {
    "Light": Theme(
        name="Light",
        plotly_template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_color="#111111",
        font_family="'Helvetica Neue', Helvetica, Arial, sans-serif",
        accent_colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"],
        gridcolor="rgba(0,0,0,0.08)",
    ),
    "Dark": Theme(
        name="Dark",
        plotly_template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        font_family="'Helvetica Neue', Helvetica, Arial, sans-serif",
        accent_colors=["#89b4fa","#fab387","#a6e3a1","#f38ba8","#cba6f7","#89dceb"],
        gridcolor="rgba(255,255,255,0.1)",
    ),
    "Minimal": Theme(
        name="Minimal",
        plotly_template="simple_white",
        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",
        font_color="#333333",
        font_family="'Georgia', serif",
        accent_colors=["#555555","#888888","#aaaaaa","#cccccc","#dddddd","#eeeeee"],
        gridcolor="rgba(0,0,0,0.05)",
    ),
    "Corporate Blue": Theme(
        name="Corporate Blue",
        plotly_template="plotly_white",
        paper_bgcolor="#f0f4f8",
        plot_bgcolor="#ffffff",
        font_color="#1a202c",
        font_family="'Segoe UI', Calibri, Arial, sans-serif",
        accent_colors=["#2b6cb0","#3182ce","#63b3ed","#bee3f8","#ebf8ff","#1a365d"],
        gridcolor="rgba(43,108,176,0.1)",
    ),
    "High Contrast": Theme(
        name="High Contrast",
        plotly_template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_color="#000000",
        font_family="'Arial Black', Arial, sans-serif",
        accent_colors=["#000000","#e6194b","#3cb44b","#ffe119","#4363d8","#f58231"],
        gridcolor="rgba(0,0,0,0.15)",
    ),
}


def apply_theme(fig: Any, theme: Theme) -> Any:
    """Apply a Theme to a Plotly figure in-place and return it."""
    fig.update_layout(
        template=theme.plotly_template,
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, family=theme.font_family),
    )
    fig.update_xaxes(gridcolor=theme.gridcolor, zerolinecolor=theme.gridcolor)
    fig.update_yaxes(gridcolor=theme.gridcolor, zerolinecolor=theme.gridcolor)
    return fig
