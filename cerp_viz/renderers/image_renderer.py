"""
Image export renderer — requires the `kaleido` package.
Degrades gracefully: raises ImportError with a helpful message if kaleido is absent.
"""
from __future__ import annotations
from typing import Any

from cerp_viz.renderers.base import BaseRenderer


class ImageRenderer(BaseRenderer):
    """Export a Plotly figure to PNG, SVG, or PDF via kaleido."""

    def __init__(
        self,
        output_path: str = "chart.png",
        fmt: str = "png",
        width: int = 1280,
        height: int = 720,
        scale: float = 2.0,
    ) -> None:
        self.output_path = output_path
        self.fmt    = fmt
        self.width  = width
        self.height = height
        self.scale  = scale

    def render(self, figure: Any) -> None:
        try:
            import kaleido  # noqa: F401
        except ImportError:
            raise ImportError(
                "PNG/SVG/PDF export requires kaleido. "
                "Install it with:  pip install kaleido"
            )
        figure.write_image(
            self.output_path,
            format=self.fmt,
            width=self.width,
            height=self.height,
            scale=self.scale,
        )

    def to_bytes(self, figure: Any) -> bytes:
        """Return image bytes suitable for st.download_button."""
        try:
            import kaleido  # noqa: F401
        except ImportError:
            raise ImportError(
                "PNG/SVG/PDF export requires kaleido. "
                "Install it with:  pip install kaleido"
            )
        return figure.to_image(
            format=self.fmt,
            width=self.width,
            height=self.height,
            scale=self.scale,
        )
