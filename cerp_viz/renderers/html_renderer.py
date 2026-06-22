from typing import Any

from cerp_viz.renderers.base import BaseRenderer


class HTMLRenderer(BaseRenderer):
    def __init__(self, output_path: str = "output.html") -> None:
        self.output_path = output_path

    def render(self, figure: Any) -> None:
        figure.write_html(self.output_path)
        print(f"Chart saved to {self.output_path}")
