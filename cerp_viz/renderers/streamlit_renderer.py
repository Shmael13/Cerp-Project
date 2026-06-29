from typing import Any

import streamlit as st

from cerp_viz.renderers.base import BaseRenderer

_PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


class StreamlitRenderer(BaseRenderer):
    def render(self, figure: Any) -> None:
        st.plotly_chart(figure, use_container_width=True, config=_PLOTLY_CONFIG)
