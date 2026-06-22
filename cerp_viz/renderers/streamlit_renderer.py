from typing import Any

import streamlit as st

from cerp_viz.renderers.base import BaseRenderer


class StreamlitRenderer(BaseRenderer):
    def render(self, figure: Any) -> None:
        st.plotly_chart(figure, use_container_width=True)
