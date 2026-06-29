import streamlit as st

_EXAMPLES = [
    ("📈 Time-series / Forecasting",
     "Date | Revenue\n2024-01-01 | 42000\n2024-02-01 | 47500\n…",
     "Forecast, Line Chart, Area Chart"),
    ("📊 Category breakdown",
     "Region | Product | Sales\nNorth | Widget A | 1200\nSouth | Widget B | 890\n…",
     "Bar Chart, Treemap, Sunburst, Pie"),
    ("🔗 Relationships / Flows",
     "Source | Target | Amount\nMarketing | Sales | 5000\nSales | Support | 1200\n…",
     "Sankey Diagram, Chord Diagram, Network Graph"),
    ("📦 Distributions",
     "Category | Value\nQ1 | 34\nQ1 | 41\nQ2 | 58\n…",
     "Box Plot, Violin Plot, Distribution"),
]


def render_welcome() -> None:
    st.markdown("## 👋 Welcome to CERP Visualizer")
    st.markdown(
        "Upload an **Excel** (.xlsx / .xls / .xlsm) or **CSV** file using the sidebar "
        "to instantly explore your data with **29 chart types**, AI-powered suggestions, "
        "scenario comparison, What-If simulation, and forecasting."
    )

    st.divider()
    st.markdown("#### What kind of data can I use?")

    cols = st.columns(2)
    for i, (title, sample, charts) in enumerate(_EXAMPLES):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.code(sample, language=None)
                st.caption(f"→ Try: {charts}")

    st.divider()
    st.info(
        "**Tips:**\n"
        "- First row should be column headers\n"
        "- Date columns are auto-detected for time-series charts\n"
        "- Numeric columns unlock Box Plot, Violin, Forecast, and What-If simulation\n"
        "- Use **Multi-file Join** (sidebar) to merge data from multiple sheets or files"
    )
