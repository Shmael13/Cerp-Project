from __future__ import annotations
import streamlit as st


def inject_styles() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base typography ─────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #0F172A !important;
}

p, li, label, .stMarkdown {
    color: #334155;
    line-height: 1.6;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px !important;
    border-bottom: 1px solid #E2E8F0 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease !important;
}

.stTabs [aria-selected="true"] {
    color: #1E40AF !important;
    border-bottom: 2px solid #2563EB !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #1E40AF !important;
    background: #F1F5F9 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 6px !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
    border: 1px solid #E2E8F0 !important;
    color: #334155 !important;
    background: #FFFFFF !important;
}

.stButton > button:hover {
    background: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

.stButton > button[kind="primary"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border-color: #2563EB !important;
}

.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* ── Expanders ───────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #334155 !important;
    background: #F8FAFC !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
}

.streamlit-expanderContent {
    border: 1px solid #E2E8F0 !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}

/* ── Input widgets ───────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-color: #E2E8F0 !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* ── Containers with border ──────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    overflow: hidden;
}

/* ── Info / Warning / Error boxes ───────────────────────────────────── */
.stAlert {
    border-radius: 6px !important;
    border-left-width: 4px !important;
    font-size: 0.875rem !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 16px 0 !important;
}

/* ── Download buttons ────────────────────────────────────────────────── */
.stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}

/* ── Caption text ────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 0.78rem !important;
    color: #94A3B8 !important;
}

/* ── Subheader style ─────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h2 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #0F172A !important;
    margin-bottom: 4px !important;
}

/* ── Page-level background ───────────────────────────────────────────── */
.stApp > header {
    background: transparent !important;
}

.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}
</style>
""", unsafe_allow_html=True)
