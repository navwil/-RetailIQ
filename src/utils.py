"""
RetailIQ - UI Design System & Commercial SaaS Utilities
Provides enterprise-grade SaaS CSS styling, Plotly visualization themes,
metric formatting, and cached state management.
"""

from typing import Any, Dict, Optional, Tuple, Union
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    THEME_PRIMARY,
    THEME_ACCENT,
    THEME_SUCCESS,
    THEME_WARNING,
    THEME_DANGER
)
from src.preprocessing import load_or_process_data
from src.customer_segmentation import compute_rfm_table, train_kmeans_segmentation, load_segmentation_artifacts


def inject_custom_css():
    """
    Injects production-grade enterprise CSS for a sleek, modern commercial SaaS appearance.
    """
    st.markdown("""
    <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
            color: #F1F5F9;
        }

        /* Hide Streamlit Header / Deploy / Footer clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}

        /* App Background */
        .stApp {
            background: linear-gradient(180deg, #0B0F19 0%, #0F172A 100%);
        }

        /* Metric KPI Cards */
        .metric-card {
            position: relative;
            background: #111827;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366F1 0%, #06B6D4 100%);
            opacity: 0.8;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(99, 102, 241, 0.5);
            box-shadow: 0 12px 28px -4px rgba(99, 102, 241, 0.2);
        }
        .metric-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .metric-value {
            font-size: 1.95rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }
        .metric-delta {
            font-size: 0.82rem;
            margin-top: 8px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .delta-positive { color: #10B981; }
        .delta-negative { color: #F43F5E; }
        .delta-neutral { color: #94A3B8; }

        /* Page Headers */
        .page-header {
            background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.4) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid #6366F1;
            padding: 20px 24px;
            border-radius: 12px;
            margin-bottom: 24px;
        }
        .page-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: #F8FAFC;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .page-desc {
            font-size: 0.9rem;
            color: #94A3B8;
            margin: 6px 0 0 0;
            line-height: 1.4;
        }

        /* Status & Feature Badges */
        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .badge-live {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-indigo {
            background: rgba(99, 102, 241, 0.15);
            color: #818CF8;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        /* Clean Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0B0F19;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        /* Dataframe / Table Polish */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            overflow: hidden;
        }

        /* Tabs Styling */
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.88rem;
            color: #94A3B8;
            padding: 10px 18px;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #6366F1 !important;
            border-bottom-color: #6366F1 !important;
        }

        /* Button Styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.88rem;
            padding: 8px 18px;
            transition: all 0.2s ease;
        }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "positive"
):
    """
    Renders a modern commercial KPI card component.
    """
    delta_class = "delta-positive" if delta_type == "positive" else ("delta-negative" if delta_type == "negative" else "delta-neutral")
    arrow = "▲" if delta_type == "positive" else ("▼" if delta_type == "negative" else "•")
    delta_html = f'<div class="metric-delta {delta_class}"><span>{arrow}</span> {delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_page_header(title: str, description: str, badge_text: str = "Live Production"):
    """
    Renders an executive styled page title header with category badge.
    """
    st.markdown(f"""
    <div class="page-header">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <h1 class="page-title">{title}</h1>
            <span class="badge-pill badge-live">● {badge_text}</span>
        </div>
        <p class="page-desc">{description}</p>
    </div>
    """, unsafe_allow_html=True)


def format_currency(val: float) -> str:
    """Formats numeric monetary values cleanly with currency symbol."""
    if abs(val) >= 1_000_000:
        return f"£{val / 1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"£{val / 1_000:.1f}k"
    else:
        return f"£{val:,.2f}"


def format_number(val: Union[int, float]) -> str:
    """Formats large integer or float numbers with separators or metric units."""
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"{val / 1_000:.1f}k"
    else:
        return f"{val:,}"


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """
    Applies high-contrast dark theme styling across Plotly figures.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.4)",
        plot_bgcolor="rgba(17, 24, 39, 0.0)",
        font=dict(family="Inter, sans-serif", color="#F1F5F9", size=12),
        margin=dict(l=35, r=25, t=40, b=35),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.12)"
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.12)"
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1
        )
    )
    return fig


@st.cache_data(show_spinner="Loading and caching retail dataset...")
def get_cached_data(source_preference: str = "csv") -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Loads and caches transaction views across all Streamlit pages.
    """
    return load_or_process_data(source_preference=source_preference)


def get_active_dataset() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Returns the active transaction views and audit metrics.
    If customer uploaded a custom dataset (CSV/Excel/PDF), returns that seamlessly;
    otherwise returns the default dataset.
    """
    if "custom_views" in st.session_state and "custom_metrics" in st.session_state:
        return st.session_state["custom_views"], st.session_state["custom_metrics"]
    return get_cached_data()


@st.cache_data(show_spinner="Computing customer segmentation model...")
def get_cached_segmentation(df_clean_sales: pd.DataFrame, n_clusters: int = 4):
    """
    Caches RFM table, clustering model, and assigned segments.
    """
    rfm_table = compute_rfm_table(df_clean_sales)
    rfm_segmented, km, scaler, meta = train_kmeans_segmentation(
        rfm_table, n_clusters=n_clusters, save_artifacts=True
    )
    return rfm_segmented, km, scaler, meta
