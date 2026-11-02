"""
RetailIQ – Online Retail Analytics & Intelligence System
Enterprise Business Intelligence & Machine Learning Platform
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    THEME_PRIMARY,
    THEME_ACCENT,
    THEME_SUCCESS
)
from src.utils import (
    inject_custom_css,
    render_metric_card,
    render_page_header,
    format_currency,
    format_number,
    apply_plotly_theme,
    get_active_dataset,
    get_cached_segmentation
)
from src.data_loader import parse_uploaded_file
from src.preprocessing import clean_retail_data
from src.eda import compute_kpis, get_daily_sales_series, get_top_products

# Page Configuration
st.set_page_config(
    page_title="RetailIQ | Enterprise Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Clean SaaS Styling
inject_custom_css()

# Minimalist, Clean Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.5rem;">🛍️</span>
            <div>
                <h3 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em;">RetailIQ</h3>
                <span style="color: #6366F1; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">Enterprise Edition</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Dataset Status Card
    active_source = st.session_state.get("custom_source", "OnlineRetail.csv (Production)")
    is_custom = "custom_views" in st.session_state
    badge_bg = "rgba(16, 185, 129, 0.15)" if not is_custom else "rgba(99, 102, 241, 0.2)"
    badge_color = "#34D399" if not is_custom else "#818CF8"

    st.markdown(f"""
    <div style="background: #111827; padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); margin-top: 14px;">
        <div style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Active Data Source</div>
        <div style="font-size: 0.85rem; color: {badge_color}; font-weight: 600; margin-top: 4px; word-break: break-all;">📁 {active_source}</div>
    </div>
    """, unsafe_allow_html=True)

    if is_custom:
        if st.button("🔄 Switch to Default Data", use_container_width=True):
            del st.session_state["custom_views"]
            del st.session_state["custom_metrics"]
            del st.session_state["custom_source"]
            st.rerun()

# ----------------- TOP INGESTION HUB -----------------
with st.expander("📤 Ingest New Sales Dataset (CSV, Excel, or PDF)", expanded=False):
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Drop customer orders file here",
            type=["csv", "xlsx", "xls", "pdf"],
            label_visibility="collapsed"
        )
    with col_up2:
        st.markdown("""
        <div style="font-size: 0.82rem; color: #94A3B8; padding-top: 4px;">
            <b>Supported Formats:</b><br/>
            • <code>.csv</code> (UTF-8, ISO)<br/>
            • <code>.xlsx / .xls</code><br/>
            • <code>.pdf</code> (Tables)
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                raw_df, summary_str = parse_uploaded_file(uploaded_file)
                views, audit_metrics = clean_retail_data(raw_df, save_processed=False)
                st.session_state["custom_views"] = views
                st.session_state["custom_metrics"] = audit_metrics
                st.session_state["custom_source"] = summary_str
                st.success(f"✅ Ingested: **{summary_str}**")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Ingestion Error: {e}")

# Fetch active dataset (Custom upload or Default)
views, audit_metrics = get_active_dataset()
df_sales = views["clean_sales"]
df_returns = views["returns"]
kpis = compute_kpis(df_sales, df_returns)

# Header
render_page_header(
    "Enterprise Commercial Intelligence",
    "Real-time operational demand, customer cohort segmentation, and predictive revenue intelligence."
)

# Executive KPI Grid
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Realized Sales Revenue", format_currency(kpis["total_revenue"]), "+100% Validated", "positive")
with col2:
    render_metric_card("Completed Invoices", format_number(kpis["total_orders"]), "Unique Orders", "neutral")
with col3:
    render_metric_card("Active Customer Accounts", format_number(kpis["unique_customers"]), "Eligible for RFM", "positive")
with col4:
    render_metric_card("Average Order Value (AOV)", format_currency(kpis["avg_order_value"]), "Per Checkout", "positive")

st.markdown("<br/>", unsafe_allow_html=True)

# Main Commercial Showcase (Two Columns)
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📈 Revenue Trajectory & Moving Averages")
    daily_df = get_daily_sales_series(df_sales)
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=daily_df["Date"], y=daily_df["Revenue"],
        name="Daily Revenue", mode="lines",
        line=dict(color="rgba(99, 102, 241, 0.35)", width=1.2)
    ))
    fig_daily.add_trace(go.Scatter(
        x=daily_df["Date"], y=daily_df["7D_MA"],
        name="7-Day Moving Avg", mode="lines",
        line=dict(color="#06B6D4", width=2.2)
    ))
    fig_daily.add_trace(go.Scatter(
        x=daily_df["Date"], y=daily_df["30D_MA"],
        name="30-Day Moving Avg", mode="lines",
        line=dict(color="#10B981", width=2.5, dash="dot")
    ))
    fig_daily.update_layout(xaxis_title="Date", yaxis_title="Revenue (£)", height=370, hovermode="x unified")
    st.plotly_chart(apply_plotly_theme(fig_daily), use_container_width=True)

with col_right:
    st.markdown("### 🎯 Customer Cohort Revenue Share")
    rfm_df, _, _, meta = get_cached_segmentation(df_sales, n_clusters=4)
    seg_summary = pd.DataFrame(meta["segment_summary"])
    fig_pie = px.pie(
        seg_summary, names="Segment", values="TotalRevenue",
        hole=0.55, color_discrete_sequence=["#6366F1", "#06B6D4", "#10B981", "#F59E0B"]
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=370, showlegend=False)
    st.plotly_chart(apply_plotly_theme(fig_pie), use_container_width=True)

# Quick Nav Cards
st.markdown("### ⚡ Operational Modules")
m_c1, m_c2, m_c3, m_c4 = st.columns(4)

with m_c1:
    st.markdown("""
    <div class="metric-card" style="padding: 16px;">
        <div style="font-size: 1.4rem;">📊</div>
        <div style="font-weight: 700; font-size: 1rem; color: #F8FAFC; margin-top: 6px;">Sales Analytics</div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Daily revenue, weekday heatmaps, and basket distributions.</div>
    </div>
    """, unsafe_allow_html=True)

with m_c2:
    st.markdown("""
    <div class="metric-card" style="padding: 16px;">
        <div style="font-size: 1.4rem;">🎯</div>
        <div style="font-weight: 700; font-size: 1rem; color: #F8FAFC; margin-top: 6px;">Customer Segments</div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">RFM behavioral personas, 3D cluster space, and VIP rosters.</div>
    </div>
    """, unsafe_allow_html=True)

with m_c3:
    st.markdown("""
    <div class="metric-card" style="padding: 16px;">
        <div style="font-size: 1.4rem;">🔮</div>
        <div style="font-weight: 700; font-size: 1rem; color: #F8FAFC; margin-top: 6px;">Demand Forecasts</div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">ARIMA & Holt-Winters with 95% confidence bounds.</div>
    </div>
    """, unsafe_allow_html=True)

with m_c4:
    st.markdown("""
    <div class="metric-card" style="padding: 16px;">
        <div style="font-size: 1.4rem;">📑</div>
        <div style="font-weight: 700; font-size: 1rem; color: #F8FAFC; margin-top: 6px;">Executive Reports</div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">One-click PDF executive briefs and CSV manifests.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.8rem;'>"
    "RetailIQ Enterprise Suite • Production Business Intelligence & Demand Forecasting"
    "</div>",
    unsafe_allow_html=True
)
