"""
RetailIQ - Page 1: Executive Overview
High-level KPI dashboard, temporal trends, country breakdown, and top-selling products.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.utils import (
    inject_custom_css,
    render_metric_card,
    render_page_header,
    format_currency,
    format_number,
    apply_plotly_theme,
    get_active_dataset
)
from src.eda import (
    compute_kpis,
    get_daily_sales_series,
    get_monthly_sales_summary,
    get_top_products,
    get_country_sales_summary
)
from config import THEME_PRIMARY, THEME_ACCENT, THEME_SUCCESS

st.set_page_config(page_title="Executive Overview | RetailIQ", page_icon="📊", layout="wide")
inject_custom_css()

# Header
render_page_header(
    "Executive Overview",
    "High-level enterprise performance metrics, operational revenue trends, and geographic distribution."
)

# Load Data
views, _ = get_active_dataset()
df_sales = views["clean_sales"]
df_returns = views["returns"]

# Global Filters in Expander
with st.expander("🔍 Interactive Filters (Date Range & Country)", expanded=False):
    f_col1, f_col2 = st.columns(2)
    min_date = df_sales["InvoiceDate"].min().date()
    max_date = df_sales["InvoiceDate"].max().date()

    with f_col1:
        date_range = st.date_input("Filter by Invoice Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    all_countries = sorted(df_sales["Country"].unique().tolist())
    with f_col2:
        selected_countries = st.multiselect("Filter by Country", options=all_countries, default=[])

# Apply filters
df_filtered = df_sales.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    df_filtered = df_filtered[(df_filtered["InvoiceDate"].dt.date >= start_d) & (df_filtered["InvoiceDate"].dt.date <= end_d)]
elif isinstance(date_range, tuple) and len(date_range) == 1:
    start_d = date_range[0]
    df_filtered = df_filtered[df_filtered["InvoiceDate"].dt.date >= start_d]

if selected_countries:
    df_filtered = df_filtered[df_filtered["Country"].isin(selected_countries)]

if len(df_filtered) == 0:
    st.warning("No records match the selected filter criteria. Please broaden your selection.")
    st.stop()

# Compute KPIs
kpis = compute_kpis(df_filtered, df_returns)

# KPI Cards Row
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    render_metric_card("Total Revenue", format_currency(kpis["total_revenue"]), "Gross Realized", "positive")
with c2:
    render_metric_card("Total Orders", format_number(kpis["total_orders"]), "Unique Invoices", "neutral")
with c3:
    render_metric_card("Unique Customers", format_number(kpis["unique_customers"]), "Active Buyers", "positive")
with c4:
    render_metric_card("Unique SKUs", format_number(kpis["unique_products"]), "Catalog Items", "neutral")
with c5:
    render_metric_card("Avg Order Value", format_currency(kpis["avg_order_value"]), "Per Checkout", "positive")
with c6:
    render_metric_card("Units Sold", format_number(kpis["total_units_sold"]), "Items Shipped", "neutral")

st.markdown("<br/>", unsafe_allow_html=True)

# Visualizations Row 1: Revenue Trends & Monthly Performance
v_col1, v_col2 = st.columns([3, 2])

with v_col1:
    st.markdown("### 📈 Daily Revenue Trend (with 7-Day & 30-Day Moving Averages)")
    daily_df = get_daily_sales_series(df_filtered)
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=daily_df["Date"], y=daily_df["Revenue"],
        name="Daily Revenue", mode="lines",
        line=dict(color="rgba(79, 70, 229, 0.4)", width=1.2)
    ))
    fig_daily.add_trace(go.Scatter(
        x=daily_df["Date"], y=daily_df["7D_MA"],
        name="7-Day Moving Avg", mode="lines",
        line=dict(color=THEME_ACCENT, width=2.2)
    ))
    fig_daily.add_trace(go.Scatter(
        x=daily_df["Date"], y=daily_df["30D_MA"],
        name="30-Day Moving Avg", mode="lines",
        line=dict(color=THEME_SUCCESS, width=2.5, dash="dot")
    ))
    fig_daily.update_layout(xaxis_title="Date", yaxis_title="Revenue (£)", height=380, hovermode="x unified")
    st.plotly_chart(apply_plotly_theme(fig_daily), use_container_width=True)

with v_col2:
    st.markdown("### 📅 Monthly Sales Performance")
    monthly_df = get_monthly_sales_summary(df_filtered)
    fig_monthly = px.bar(
        monthly_df, x="YearMonth", y="Revenue",
        text_auto=".2s", color="Revenue",
        color_continuous_scale="Purples",
        title=""
    )
    fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Revenue (£)", height=380, coloraxis_showscale=False)
    st.plotly_chart(apply_plotly_theme(fig_monthly), use_container_width=True)

# Visualizations Row 2: Top Products & Country Distribution
v_col3, v_col4 = st.columns([1, 1])

with v_col3:
    st.markdown("### 🏆 Top 10 Revenue-Generating Products")
    top_p = get_top_products(df_filtered, top_n=10, by="Revenue")
    # Truncate long descriptions
    top_p["ShortDesc"] = top_p["Description"].apply(lambda s: s[:28] + "..." if len(str(s)) > 28 else str(s))
    fig_prod = px.bar(
        top_p.sort_values("Revenue", ascending=True),
        x="Revenue", y="ShortDesc", orientation="h",
        text_auto=".2s", color="Revenue",
        color_continuous_scale="Viridis"
    )
    fig_prod.update_layout(xaxis_title="Revenue (£)", yaxis_title="", height=400, coloraxis_showscale=False)
    st.plotly_chart(apply_plotly_theme(fig_prod), use_container_width=True)

with v_col4:
    st.markdown("### 🌍 Geographic Revenue Distribution")
    country_df = get_country_sales_summary(df_filtered).head(8)
    fig_country = px.pie(
        country_df, names="Country", values="Revenue",
        hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_country.update_traces(textposition="inside", textinfo="percent+label")
    fig_country.update_layout(height=400, showlegend=False)
    st.plotly_chart(apply_plotly_theme(fig_country), use_container_width=True)
