"""
RetailIQ - Page 5: Product Analytics
SKU-level performance, revenue drivers, catalog velocity, and international distribution.
"""

import streamlit as st
import plotly.express as px
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
from src.eda import get_top_products
from config import THEME_PRIMARY, THEME_ACCENT, THEME_SUCCESS

st.set_page_config(page_title="Product Analytics | RetailIQ", page_icon="📦", layout="wide")
inject_custom_css()

render_page_header(
    "Product Analytics & SKU Intelligence",
    "Detailed analysis of catalog velocity, high-yield items, unit volumes, and cross-border popularity."
)

views, _ = get_active_dataset()
df_sales = views["clean_sales"]

# Interactive SKU Search & Filter
with st.expander("🔍 Search & Filter Product Catalog", expanded=True):
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search_query = st.text_input("Search by SKU Description or StockCode", placeholder="e.g. HEART, BAG, 85123A")
    with col_s2:
        sort_metric = st.selectbox("Rank By Metric", options=["Revenue", "Quantity", "Orders"], index=0)
    with col_s3:
        top_k = st.slider("Top N Display", min_value=5, max_value=50, value=15)

# Aggregate Product Summaries
prod_summary = (
    df_sales.groupby(["StockCode", "Description"])
    .agg(
        Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
        AvgPrice=("UnitPrice", "mean"),
        CountryCount=("Country", "nunique")
    )
    .reset_index()
)

# Apply Search Filter if entered
if search_query.strip():
    q = search_query.strip().lower()
    prod_summary = prod_summary[
        prod_summary["Description"].str.lower().str.contains(q) |
        prod_summary["StockCode"].str.lower().str.contains(q)
    ]

# Sort and slice
prod_ranked = prod_summary.sort_values(sort_metric, ascending=False).head(top_k).reset_index(drop=True)
total_catalog_rev = df_sales["Revenue"].sum()
prod_ranked["RevenueShare_Pct"] = (prod_ranked["Revenue"] / total_catalog_rev) * 100

# Top Catalog KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Active Catalog SKUs", format_number(df_sales["StockCode"].nunique()), "Unique Products", "positive")
with c2:
    top_sku_name = prod_ranked.iloc[0]["Description"][:18] if len(prod_ranked) > 0 else "N/A"
    top_sku_rev = prod_ranked.iloc[0]["Revenue"] if len(prod_ranked) > 0 else 0
    render_metric_card(f"Top Item ({top_sku_name})", format_currency(top_sku_rev), "Leading Revenue Driver", "positive")
with c3:
    render_metric_card("Total Units Dispatched", format_number(df_sales["Quantity"].sum()), "Units Across Orders", "neutral")
with c4:
    mean_price = df_sales["UnitPrice"].mean()
    render_metric_card("Average Catalog Price", f"£{mean_price:.2f}", "Across All Transactions", "neutral")

st.markdown("<br/>", unsafe_allow_html=True)

# Visualizations: Top SKUs Bar Chart & Product Country Matrix
col_v1, col_v2 = st.columns([3, 2])

with col_v1:
    st.markdown(f"#### Top {top_k} Products Ranked by {sort_metric}")
    prod_ranked["DisplayTitle"] = prod_ranked.apply(lambda r: f"{r['StockCode']} - {str(r['Description'])[:25]}", axis=1)
    fig_top = px.bar(
        prod_ranked.sort_values(sort_metric, ascending=True),
        x=sort_metric,
        y="DisplayTitle",
        orientation="h",
        color=sort_metric,
        color_continuous_scale="Viridis",
        text_auto=".2s"
    )
    fig_top.update_layout(height=480, xaxis_title=sort_metric, yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(apply_plotly_theme(fig_top), use_container_width=True)

with col_v2:
    st.markdown("#### Product Revenue vs. Quantity Quadrant")
    fig_scatter = px.scatter(
        prod_ranked,
        x="Quantity",
        y="Revenue",
        size="Orders",
        color="Revenue",
        hover_name="Description",
        hover_data=["StockCode", "AvgPrice"],
        color_continuous_scale="Plasma"
    )
    fig_scatter.update_layout(height=480, xaxis_title="Total Units Sold", yaxis_title="Total Revenue (£)")
    st.plotly_chart(apply_plotly_theme(fig_scatter), use_container_width=True)

# Product Detail Drilldown
st.markdown("### 📋 SKU Operational Table")
display_prod = prod_ranked[["StockCode", "Description", "Revenue", "Quantity", "Orders", "AvgPrice", "CountryCount", "RevenueShare_Pct"]].copy()
display_prod["Revenue"] = display_prod["Revenue"].apply(lambda v: f"£{v:,.2f}")
display_prod["Quantity"] = display_prod["Quantity"].apply(lambda v: f"{int(v):,}")
display_prod["Orders"] = display_prod["Orders"].apply(lambda v: f"{int(v):,}")
display_prod["AvgPrice"] = display_prod["AvgPrice"].apply(lambda v: f"£{v:,.2f}")
display_prod["RevenueShare_Pct"] = display_prod["RevenueShare_Pct"].apply(lambda v: f"{v:.2f}%")

st.dataframe(display_prod, use_container_width=True, hide_index=True)
