"""
RetailIQ - Page 6: Customer Insights
Individual customer transaction audit, observed historical spend patterns,
and cohort engagement profiles.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

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
from config import THEME_PRIMARY, THEME_ACCENT, THEME_SUCCESS, THEME_WARNING, THEME_DANGER

st.set_page_config(page_title="Customer Insights | RetailIQ", page_icon="👥", layout="wide")
inject_custom_css()

render_page_header(
    "Customer Insights & Account Diagnostics",
    "Comprehensive exploration of observed historical customer value, transaction frequency, and individual purchasing histories."
)

views, _ = get_active_dataset()
df_sales = views["clean_sales"]

# Load segmented customers
rfm_df, _, _, meta = get_cached_segmentation(df_sales, n_clusters=4)

# Top KPI Metrics Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    mean_spend = rfm_df["Monetary"].mean()
    render_metric_card("Average Customer Spend", format_currency(mean_spend), "Observed Lifetime Mean", "neutral")
with c2:
    median_spend = rfm_df["Monetary"].median()
    render_metric_card("Median Customer Spend", format_currency(median_spend), "50th Percentile Spend", "neutral")
with c3:
    repeat_customers = (rfm_df["Frequency"] > 1).sum()
    repeat_rate = (repeat_customers / len(rfm_df)) * 100
    render_metric_card("Repeat Customer Rate", f"{repeat_rate:.1f}%", f"{repeat_customers:,} Multi-Order Buyers", "positive")
with c4:
    top_10pct_rev = rfm_df.nlargest(int(len(rfm_df) * 0.1), "Monetary")["Monetary"].sum()
    pareto_share = (top_10pct_rev / rfm_df["Monetary"].sum()) * 100
    render_metric_card("Top 10% Spend Share", f"{pareto_share:.1f}%", "Classic Pareto Distribution", "positive")

st.markdown("<br/>", unsafe_allow_html=True)

st.caption("💡 Customer value metrics represent cumulative realized revenue recorded across historical account checkouts.")

# Individual Customer Lookup
st.markdown("### 🔎 Individual Customer Account Deep Dive")
col_l1, col_l2 = st.columns([1, 2])

with col_l1:
    # High value suggestions
    top_5_ids = rfm_df.nlargest(5, "Monetary")["CustomerID"].tolist()
    st.markdown(f"**Top 5 Spenders in Dataset:** {', '.join(top_5_ids)}")
    selected_customer_id = st.text_input("Enter Customer ID to Inspect", value=top_5_ids[0])

if selected_customer_id.strip():
    c_id = selected_customer_id.strip()
    cust_rfm = rfm_df[rfm_df["CustomerID"] == c_id]

    if len(cust_rfm) > 0:
        c_record = cust_rfm.iloc[0]
        cust_tx = df_sales[df_sales["CustomerID"] == c_id].sort_values("InvoiceDate", ascending=False)

        with col_l2:
            st.markdown(f"#### Account Summary: Customer #{c_id}")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Segment", c_record["Segment"])
            with sc2:
                st.metric("Total Spent", format_currency(c_record["Monetary"]))
            with sc3:
                st.metric("Order Count", f"{c_record['Frequency']} orders")
            with sc4:
                st.metric("Days Since Last Buy", f"{c_record['Recency']} days ago")

        st.markdown(f"#### Transaction History for Customer #{c_id} ({len(cust_tx):,} line items)")
        tx_display = cust_tx[["InvoiceNo", "InvoiceDate", "StockCode", "Description", "Quantity", "UnitPrice", "Revenue", "Country"]].copy()
        tx_display["InvoiceDate"] = tx_display["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
        tx_display["UnitPrice"] = tx_display["UnitPrice"].apply(lambda v: f"£{v:,.2f}")
        tx_display["Revenue"] = tx_display["Revenue"].apply(lambda v: f"£{v:,.2f}")
        st.dataframe(tx_display.head(50), use_container_width=True, hide_index=True)
    else:
        st.warning(f"Customer ID '{c_id}' not found in clean customer records.")

st.markdown("---")

# High Value vs At-Risk Cohorts
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("### 💎 Top 10 High-Value Customers (Observed Spend)")
    top_cust = rfm_df.sort_values("Monetary", ascending=False).head(10).copy()
    fig_top_c = px.bar(
        top_cust.sort_values("Monetary", ascending=True),
        x="Monetary", y="CustomerID", orientation="h",
        color="Monetary", color_continuous_scale="Purples", text_auto=".2s"
    )
    fig_top_c.update_layout(height=360, xaxis_title="Observed Spend (£)", yaxis_title="Customer ID", coloraxis_showscale=False)
    st.plotly_chart(apply_plotly_theme(fig_top_c), use_container_width=True)

with col_c2:
    st.markdown("### ⚠️ At-Risk High-Value Customers (Lapsing Buyers)")
    # High historical monetary but recency > 90 days
    at_risk = rfm_df[(rfm_df["Recency"] > 90) & (rfm_df["Monetary"] > rfm_df["Monetary"].quantile(0.75))].sort_values("Monetary", ascending=False).head(10)
    fig_risk = px.scatter(
        at_risk,
        x="Recency", y="Monetary", size="Frequency",
        color="Recency", color_continuous_scale="Reds",
        hover_data=["CustomerID", "Frequency"]
    )
    fig_risk.update_layout(height=360, xaxis_title="Days Inactive (Recency)", yaxis_title="Historical Spend (£)")
    st.plotly_chart(apply_plotly_theme(fig_risk), use_container_width=True)
