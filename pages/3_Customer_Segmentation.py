"""
RetailIQ - Page 3: Customer Segmentation
RFM Analysis, Skewness Correction, K-Means Clustering, and Cohort Persona Diagnostics.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
from config import THEME_PRIMARY, THEME_ACCENT, THEME_SUCCESS, THEME_WARNING

st.set_page_config(page_title="Customer Segmentation | RetailIQ", page_icon="🎯", layout="wide")
inject_custom_css()

render_page_header(
    "Customer Segmentation (RFM & K-Means)",
    "Unsupervised machine learning clustering identifying high-value customer cohorts, retention risks, and engagement opportunities."
)

views, _ = get_active_dataset()
df_sales = views["clean_sales"]

# Interactive Controls Bar
col_k1, col_k2 = st.columns([1, 2])
with col_k1:
    k_clusters = st.select_slider(
        "Cluster Granularity (k)",
        options=[3, 4, 5, 6],
        value=4,
        help="Tunes K-Means cluster count. The mathematical optimum is k=4."
    )
with col_k2:
    st.markdown("""
    <div style="padding-top: 26px; font-size: 0.82rem; color: #94A3B8;">
        Target Cohorts: <b>Champions</b> (VIP) • <b>Loyal Customers</b> • <b>Potential Loyalists</b> • <b>At-Risk Buyers</b>
    </div>
    """, unsafe_allow_html=True)

# Compute or Fetch Cached Model
rfm_df, km_model, scaler, meta = get_cached_segmentation(df_sales, n_clusters=k_clusters)
seg_summary = pd.DataFrame(meta["segment_summary"])

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Eligible Identified Customers", format_number(len(rfm_df)), "100% Valid Accounts", "positive")
with c2:
    render_metric_card("Active Cluster Groups (k)", f"{k_clusters} Segments", f"Silhouette: {meta['silhouette_score']:.3f}", "positive")
with c3:
    render_metric_card("Total Cohort Spend", format_currency(meta["total_revenue"]), "Aggregated Monetary", "positive")
with c4:
    champions_rev = seg_summary.loc[seg_summary["Segment"] == "Champions", "TotalRevenue"].sum() if "Champions" in seg_summary["Segment"].values else 0
    render_metric_card("Champion Segment Spend", format_currency(champions_rev), "Top VIP Cohort", "positive")

st.markdown("<br/>", unsafe_allow_html=True)

# Visualizations: 2D/3D Projections & Revenue Distribution
tab1, tab2, tab3 = st.tabs(["🌐 2D & 3D Cluster Visualizations", "📊 Segment Share & Characteristics", "👥 Interactive Customer Directory"])

with tab1:
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("#### 2D Behavioral Feature Map (Log-Recency vs Log-Monetary)")
        rfm_sample = rfm_df.sample(min(3000, len(rfm_df)), random_state=42).copy()
        fig_2d = px.scatter(
            rfm_sample,
            x="Recency",
            y="Monetary",
            color="Segment",
            log_x=True,
            log_y=True,
            hover_data=["CustomerID", "Frequency"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.75
        )
        fig_2d.update_layout(height=420, xaxis_title="Recency (Days, Log Scale)", yaxis_title="Monetary (£, Log Scale)")
        st.plotly_chart(apply_plotly_theme(fig_2d), use_container_width=True)

    with v_col2:
        st.markdown("#### 3D Multivariate RFM Space (Recency × Frequency × Monetary)")
        fig_3d = px.scatter_3d(
            rfm_sample,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Segment",
            log_x=True,
            log_y=True,
            log_z=True,
            hover_data=["CustomerID"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7
        )
        fig_3d.update_layout(
            height=420,
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    c_s1, c_s2 = st.columns(2)

    with c_s1:
        st.markdown("#### Revenue Share by Customer Segment")
        fig_pie = px.pie(
            seg_summary,
            names="Segment",
            values="TotalRevenue",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=360)
        st.plotly_chart(apply_plotly_theme(fig_pie), use_container_width=True)

    with c_s2:
        st.markdown("#### Customer Population Share by Segment")
        fig_bar = px.bar(
            seg_summary,
            x="Segment",
            y="CustomerCount",
            color="Segment",
            text="CustomerCount",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bar.update_layout(height=360, yaxis_title="Number of Customers", showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig_bar), use_container_width=True)

    st.markdown("#### Segment Profile Diagnostic Matrix")
    display_matrix = seg_summary.copy()
    display_matrix["TotalRevenue"] = display_matrix["TotalRevenue"].apply(lambda v: f"£{v:,.2f}")
    display_matrix["AvgMonetary"] = display_matrix["AvgMonetary"].apply(lambda v: f"£{v:,.2f}")
    display_matrix["AvgRecency"] = display_matrix["AvgRecency"].apply(lambda v: f"{v:.1f} days")
    display_matrix["AvgFrequency"] = display_matrix["AvgFrequency"].apply(lambda v: f"{v:.1f} orders")
    display_matrix["CustomerShare_Pct"] = display_matrix["CustomerShare_Pct"].apply(lambda v: f"{v:.1f}%")
    display_matrix["RevenueShare_Pct"] = display_matrix["RevenueShare_Pct"].apply(lambda v: f"{v:.1f}%")
    st.dataframe(display_matrix, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("#### Filter Customers by Segment Persona")
    available_segs = ["All Segments"] + sorted(rfm_df["Segment"].unique().tolist())
    selected_seg = st.selectbox("Select Cohort Filter", options=available_segs)

    filtered_customers = rfm_df if selected_seg == "All Segments" else rfm_df[rfm_df["Segment"] == selected_seg]

    # Search by Customer ID
    cust_search = st.text_input("Search by Customer ID (Optional)", placeholder="e.g. 17850")
    if cust_search.strip():
        filtered_customers = filtered_customers[filtered_customers["CustomerID"].str.contains(cust_search.strip())]

    st.markdown(f"**Displaying {len(filtered_customers):,} customers:**")
    display_table = filtered_customers[["CustomerID", "Segment", "Recency", "Frequency", "Monetary", "Cluster"]].copy()
    display_table = display_table.sort_values("Monetary", ascending=False).reset_index(drop=True)
    display_table["Monetary"] = display_table["Monetary"].apply(lambda v: f"£{v:,.2f}")
    display_table["Recency"] = display_table["Recency"].apply(lambda v: f"{v} days")
    display_table["Frequency"] = display_table["Frequency"].apply(lambda v: f"{v} orders")

    st.dataframe(display_table.head(100), use_container_width=True, hide_index=True)
    st.caption("Showing top 100 customer records matching filter criteria.")
