"""
RetailIQ - Page 7: Data Explorer & Quality Diagnostics
Complete dataset audit, schema inspection, missing values, anomaly tracking, and clean data export.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.utils import (
    inject_custom_css,
    render_metric_card,
    render_page_header,
    format_number,
    get_active_dataset
)
from src.data_loader import parse_uploaded_file
from src.preprocessing import clean_retail_data
from config import THEME_PRIMARY, THEME_ACCENT, THEME_WARNING, THEME_DANGER

st.set_page_config(page_title="Data Explorer | RetailIQ", page_icon="🔍", layout="wide")
inject_custom_css()

render_page_header(
    "Data Explorer & Ingestion Diagnostics",
    "Comprehensive audit of dataset hygiene, schema integrity, missing data distribution, cancellations, and custom dataset upload."
)

views, audit_metrics = get_active_dataset()
df_all = views["all"]
df_clean = views["clean_sales"]
df_returns = views["returns"]

# KPI Metrics Summary
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_metric_card("Raw Ingested Records", format_number(audit_metrics["initial_rows"]), "Full Transaction File", "neutral")
with m2:
    render_metric_card("Pruned Duplicate Rows", format_number(audit_metrics["duplicate_count"]), f"{audit_metrics['duplicate_count']/audit_metrics['initial_rows']*100:.2f}% of Total", "warning")
with m3:
    render_metric_card("Missing CustomerIDs", format_number(audit_metrics["missing_customer_count"]), "Guest Shoppers (~24.9%)", "warning")
with m4:
    render_metric_card("Invoiced Cancellations", format_number(audit_metrics["cancellations_count"]), "Prefixed with 'C'", "negative")
with m5:
    render_metric_card("Clean Customer Sales", format_number(audit_metrics["clean_customer_sales_rows"]), "Production Analytics View", "positive")

st.markdown("<br/>", unsafe_allow_html=True)

# Tabs for Structured Data Inspection
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Previews & Partition Views", "🔍 Missing Values & Schema", "⚠️ Anomalies & Returns Analysis", "💾 Clean Dataset Export"])

with tab1:
    st.markdown("#### Transaction Partition Selector")
    view_selection = st.radio(
        "Select Dataset View to Inspect",
        options=["Clean Customer Sales (df_clean_sales)", "Macro Realized Sales (df_macro_sales)", "Returns & Cancellations (df_returns)", "Full Enriched Transactions (df_all)"],
        horizontal=True
    )
    
    if "Clean Customer" in view_selection:
        target_df = df_clean
    elif "Macro Realized" in view_selection:
        target_df = views["macro_sales"]
    elif "Returns" in view_selection:
        target_df = df_returns
    else:
        target_df = df_all

    st.markdown(f"**Shape:** `{target_df.shape[0]:,}` rows × `{target_df.shape[1]}` columns")
    st.dataframe(target_df.head(100), use_container_width=True)

with tab2:
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### Schema & Data Types")
        schema_df = pd.DataFrame({
            "Column Name": list(df_all.columns),
            "Data Type": [str(t) for t in df_all.dtypes.values],
            "Non-Null Count": [int(df_all[c].notnull().sum()) for c in df_all.columns],
            "Null Count": [int(df_all[c].isnull().sum()) for c in df_all.columns]
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    with col_d2:
        st.markdown("#### Missing Values Distribution (%)")
        null_series = (df_all.isnull().sum() / len(df_all) * 100).round(2)
        null_df = pd.DataFrame({
            "Column": null_series.index,
            "Null Percentage (%)": null_series.values
        }).sort_values("Null Percentage (%)", ascending=False)
        st.dataframe(null_df, use_container_width=True, hide_index=True)

        st.info(
            "📌 **Engineering Treatment**: Missing descriptions were imputed from historical descriptions matching the identical `StockCode`. "
            "Missing `CustomerID` records represent unregistered guest transactions, which are retained for macro sales aggregates but excluded from customer segmentation."
        )

with tab3:
    st.markdown("#### Returns & Price Anomaly Breakdown")
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("**Sample Cancellation Transactions (InvoiceNo starting with 'C'):**")
        cancels = df_returns[df_returns["IsCancelled"]].head(15)
        st.dataframe(cancels[["InvoiceNo", "InvoiceDate", "StockCode", "Quantity", "UnitPrice", "Revenue", "CustomerID"]], use_container_width=True, hide_index=True)

    with col_a2:
        st.markdown("**Sample Non-Positive Price Records (UnitPrice <= 0):**")
        zero_prices = df_all[df_all["UnitPrice"] <= 0].head(15)
        st.dataframe(zero_prices[["InvoiceNo", "InvoiceDate", "StockCode", "Description", "Quantity", "UnitPrice", "TransactionType"]], use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### 📤 Upload New Dataset (CSV, Excel, or PDF)")
    st.markdown("Easily ingest your company's transaction logs in **CSV**, **Excel**, or **PDF** format:")
    up_file = st.file_uploader("Upload Transaction File", type=["csv", "xlsx", "xls", "pdf"], key="explorer_uploader")
    if up_file is not None:
        try:
            with st.spinner("Parsing and cleaning custom dataset..."):
                raw_df, sum_str = parse_uploaded_file(up_file)
                u_views, u_metrics = clean_retail_data(raw_df, save_processed=False)
                st.session_state["custom_views"] = u_views
                st.session_state["custom_metrics"] = u_metrics
                st.session_state["custom_source"] = sum_str
                st.success(f"✅ Ingested: **{sum_str}**")
                st.rerun()
        except Exception as err:
            st.error(f"Upload Error: {err}")

    st.markdown("---")
    st.markdown("### 📥 Download Cleaned Dataset")
    st.markdown("""
    Download the verified, deduplicated, and preprocessed transaction records ready for external machine learning, reporting, or BI ingestion.
    """)

    csv_bytes = df_clean.head(10000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Clean Customer Sales (Top 10k Records CSV)",
        data=csv_bytes,
        file_name="retail_iq_cleaned_transactions_sample.csv",
        mime="text/csv",
        help="Exports clean transactions with positive revenue and valid customer IDs."
    )
