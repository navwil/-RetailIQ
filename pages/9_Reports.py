"""
RetailIQ - Page 9: Executive & Academic Reports
Automated generation and instant download of PDF executive summaries,
segmentation manifests, and comprehensive academic project documentation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import io

from src.utils import (
    inject_custom_css,
    render_metric_card,
    render_page_header,
    format_currency,
    get_active_dataset,
    get_cached_segmentation
)
from src.eda import compute_kpis, get_top_products, get_country_sales_summary
from src.sales_forecasting import prepare_sales_series, compare_forecasting_models
from src.report_generator import generate_academic_project_report, generate_executive_sales_report
from config import REPORTS_DIR, THEME_PRIMARY, THEME_ACCENT

st.set_page_config(page_title="Executive Reports | RetailIQ", page_icon="📑", layout="wide")
inject_custom_css()

render_page_header(
    "Report Generation & Executive Exports",
    "Publish and export executive summary briefs, segment rosters, forecast manifests, and complete retail intelligence documentation in PDF and CSV formats."
)

views, _ = get_active_dataset()
df_sales = views["clean_sales"]
df_returns = views["returns"]

# Load Active Segmentation Models & RFM
rfm_df, _, _, meta = get_cached_segmentation(df_sales, n_clusters=4)
seg_summary = pd.DataFrame(meta["segment_summary"])
kpis = compute_kpis(df_sales, df_returns)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Comprehensive Intelligence Report (Complete PDF)")
    st.markdown("""
    Download the complete, publication-ready **Retail Intelligence System Report** formatted across 10 executive chapters:
    - Problem Statement, Commercial Objectives, and Analytical Formulations
    - Retail Hygiene, Ingestion & Cancellation Handling
    - RFM Normalization & K-Means Cohort Findings
    - Time-Series Demand Forecasting Benchmarks
    - Real experimental data, metrics, tables, and references
    """)

    if st.button("📄 Generate & Refresh Intelligence Report (PDF)", key="btn_acad"):
        with st.spinner("Compiling intelligence report with active experimental metrics..."):
            ts_series = prepare_sales_series(df_sales, frequency="D")
            comp_res = compare_forecasting_models(ts_series, test_size=30)
            pdf_path = generate_academic_project_report(
                kpis=kpis,
                segment_summary=seg_summary,
                forecast_metrics=comp_res["comparison_table"]
            )
            st.success("✅ Intelligence Report successfully compiled!")

    report_file = REPORTS_DIR / "RetailIQ_Intelligence_Report.pdf"
    if not report_file.exists():
        # Compile if not already present
        generate_academic_project_report(kpis=kpis, segment_summary=seg_summary)

    if report_file.exists():
        with open(report_file, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="📥 Download Comprehensive Intelligence Report (PDF)",
            data=pdf_bytes,
            file_name="RetailIQ_Comprehensive_Intelligence_Report.pdf",
            mime="application/pdf"
        )

with col2:
    st.markdown("### 📊 Executive Sales Analysis Report (PDF)")
    st.markdown("""
    Download a high-level C-Suite executive brief summarizing commercial operations:
    - Realized gross sales, completed orders, and overall AOV
    - Top revenue-generating products and category distribution
    - Geographic international market penetration
    """)

    if st.button("📄 Generate Executive Sales Brief (PDF)", key="btn_sales"):
        with st.spinner("Generating executive brief..."):
            top_p = get_top_products(df_sales, top_n=10)
            country_s = get_country_sales_summary(df_sales).head(10)
            s_pdf = generate_executive_sales_report(kpis, top_p, country_s)
            st.success("✅ Executive Sales Brief generated!")

    sales_report_file = REPORTS_DIR / "RetailIQ_Sales_Executive_Report.pdf"
    if sales_report_file.exists():
        with open(sales_report_file, "rb") as f:
            s_bytes = f.read()
        st.download_button(
            label="📥 Download Executive Sales Report (PDF)",
            data=s_bytes,
            file_name="RetailIQ_Sales_Executive_Report.pdf",
            mime="application/pdf"
        )

st.markdown("---")
st.markdown("### 💾 Raw Tabular Manifest Downloads (CSV)")

c_csv1, c_csv2, c_csv3 = st.columns(3)

with c_csv1:
    st.markdown("#### 🎯 Customer Segments Manifest")
    st.caption(f"Exports all {len(rfm_df):,} customers with assigned RFM metrics and clusters.")
    csv_seg = rfm_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Customer Segments (CSV)",
        data=csv_seg,
        file_name="retail_iq_customer_segments.csv",
        mime="text/csv"
    )

with c_csv2:
    st.markdown("#### 📈 Daily Sales Time-Series")
    st.caption("Exports continuous regular daily revenue and transaction counts.")
    ts_export = prepare_sales_series(df_sales, frequency="D").reset_index()
    ts_export.columns = ["Date", "DailyRevenue"]
    csv_ts = ts_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Daily Sales Series (CSV)",
        data=csv_ts,
        file_name="retail_iq_daily_sales_series.csv",
        mime="text/csv"
    )

with c_csv3:
    st.markdown("#### 🏆 Product Performance Roster")
    st.caption("Exports aggregated SKU metrics (units, revenue, order count).")
    prod_export = get_top_products(df_sales, top_n=500, by="Revenue")
    csv_prod = prod_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Product Catalog Metrics (CSV)",
        data=csv_prod,
        file_name="retail_iq_product_metrics.csv",
        mime="text/csv"
    )
