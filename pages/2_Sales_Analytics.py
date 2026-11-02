"""
RetailIQ - Page 2: Sales Analytics
Granular revenue dynamics, temporal purchasing patterns, and growth trends.
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
    get_daily_sales_series,
    get_monthly_sales_summary,
    get_country_sales_summary,
    get_top_products,
    get_hourly_and_weekday_patterns
)
from config import THEME_PRIMARY, THEME_ACCENT, THEME_SUCCESS

st.set_page_config(page_title="Sales Analytics | RetailIQ", page_icon="📈", layout="wide")
inject_custom_css()

render_page_header(
    "Sales Analytics & Commercial Insights",
    "Detailed analysis of transaction volumes, order values, cyclical patterns, and month-over-month growth."
)

views, _ = get_active_dataset()
df_sales = views["clean_sales"]

# Summary Series
daily_df = get_daily_sales_series(df_sales)
monthly_df = get_monthly_sales_summary(df_sales)
weekday_df, hourly_df = get_hourly_and_weekday_patterns(df_sales)

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
latest_month = monthly_df.iloc[-1]
prev_month = monthly_df.iloc[-2] if len(monthly_df) > 1 else latest_month
mom_growth = latest_month["MoM_Growth_Pct"] if pd.notnull(latest_month["MoM_Growth_Pct"]) else 0.0

with m1:
    render_metric_card("Peak Daily Revenue", format_currency(daily_df["Revenue"].max()), f"Mean: {format_currency(daily_df['Revenue'].mean())}/day", "positive")
with m2:
    render_metric_card("Latest Month Sales", format_currency(latest_month["Revenue"]), f"{mom_growth:+.1f}% vs Prev Month", "positive" if mom_growth >= 0 else "negative")
with m3:
    render_metric_card("Total Orders Processed", format_number(daily_df["Orders"].sum()), "Valid Transactions", "neutral")
with m4:
    render_metric_card("Overall Basket AOV", format_currency(df_sales["Revenue"].sum() / df_sales["InvoiceNo"].nunique()), "Per Order Checkout", "positive")

st.markdown("<br/>", unsafe_allow_html=True)

# Tabs for Organized Analysis
tab1, tab2, tab3 = st.tabs(["📊 Revenue & Growth Dynamics", "⏰ Temporal Purchasing Patterns", "🌍 Geographic & Basket Distribution"])

with tab1:
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("#### Daily Revenue Fluctuations & Moving Trends")
        fig_d = px.line(daily_df, x="Date", y=["Revenue", "7D_MA"], color_discrete_sequence=[THEME_PRIMARY, THEME_ACCENT])
        fig_d.update_layout(height=360, xaxis_title="Date", yaxis_title="Revenue (£)", legend_title_text="Metric")
        st.plotly_chart(apply_plotly_theme(fig_d), use_container_width=True)

    with col_t2:
        st.markdown("#### Month-over-Month (MoM) Growth Rate (%)")
        growth_df = monthly_df.dropna(subset=["MoM_Growth_Pct"]).copy()
        fig_g = px.bar(
            growth_df, x="YearMonth", y="MoM_Growth_Pct",
            color="MoM_Growth_Pct",
            color_continuous_scale="RdYlGn",
            text_auto=".1f"
        )
        fig_g.update_layout(height=360, xaxis_title="Month", yaxis_title="Growth (%)", coloraxis_showscale=False)
        st.plotly_chart(apply_plotly_theme(fig_g), use_container_width=True)

    st.markdown("#### Monthly Performance Summary Table")
    display_monthly = monthly_df.copy()
    display_monthly["Revenue"] = display_monthly["Revenue"].apply(lambda v: f"£{v:,.2f}")
    display_monthly["AOV"] = display_monthly["AOV"].apply(lambda v: f"£{v:,.2f}")
    display_monthly["MoM_Growth_Pct"] = display_monthly["MoM_Growth_Pct"].apply(lambda v: f"{v:+.1f}%" if pd.notnull(v) else "N/A")
    st.dataframe(display_monthly, use_container_width=True, hide_index=True)

with tab2:
    col_w, col_h = st.columns(2)
    with col_w:
        st.markdown("#### Weekly Sales Distribution (Day of Week)")
        fig_w = px.bar(
            weekday_df, x="DayOfWeek", y="Revenue",
            color="Revenue", color_continuous_scale="Purples",
            text_auto=".2s"
        )
        fig_w.update_layout(height=350, xaxis_title="Day of Week", yaxis_title="Revenue (£)", coloraxis_showscale=False)
        st.plotly_chart(apply_plotly_theme(fig_w), use_container_width=True)
        st.caption("Notice: E-commerce operations in this retail dataset are concentrated between Sunday and Friday, with zero trade logged on Saturdays.")

    with col_h:
        st.markdown("#### Hourly Purchasing Concentration (Hour of Day)")
        fig_h = px.line(
            hourly_df, x="Hour", y="Revenue", markers=True,
            line_shape="spline", color_discrete_sequence=[THEME_ACCENT]
        )
        fig_h.update_layout(height=350, xaxis_title="Hour (24h format)", yaxis_title="Revenue (£)")
        st.plotly_chart(apply_plotly_theme(fig_h), use_container_width=True)
        st.caption("Peak commercial buying activity surges between 11:00 AM and 3:00 PM GMT.")

with tab3:
    col_geo, col_dist = st.columns(2)
    with col_geo:
        st.markdown("#### Country-Wise Revenue Contribution (Top 10)")
        country_df = get_country_sales_summary(df_sales).head(10)
        fig_c = px.bar(
            country_df, x="Revenue", y="Country", orientation="h",
            color="Revenue", color_continuous_scale="Blues", text_auto=".2s"
        )
        fig_c.update_layout(height=380, xaxis_title="Revenue (£)", yaxis_title="Country", coloraxis_showscale=False)
        st.plotly_chart(apply_plotly_theme(fig_c), use_container_width=True)

    with col_dist:
        st.markdown("#### Order Value Distribution (Binned Checkout Revenue)")
        # Calculate order totals
        order_totals = df_sales.groupby("InvoiceNo")["Revenue"].sum()
        # Cap outlier visual at 99th percentile for clean rendering
        cap_val = order_totals.quantile(0.98)
        filtered_orders = order_totals[order_totals <= cap_val]
        fig_hist = px.histogram(
            filtered_orders, nbins=40,
            color_discrete_sequence=[THEME_PRIMARY],
            labels={"value": "Order Value (£)"}
        )
        fig_hist.update_layout(height=380, xaxis_title="Order Total (£)", yaxis_title="Invoice Count", showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig_hist), use_container_width=True)
