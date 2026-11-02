"""
RetailIQ - Page 4: Sales Forecasting
Time-series revenue projection using statistical ARIMA/SARIMAX and Holt-Winters models
with out-of-sample chronological validation and confidence intervals.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.utils import (
    inject_custom_css,
    render_metric_card,
    render_page_header,
    format_currency,
    apply_plotly_theme,
    get_active_dataset
)
from src.sales_forecasting import (
    prepare_sales_series,
    compare_forecasting_models,
    generate_future_forecast
)
from config import THEME_PRIMARY, THEME_ACCENT, THEME_SUCCESS, THEME_WARNING

st.set_page_config(page_title="Sales Forecasting | RetailIQ", page_icon="🔮", layout="wide")
inject_custom_css()

render_page_header(
    "Time-Series Sales Forecasting",
    "Chronologically validated statistical demand modeling projecting future retail revenue with 95% confidence bounds."
)

views, _ = get_active_dataset()
df_sales = views["clean_sales"]

# Modern Controls Bar
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    freq_choice = st.selectbox("Aggregation Cadence", options=["Daily ('D')", "Weekly ('W')"], index=0)
    freq_code = "D" if "Daily" in freq_choice else "W"
with col_f2:
    horizon_choice = st.selectbox("Projection Horizon", options=[7, 30, 90], index=1, format_func=lambda h: f"{h} Days Forward")
with col_f3:
    algo_choice = st.selectbox("Forecasting Engine", options=["ARIMA / SARIMAX", "Holt-Winters Exponential Smoothing"], index=0)

# Prepare continuous series
ts_series = prepare_sales_series(df_sales, frequency=freq_code)

# Run Comparative Evaluation on Hold-Out Test Set
with st.spinner("Training time-series models & computing out-of-sample metrics..."):
    comp_res = compare_forecasting_models(ts_series, test_size=30 if freq_code == "D" else 6)
    train_s = comp_res["train"]
    test_s = comp_res["test"]
    comp_table = comp_res["comparison_table"]

    selected_key = "ARIMA" if "ARIMA" in algo_choice else "ExponentialSmoothing"
    active_eval = comp_res["models"][selected_key]
    eval_metrics = active_eval["metrics"]
    test_preds = active_eval["predictions"]

    # Generate Forward-Looking Forecast
    future_pred, conf_int = generate_future_forecast(
        ts_series,
        model_type=selected_key,
        horizon_days=horizon_choice if freq_code == "D" else max(1, horizon_choice // 7)
    )

# Display Evaluation Metrics on Holdout Test Set
st.markdown("### 🎯 Out-of-Sample Validation Metrics (Final Holdout Period)")
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Mean Absolute Error (MAE)", f"£{eval_metrics['MAE']:,.2f}", "Holdout Deviation", "neutral")
with m2:
    render_metric_card("Root Mean Squared Error (RMSE)", f"£{eval_metrics['RMSE']:,.2f}", "Penalty on Large Errors", "neutral")
with m3:
    render_metric_card("Mean Abs Percentage Error", f"{eval_metrics['MAPE_Pct']:.1f}%", "Relative Error Rate", "positive" if eval_metrics['MAPE_Pct'] < 30 else "warning")
with m4:
    render_metric_card("Recommended Best Model", comp_res["best_model_name"], "Based on Lowest RMSE", "positive")

st.markdown("<br/>", unsafe_allow_html=True)

# Tabs: Out-of-Sample Actual vs Predicted & Future Projections
tab1, tab2, tab3 = st.tabs(["📉 Actual vs. Predicted (Holdout Test)", "🚀 Forward-Looking Projections", "📋 Comparative Benchmark"])

with tab1:
    st.markdown(f"#### Out-of-Sample Backtest: Actual vs {algo_choice} Predictions")
    fig_eval = go.Figure()

    # Contextual recent training data (last 60 days of train)
    recent_train = train_s.iloc[-60:]
    fig_eval.add_trace(go.Scatter(
        x=recent_train.index, y=recent_train.values,
        name="Historical Training Data", mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.6)", width=1.5)
    ))
    fig_eval.add_trace(go.Scatter(
        x=test_s.index, y=test_s.values,
        name="Actual Test Revenue", mode="lines+markers",
        line=dict(color=THEME_SUCCESS, width=2.5),
        marker=dict(size=5)
    ))
    fig_eval.add_trace(go.Scatter(
        x=test_preds.index, y=test_preds.values,
        name=f"{algo_choice} Predicted", mode="lines+markers",
        line=dict(color=THEME_PRIMARY, width=2.2, dash="dash"),
        marker=dict(size=5)
    ))
    fig_eval.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Revenue (£)",
        hovermode="x unified"
    )
    st.plotly_chart(apply_plotly_theme(fig_eval), use_container_width=True)

with tab2:
    st.markdown(f"#### Future {horizon_choice}-Day Projections (95% Confidence Interval)")
    fig_future = go.Figure()

    # Historical context (last 90 observations)
    hist_context = ts_series.iloc[-90:]
    fig_future.add_trace(go.Scatter(
        x=hist_context.index, y=hist_context.values,
        name="Historical Revenue", mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.7)", width=1.8)
    ))

    # Upper and Lower Confidence Bounds
    if conf_int is not None:
        fig_future.add_trace(go.Scatter(
            x=list(conf_int.index) + list(conf_int.index[::-1]),
            y=list(conf_int["UpperBound"]) + list(conf_int["LowerBound"][::-1]),
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Interval"
        ))

    # Future Point Forecast
    fig_future.add_trace(go.Scatter(
        x=future_pred.index, y=future_pred.values,
        name="Forecast Point Mean", mode="lines+markers",
        line=dict(color=THEME_ACCENT, width=2.5),
        marker=dict(size=4)
    ))

    fig_future.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Projected Revenue (£)",
        hovermode="x unified"
    )
    st.plotly_chart(apply_plotly_theme(fig_future), use_container_width=True)

with tab3:
    st.markdown("#### Model Benchmark Summary")
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

# Historical Limitations & Explanations
st.markdown("---")
st.markdown("### ⚠️ Methodological Limitations & Analytical Caveats")
st.warning("""
1. **Temporal Horizon Span**: The UCI Online Retail dataset covers exactly 374 calendar days (December 1, 2010 to December 9, 2011). Because only a single annual cycle is present, year-over-year annual seasonality cannot be separated from unique seasonal macroeconomic events.
2. **Holiday Surge Asymmetry**: The final weeks of November and early December 2011 exhibit a massive seasonal holiday spike. Models projecting forward without multiple years of historical baselines will naturally extrapolate recent holiday momentum.
3. **Weekly Store Operating Patterns**: The retail store recorded zero transactions on Saturdays, resulting in an inherent 7-day cyclical oscillation that ARIMA and Exponential Smoothing capture via weekly seasonal periods.
""")
