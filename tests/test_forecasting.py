"""
Unit tests for time-series sales forecasting and model evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from src.sales_forecasting import (
    prepare_sales_series,
    chronological_train_test_split,
    compare_forecasting_models,
    generate_future_forecast
)


@pytest.fixture
def mock_daily_sales():
    """Generates continuous daily sales data spanning 120 days."""
    np.random.seed(42)
    dates = pd.date_range(start="2011-01-01", periods=120, freq="D")
    records = []
    for d in dates:
        # Simulate some weekday patterns
        multiplier = 1.5 if d.dayofweek in [0, 4] else 1.0
        revenue = max(100.0, float(np.random.normal(1500 * multiplier, 300)))
        records.append({
            "InvoiceDate": d,
            "Revenue": revenue
        })
    return pd.DataFrame(records)


def test_prepare_sales_series(mock_daily_sales):
    ts = prepare_sales_series(mock_daily_sales, frequency="D")
    assert len(ts) == 120
    assert ts.index.freqstr == "D"
    assert (ts >= 0).all()


def test_chronological_train_test_split(mock_daily_sales):
    ts = prepare_sales_series(mock_daily_sales, frequency="D")
    train, test = chronological_train_test_split(ts, test_size=20)
    assert len(train) == 100
    assert len(test) == 20
    assert train.index.max() < test.index.min()


def test_compare_forecasting_models(mock_daily_sales):
    ts = prepare_sales_series(mock_daily_sales, frequency="D")
    res = compare_forecasting_models(ts, test_size=15)
    
    assert "ARIMA" in res["models"]
    assert "ExponentialSmoothing" in res["models"]
    assert len(res["comparison_table"]) == 2
    
    arima_metrics = res["models"]["ARIMA"]["metrics"]
    assert "MAE" in arima_metrics
    assert "RMSE" in arima_metrics
    assert arima_metrics["MAE"] > 0
    assert arima_metrics["RMSE"] > 0


def test_generate_future_forecast(mock_daily_sales):
    ts = prepare_sales_series(mock_daily_sales, frequency="D")
    horizon = 14
    forecast, conf_int = generate_future_forecast(ts, model_type="ARIMA", horizon_days=horizon)
    
    assert len(forecast) == horizon
    assert forecast.index[0] == ts.index.max() + pd.Timedelta(days=1)
    assert conf_int is not None
    assert len(conf_int) == horizon
    assert (conf_int["LowerBound"] <= conf_int["UpperBound"]).all()
