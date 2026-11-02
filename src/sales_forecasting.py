"""
RetailIQ - Sales Forecasting Module
Implements time-series preparation, chronological train-test validation,
statistical forecasting using ARIMA and Holt-Winters Exponential Smoothing,
future projection with confidence intervals, and comparative performance evaluation.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.model_evaluation import compute_forecasting_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress minor convergence warnings for production dashboard UX
warnings.filterwarnings("ignore")


def prepare_sales_series(df_sales: pd.DataFrame, frequency: str = "D") -> pd.Series:
    """
    Transforms transactional sales into a continuous, regular time-series indexed by date.
    Missing dates within the min-max span are filled with 0 to prevent timestamp gaps.
    
    Parameters:
        df_sales: Sales transactions with InvoiceDate and Revenue.
        frequency: 'D' (daily), 'W' (weekly), or 'M' (monthly).
    """
    df = df_sales.copy()
    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.date
    daily = df.groupby("Date")["Revenue"].sum().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.set_index("Date").sort_index()

    if frequency.upper() == "D":
        # Continuous daily index
        full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
        ts = daily.reindex(full_idx, fill_value=0.0)["Revenue"]
        ts.index.name = "Date"
        return ts
    elif frequency.upper() == "W":
        weekly = daily.resample("W-MON").sum()["Revenue"]
        weekly.index.name = "Date"
        return weekly
    elif frequency.upper() == "M":
        monthly = daily.resample("MS").sum()["Revenue"]
        monthly.index.name = "Date"
        return monthly
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Choose 'D', 'W', or 'M'.")


def chronological_train_test_split(
    ts_series: pd.Series,
    test_size: int = 30
) -> Tuple[pd.Series, pd.Series]:
    """
    Performs chronological train-test split strictly preventing lookahead bias.
    """
    n = len(ts_series)
    if n < 4:
        raise ValueError(f"Time series too short for forecasting ({n} points). Minimum 4 observations required.")

    if n <= test_size + 5:
        # Guarantee at least 70% in train and at least 1 in test
        test_size = max(1, int(n * 0.2))

    # Ensure train has at least 3 points
    if n - test_size < 3:
        test_size = max(1, n - 3)

    train = ts_series.iloc[:-test_size]
    test = ts_series.iloc[-test_size:]
    return train, test


def fit_arima(
    train_series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 0, 1, 7)
) -> Any:
    """
    Fits an ARIMA / SARIMAX statistical model.
    Default order incorporates differencing and weekly seasonal periodicity for retail data.
    """
    # Guard against too short series for seasonal order
    if seasonal_order and len(train_series) < 2 * seasonal_order[3]:
        seasonal_order = None

    try:
        model = ARIMA(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit()
        return fitted
    except Exception as e:
        logger.warning(f"ARIMA fitting fallback due to: {e}. Fitting simple ARIMA(1,0,0).")
        try:
            model = ARIMA(train_series, order=(1, 1, 0))
            return model.fit()
        except Exception:
            model = ARIMA(train_series, order=(1, 0, 0))
            return model.fit()


def fit_exponential_smoothing(
    train_series: pd.Series,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
    seasonal_periods: int = 7
) -> Any:
    """
    Fits a Holt-Winters Exponential Smoothing model.
    """
    try:
        # Require positive values for multiplicative, or use additive
        model = ExponentialSmoothing(
            train_series,
            trend=trend,
            seasonal=seasonal if len(train_series) >= 2 * seasonal_periods else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        return model.fit()
    except Exception as e:
        logger.warning(f"Exponential Smoothing error ({e}). Fitting simple additive trend.")
        model = ExponentialSmoothing(train_series, trend="add", seasonal=None)
        return model.fit()


def compare_forecasting_models(
    ts_series: pd.Series,
    test_size: int = 30
) -> Dict[str, Any]:
    """
    Trains multiple models on historical training data, predicts over the hold-out test set,
    and returns comparative metrics and test visualization series.
    """
    train, test = chronological_train_test_split(ts_series, test_size=test_size)
    results = {}

    # 1. ARIMA / SARIMAX
    arima_fit = fit_arima(train)
    arima_forecast_obj = arima_fit.get_forecast(steps=len(test))
    arima_pred = arima_forecast_obj.predicted_mean
    arima_pred.index = test.index
    # Bound predictions at 0 (sales cannot be negative)
    arima_pred = np.maximum(arima_pred, 0)
    arima_metrics = compute_forecasting_metrics(test, arima_pred)

    results["ARIMA"] = {
        "model": arima_fit,
        "predictions": arima_pred,
        "metrics": arima_metrics
    }

    # 2. Holt-Winters Exponential Smoothing
    hw_fit = fit_exponential_smoothing(train)
    hw_pred = hw_fit.forecast(steps=len(test))
    hw_pred.index = test.index
    hw_pred = np.maximum(hw_pred, 0)
    hw_metrics = compute_forecasting_metrics(test, hw_pred)

    results["ExponentialSmoothing"] = {
        "model": hw_fit,
        "predictions": hw_pred,
        "metrics": hw_metrics
    }

    # Summary table
    comparison_table = pd.DataFrame([
        {"Model": "ARIMA / SARIMAX", **arima_metrics},
        {"Model": "Holt-Winters Exp Smoothing", **hw_metrics}
    ])

    best_model_name = "ARIMA" if arima_metrics["RMSE"] <= hw_metrics["RMSE"] else "ExponentialSmoothing"

    return {
        "train": train,
        "test": test,
        "models": results,
        "comparison_table": comparison_table,
        "best_model_name": best_model_name
    }


def generate_future_forecast(
    ts_series: pd.Series,
    model_type: str = "ARIMA",
    horizon_days: int = 30
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Fits model on the entire historical dataset and generates forward-looking forecast
    along with 95% confidence intervals if supported.
    
    Returns:
        Tuple of (forecast_series, confidence_intervals_df)
    """
    # Create future date index
    last_date = ts_series.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    conf_int_df: Optional[pd.DataFrame] = None

    if model_type.upper().startswith("ARIMA"):
        fit = fit_arima(ts_series)
        fc_obj = fit.get_forecast(steps=horizon_days)
        pred_mean = fc_obj.predicted_mean
        pred_mean.index = future_dates
        pred_mean = np.maximum(pred_mean, 0)

        conf_int = fc_obj.conf_int(alpha=0.05)
        conf_int.index = future_dates
        conf_int_df = pd.DataFrame({
            "LowerBound": np.maximum(conf_int.iloc[:, 0], 0),
            "UpperBound": np.maximum(conf_int.iloc[:, 1], 0)
        }, index=future_dates)
        return pred_mean, conf_int_df

    else:
        fit = fit_exponential_smoothing(ts_series)
        pred = fit.forecast(steps=horizon_days)
        pred.index = future_dates
        pred = np.maximum(pred, 0)
        # Approximate standard error bounds based on residual variance
        residuals = fit.resid
        std_err = float(np.std(residuals)) if len(residuals) > 0 else 0.0
        conf_int_df = pd.DataFrame({
            "LowerBound": np.maximum(pred - 1.96 * std_err, 0),
            "UpperBound": np.maximum(pred + 1.96 * std_err, 0)
        }, index=future_dates)
        return pred, conf_int_df
