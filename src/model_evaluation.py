"""
RetailIQ - Model Evaluation Module
Calculates formal statistical and machine learning evaluation metrics
for customer clustering and time-series sales forecasting.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    davies_bouldin_score
)


def compute_forecasting_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Computes regression & forecasting metrics: MAE, RMSE, MAPE, and R-squared.
    Safely handles division by zero or near-zero actual values for MAPE.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Safe MAPE: exclude points where actual is 0 or extremely close to 0
    non_zero_mask = np.abs(y_t) > 1e-3
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100)
    else:
        mape = 0.0

    # Symmetric MAPE (sMAPE) as robust secondary metric
    denominator = (np.abs(y_t) + np.abs(y_p)) / 2.0
    smape_mask = denominator > 1e-3
    if np.sum(smape_mask) > 0:
        smape = float(np.mean(np.abs(y_p[smape_mask] - y_t[smape_mask]) / denominator[smape_mask]) * 100)
    else:
        smape = 0.0

    try:
        r2 = float(r2_score(y_t, y_p))
    except Exception:
        r2 = 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_Pct": round(mape, 2),
        "sMAPE_Pct": round(smape, 2),
        "R2": round(r2, 4)
    }


def compute_clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    inertia: float
) -> Dict[str, float]:
    """
    Computes formal unsupervised clustering metrics:
    Silhouette Score, Davies-Bouldin Index, and Inertia.
    """
    sil = float(silhouette_score(features, labels))
    db_index = float(davies_bouldin_score(features, labels))

    return {
        "SilhouetteScore": round(sil, 4),
        "DaviesBouldinIndex": round(db_index, 4),
        "Inertia": round(float(inertia), 2)
    }
