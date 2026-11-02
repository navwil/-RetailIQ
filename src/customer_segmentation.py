"""
RetailIQ - Customer Segmentation Module
Computes RFM metrics, applies logarithmic skewness transformation, standardizes features,
determines optimal k via Elbow & Silhouette metrics, trains K-Means clustering,
and maps clusters to descriptive business segment personas.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from config import (
    KMEANS_MODEL_PATH,
    SCALER_PATH,
    SEGMENTATION_METADATA_PATH,
    RANDOM_STATE,
    RFM_DEFAULT_CLUSTERS,
    RFM_CLUSTER_RANGE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_rfm_table(df_clean_sales: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Computes Customer-level Recency, Frequency, and Monetary (RFM) metrics.
    
    Parameters:
        df_clean_sales: Clean sales DataFrame with CustomerID, InvoiceDate, InvoiceNo, Revenue.
        snapshot_date: Reference date for recency calculation (default is max date + 1 day).
        
    Returns:
        DataFrame indexed by CustomerID with columns: Recency, Frequency, Monetary.
    """
    if snapshot_date is None:
        snapshot_date = df_clean_sales["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df_clean_sales.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda d: (snapshot_date - d.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum")
        )
        .reset_index()
    )

    # Filter out non-positive monetary if any edge anomaly exists
    rfm = rfm[rfm["Monetary"] > 0].copy()
    rfm["CustomerID"] = rfm["CustomerID"].astype(str)
    return rfm


def evaluate_cluster_candidates(
    rfm_df: pd.DataFrame,
    cluster_range: range = RFM_CLUSTER_RANGE
) -> Dict[str, Any]:
    """
    Calculates Inertia (Elbow method) and Silhouette Scores across cluster numbers k.
    """
    features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
    # Log transform to alleviate heavy right-skew
    features_log = np.log1p(features)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_log)

    k_values = list(cluster_range)
    inertias = []
    silhouette_scores = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(scaled_data)
        inertias.append(float(km.inertia_))
        sil_score = float(silhouette_score(scaled_data, labels))
        silhouette_scores.append(sil_score)

    # Find highest silhouette k as recommended
    best_k = k_values[int(np.argmax(silhouette_scores))]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "recommended_k": best_k
    }


def _derive_segment_name(row_means: pd.Series, overall_means: pd.Series) -> str:
    """
    Data-driven heuristic to map cluster mean characteristics to marketing personas.
    Low Recency is positive (recent). High Frequency & Monetary are positive.
    """
    r_ratio = row_means["Recency"] / overall_means["Recency"]
    f_ratio = row_means["Frequency"] / overall_means["Frequency"]
    m_ratio = row_means["Monetary"] / overall_means["Monetary"]

    if r_ratio < 0.6 and f_ratio >= 1.5 and m_ratio >= 1.5:
        return "Champions"
    elif r_ratio < 0.8 and f_ratio >= 1.0 and m_ratio >= 1.0:
        return "Loyal Customers"
    elif r_ratio < 0.7 and f_ratio < 1.0:
        return "New / Promising Customers"
    elif r_ratio >= 1.2 and (f_ratio >= 1.0 or m_ratio >= 1.0):
        return "At-Risk Customers"
    elif r_ratio >= 1.2 and f_ratio < 1.0 and m_ratio < 1.0:
        return "Hibernating Customers"
    else:
        return "Potential Loyalists"


def train_kmeans_segmentation(
    rfm_df: pd.DataFrame,
    n_clusters: int = RFM_DEFAULT_CLUSTERS,
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, Dict[str, Any]]:
    """
    Trains K-Means model on log-transformed and standardized RFM features,
    assigns data-derived segment names, and persists artifacts.
    """
    rfm = rfm_df.copy()
    features = rfm[["Recency", "Frequency", "Monetary"]]
    features_log = np.log1p(features)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_log)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=15)
    clusters = kmeans.fit_predict(scaled_data)
    rfm["Cluster"] = clusters

    sil_score = float(silhouette_score(scaled_data, clusters))
    inertia = float(kmeans.inertia_)

    # Compute cluster profile and assign data-driven segment personas
    overall_means = rfm[["Recency", "Frequency", "Monetary"]].mean()
    cluster_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Create composite RFM value score: high F & M, low R
    # Normalize each column within cluster means to range [0, 1]
    norm_f = (cluster_means["Frequency"] - cluster_means["Frequency"].min()) / (cluster_means["Frequency"].max() - cluster_means["Frequency"].min() + 1e-6)
    norm_m = (cluster_means["Monetary"] - cluster_means["Monetary"].min()) / (cluster_means["Monetary"].max() - cluster_means["Monetary"].min() + 1e-6)
    norm_r = (cluster_means["Recency"].max() - cluster_means["Recency"]) / (cluster_means["Recency"].max() - cluster_means["Recency"].min() + 1e-6)
    
    composite_scores = (norm_r * 0.3) + (norm_f * 0.35) + (norm_m * 0.35)
    sorted_clusters_by_score = composite_scores.sort_values(ascending=False).index.tolist()

    # Standard segment names ordered from best to lowest engagement
    segment_hierarchy = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "At-Risk Customers",
        "Hibernating Customers",
        "New Customers"
    ]

    cluster_to_segment = {}
    for idx, c_id in enumerate(sorted_clusters_by_score):
        if idx < len(segment_hierarchy):
            cluster_to_segment[c_id] = segment_hierarchy[idx]
        else:
            cluster_to_segment[c_id] = f"Cluster {c_id}"

    rfm["Segment"] = rfm["Cluster"].map(cluster_to_segment)

    # Segment Profile Summary
    segment_summary = (
        rfm.groupby(["Cluster", "Segment"])
        .agg(
            CustomerCount=("CustomerID", "count"),
            TotalRevenue=("Monetary", "sum"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            MedianRecency=("Recency", "median"),
            MedianFrequency=("Frequency", "median"),
            MedianMonetary=("Monetary", "median")
        )
        .reset_index()
    )
    total_cust = len(rfm)
    total_rev = rfm["Monetary"].sum()
    segment_summary["CustomerShare_Pct"] = (segment_summary["CustomerCount"] / total_cust) * 100
    segment_summary["RevenueShare_Pct"] = (segment_summary["TotalRevenue"] / total_rev) * 100

    metadata = {
        "n_clusters": n_clusters,
        "silhouette_score": sil_score,
        "inertia": inertia,
        "total_customers": total_cust,
        "total_revenue": total_rev,
        "cluster_to_segment": cluster_to_segment,
        "segment_summary": segment_summary.to_dict(orient="records"),
        "overall_means": overall_means.to_dict()
    }

    if save_artifacts:
        joblib.dump(kmeans, KMEANS_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(metadata, SEGMENTATION_METADATA_PATH)
        logger.info(f"Persisted KMeans model to {KMEANS_MODEL_PATH}")

    return rfm, kmeans, scaler, metadata


def load_segmentation_artifacts() -> Tuple[Optional[KMeans], Optional[StandardScaler], Optional[Dict[str, Any]]]:
    """Loads saved segmentation model, scaler, and metadata if available."""
    if KMEANS_MODEL_PATH.exists() and SCALER_PATH.exists() and SEGMENTATION_METADATA_PATH.exists():
        km = joblib.load(KMEANS_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        meta = joblib.load(SEGMENTATION_METADATA_PATH)
        return km, scaler, meta
    return None, None, None
