"""
Unit tests for RFM calculation and K-Means customer segmentation.
"""

import pytest
import pandas as pd
import numpy as np
from src.customer_segmentation import (
    compute_rfm_table,
    evaluate_cluster_candidates,
    train_kmeans_segmentation
)


@pytest.fixture
def mock_sales_transactions():
    """Generates synthetic sales data for multiple customers across dates."""
    np.random.seed(42)
    cust_ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    records = []
    base_date = pd.Timestamp("2011-01-01")

    for i in range(100):
        c_id = np.random.choice(cust_ids)
        days_offset = np.random.randint(0, 300)
        inv_date = base_date + pd.Timedelta(days=days_offset)
        qty = np.random.randint(1, 20)
        price = np.random.uniform(2.0, 50.0)
        records.append({
            "InvoiceNo": f"INV{10000 + i}",
            "StockCode": f"PROD{np.random.randint(1, 10)}",
            "Description": "Test Product",
            "Quantity": qty,
            "InvoiceDate": inv_date,
            "UnitPrice": price,
            "CustomerID": str(c_id),
            "Country": "United Kingdom",
            "Revenue": qty * price
        })

    return pd.DataFrame(records)


def test_compute_rfm_table(mock_sales_transactions):
    rfm = compute_rfm_table(mock_sales_transactions)
    assert len(rfm) == 10
    assert set(rfm.columns) == {"CustomerID", "Recency", "Frequency", "Monetary"}
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()


def test_evaluate_cluster_candidates(mock_sales_transactions):
    rfm = compute_rfm_table(mock_sales_transactions)
    eval_res = evaluate_cluster_candidates(rfm, cluster_range=range(2, 5))
    assert "k_values" in eval_res
    assert len(eval_res["inertias"]) == 3
    assert len(eval_res["silhouette_scores"]) == 3
    assert eval_res["recommended_k"] in [2, 3, 4]


def test_train_kmeans_segmentation(mock_sales_transactions):
    rfm = compute_rfm_table(mock_sales_transactions)
    rfm_segmented, km, scaler, meta = train_kmeans_segmentation(
        rfm, n_clusters=3, save_artifacts=False
    )
    assert "Cluster" in rfm_segmented.columns
    assert "Segment" in rfm_segmented.columns
    assert rfm_segmented["Cluster"].nunique() == 3
    assert meta["n_clusters"] == 3
    assert -1.0 <= meta["silhouette_score"] <= 1.0
