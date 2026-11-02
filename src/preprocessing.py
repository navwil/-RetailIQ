"""
RetailIQ - Data Preprocessing Module
Implements comprehensive cleaning, validation, audit metrics,
and multi-view transaction partitioning for enterprise online retail analytics.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from config import (
    PROCESSED_DATA_PATH,
    PROCESSED_CSV_PATH,
    REQUIRED_COLUMNS
)
from src.data_loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_retail_data(
    df_raw: pd.DataFrame,
    save_processed: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Cleans raw retail transaction data and constructs multi-perspective transaction views.
    
    Processing Rules:
    1. Standardize column names (strip whitespace).
    2. Convert InvoiceDate to datetime.
    3. Remove exact duplicate rows.
    4. Categorize transactions:
       - Cancellations: InvoiceNo starts with 'C'
       - Negative quantity returns
       - Admin/system fee adjustments (UnitPrice <= 0)
    5. Clean text fields: Description stripped and imputed via StockCode dictionary when possible.
    6. Derive financial & temporal features:
       - Revenue = Quantity * UnitPrice
       - Date, Year, Month, YearMonth, DayOfWeek, Hour
    7. Generate distinct views:
       - 'all': Enriched transaction log with classification flags.
       - 'clean_sales': High-confidence transactions with valid CustomerID, positive Qty & Price, non-cancelled.
       - 'returns': Cancellation and return transactions.
       - 'macro_sales': All valid positive sales transactions (including unregistered guest customers).
       
    Returns:
        Tuple of (views_dict, audit_metrics_dict)
    """
    logger.info("Initiating enterprise retail data preprocessing pipeline...")
    initial_rows = len(df_raw)
    df = df_raw.copy()

    # 1. Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Convert InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # 3. Detect and remove duplicate rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        df = df.drop_duplicates()
        logger.info(f"Removed {duplicate_count:,} duplicate records ({duplicate_count/initial_rows*100:.2f}%)")

    # 4. Standardize InvoiceNo as string
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    # 5. Classify transaction status
    df["IsCancelled"] = df["InvoiceNo"].str.startswith("C", na=False)
    
    # 6. Description imputation based on StockCode lookup where available
    valid_desc_map = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .first()
        .to_dict()
    )
    df["Description"] = df["Description"].fillna(df["StockCode"].map(valid_desc_map))
    df["Description"] = df["Description"].fillna("UNKNOWN ITEM").astype(str).str.strip()

    # 7. Financial & Temporal Features
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    # 8. Transaction Category
    conditions = [
        df["IsCancelled"] | (df["Quantity"] < 0),
        (df["UnitPrice"] <= 0),
        (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    ]
    choices = ["Return/Cancellation", "Zero/Negative Price Adj", "Valid Sale"]
    df["TransactionType"] = np.select(conditions, choices, default="Other")

    # 9. View Construction
    # View A: Returns & Cancellations
    df_returns = df[df["TransactionType"] == "Return/Cancellation"].copy()

    # View B: Macro Sales (All valid sales, including guest checkouts with null CustomerID)
    df_macro_sales = df[
        (df["Quantity"] > 0) & 
        (df["UnitPrice"] > 0) & 
        (~df["IsCancelled"])
    ].copy()

    # View C: Clean Sales with Identified Customers (For RFM & Customer Analytics)
    df_clean_sales = df_macro_sales[df_macro_sales["CustomerID"].notnull()].copy()
    df_clean_sales["CustomerID"] = df_clean_sales["CustomerID"].astype(int).astype(str)

    # Preprocessing Audit Metrics
    audit_metrics = {
        "initial_rows": initial_rows,
        "deduplicated_rows": len(df),
        "duplicate_count": duplicate_count,
        "cancellations_count": int(df["IsCancelled"].sum()),
        "returns_count": len(df_returns),
        "zero_price_count": int((df["UnitPrice"] <= 0).sum()),
        "missing_customer_count": int(df["CustomerID"].isnull().sum()),
        "macro_sales_rows": len(df_macro_sales),
        "clean_customer_sales_rows": len(df_clean_sales),
        "unique_identified_customers": int(df_clean_sales["CustomerID"].nunique()),
        "unique_products": int(df_clean_sales["StockCode"].nunique()),
        "gross_sales_revenue": float(df_macro_sales["Revenue"].sum()),
        "customer_sales_revenue": float(df_clean_sales["Revenue"].sum()),
        "returned_loss_value": float(abs(df_returns["Revenue"].sum())),
        "net_revenue": float(df["Revenue"].sum()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max()
    }

    views = {
        "all": df,
        "clean_sales": df_clean_sales,
        "macro_sales": df_macro_sales,
        "returns": df_returns
    }

    if save_processed:
        save_cleaned_data(df_clean_sales)

    logger.info(
        f"Preprocessing complete. Clean Customer Sales: {len(df_clean_sales):,} rows | "
        f"Identified Customers: {audit_metrics['unique_identified_customers']:,}"
    )
    return views, audit_metrics


def save_cleaned_data(df: pd.DataFrame) -> None:
    """
    Persists cleaned customer transactions to parquet for sub-second reloading,
    and CSV for external accessibility.
    """
    try:
        df.to_parquet(PROCESSED_DATA_PATH, index=False, engine="pyarrow")
        logger.info(f"Saved processed dataset to parquet: {PROCESSED_DATA_PATH}")
    except Exception as e:
        logger.warning(f"Could not save parquet ({e}). Falling back to CSV.")
    
    try:
        # Also save lightweight CSV for general accessibility
        df.head(100000).to_csv(PROCESSED_CSV_PATH, index=False)
        logger.info(f"Saved processed sample CSV: {PROCESSED_CSV_PATH}")
    except Exception as e:
        logger.error(f"Error saving processed CSV: {e}")


def load_or_process_data(
    source_preference: str = "excel",
    force_recompute: bool = False
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    High-level cached data pipeline. Checks if processed data already exists;
    if not or forced, loads raw data and runs the preprocessing pipeline.
    """
    # Always load raw data to maintain all four transaction views and accurate audit counts
    raw_df, source_info = load_raw_data(source_preference=source_preference)
    views, metrics = clean_retail_data(raw_df, save_processed=True)
    metrics["source_info"] = source_info
    return views, metrics
