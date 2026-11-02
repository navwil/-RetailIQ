"""
RetailIQ - Configuration Module
Centralized settings for data paths, model parameters, and UI defaults.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure runtime directories exist
for path in [PROCESSED_DATA_DIR, MODELS_DIR, CHARTS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset Paths (Configurable)
EXCEL_DATA_PATH = BASE_DIR / "Online Retail.xlsx"
CSV_DATA_PATH = BASE_DIR / "OnlineRetail.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "online_retail_clean.parquet"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "online_retail_clean.csv"

# Model Paths
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_rfm_model.joblib"
SCALER_PATH = MODELS_DIR / "rfm_scaler.joblib"
SEGMENTATION_METADATA_PATH = MODELS_DIR / "segmentation_metadata.joblib"
FORECAST_METRICS_PATH = MODELS_DIR / "forecast_metrics.joblib"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Preprocessing Constants
REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Country"
]

# Customer Segmentation (RFM) Constants
RFM_DEFAULT_CLUSTERS = 4
RFM_CLUSTER_RANGE = range(2, 9)
RFM_SEGMENT_NAMES = {
    "Champions": "High monetary, high frequency, recent purchase (VIP)",
    "Loyal Customers": "Consistent purchases, good monetary contribution",
    "Potential Loyalists": "Recent buyers with average frequency and spend",
    "New Customers": "Recent buyers but low frequency to date",
    "At-Risk Customers": "Previously frequent/high-spenders, long time since last purchase",
    "Hibernating Customers": "Low spend, low frequency, long time inactive"
}

# Time-Series Forecasting Constants
FORECAST_HORIZONS = [7, 30, 90]
DEFAULT_HORIZON = 30
TEST_SIZE_DAYS = 30

# UI Branding & Styling
APP_TITLE = "RetailIQ"
APP_SUBTITLE = "Online Retail Analytics & Intelligence System"
APP_VERSION = "2.0.0"
COMPANY_NAME = "RetailIQ Enterprise"
THEME_PRIMARY = "#4F46E5"
THEME_ACCENT = "#06B6D4"
THEME_SUCCESS = "#10B981"
THEME_WARNING = "#F59E0B"
THEME_DANGER = "#EF4444"
