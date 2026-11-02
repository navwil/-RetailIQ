"""
RetailIQ - Data Loader Module
Handles loading and structural validation of raw datasets from Excel, CSV, and customer uploads (CSV/Excel/PDF).
Strictly adheres to real data loading; never fabricates mock data.
"""

import io
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
from pypdf import PdfReader

from config import (
    EXCEL_DATA_PATH,
    CSV_DATA_PATH,
    REQUIRED_COLUMNS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Intelligent column alias mapping for customer uploaded files
COLUMN_ALIASES = {
    # InvoiceNo
    "invoiceno": "InvoiceNo", "invoicenum": "InvoiceNo", "invoicenumber": "InvoiceNo",
    "invoice": "InvoiceNo", "orderid": "InvoiceNo", "orderno": "InvoiceNo", "ordernumber": "InvoiceNo",
    # StockCode
    "stockcode": "StockCode", "itemcode": "StockCode", "item": "StockCode", "sku": "StockCode",
    "productid": "StockCode", "productcode": "StockCode", "partnumber": "StockCode",
    # Description
    "description": "Description", "desc": "Description", "itemdescription": "Description",
    "productname": "Description", "product": "Description", "title": "Description", "itemname": "Description",
    # Quantity
    "quantity": "Quantity", "qty": "Quantity", "units": "Quantity", "volume": "Quantity", "count": "Quantity",
    # InvoiceDate
    "invoicedate": "InvoiceDate", "date": "InvoiceDate", "timestamp": "InvoiceDate",
    "orderdate": "InvoiceDate", "purchasedate": "InvoiceDate", "datetime": "InvoiceDate",
    # UnitPrice
    "unitprice": "UnitPrice", "price": "UnitPrice", "itemprice": "UnitPrice", "rate": "UnitPrice",
    "cost": "UnitPrice", "unitcost": "UnitPrice", "amount": "UnitPrice",
    # CustomerID
    "customerid": "CustomerID", "customer": "CustomerID", "clientid": "CustomerID",
    "userid": "CustomerID", "accountid": "CustomerID", "buyerid": "CustomerID",
    # Country
    "country": "Country", "region": "Country", "nation": "Country", "market": "Country", "location": "Country"
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names using intelligent alias mapping.
    """
    df = df.copy()
    new_cols = []
    for c in df.columns:
        clean = re.sub(r"[^a-zA-Z0-9]", "", str(c)).lower()
        new_cols.append(COLUMN_ALIASES.get(clean, str(c).strip()))
    df.columns = new_cols
    return df


def validate_columns(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Checks if all required columns exist in the DataFrame.
    """
    df_cols = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df_cols]
    return (len(missing) == 0, missing)


def parse_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Parses customer-uploaded file in CSV, Excel (.xlsx/.xls), or PDF format.
    
    Returns:
        Tuple of (pd.DataFrame, file_summary_str)
    """
    filename = uploaded_file.name
    file_ext = Path(filename).suffix.lower()

    if file_ext == ".csv":
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file)

    elif file_ext == ".pdf":
        reader = PdfReader(uploaded_file)
        all_lines = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_lines.extend(text.splitlines())

        # Attempt tabular extraction
        rows = []
        header = None
        for line in all_lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Split by comma or tab if present, else by 2+ spaces
            if "," in line_str:
                parts = [p.strip() for p in line_str.split(",")]
            elif "\t" in line_str:
                parts = [p.strip() for p in line_str.split("\t")]
            else:
                parts = [p.strip() for p in re.split(r"\s{2,}", line_str)]

            if len(parts) >= 6:
                if header is None:
                    header = parts
                else:
                    rows.append(parts)

        if not rows or header is None:
            raise ValueError(
                f"Could not automatically detect tabular transaction data in '{filename}'. "
                f"Please ensure the PDF contains a structured table with InvoiceNo, StockCode, Quantity, etc., or upload CSV/Excel."
            )
        # Pad or truncate rows to header length
        cleaned_rows = []
        for r in rows:
            if len(r) == len(header):
                cleaned_rows.append(r)
            elif len(r) > len(header):
                cleaned_rows.append(r[:len(header)])
            else:
                cleaned_rows.append(r + [None] * (len(header) - len(r)))

        df = pd.DataFrame(cleaned_rows, columns=header)

    else:
        raise ValueError(f"Unsupported file format '{file_ext}'. Please upload a .csv, .xlsx, or .pdf file.")

    # Standardize column aliases
    df = normalize_columns(df)
    valid, missing = validate_columns(df)
    if not valid:
        raise ValueError(
            f"Uploaded file '{filename}' is missing required retail columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    summary_str = f"Uploaded: {filename} ({len(df):,} records)"
    return df, summary_str


def inspect_and_parse_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses and deeply inspects a customer-uploaded file (CSV, Excel, or PDF).
    Returns the parsed DataFrame and a rich verification diagnostics dictionary.
    """
    filename = uploaded_file.name
    file_ext = Path(filename).suffix.lower()
    file_size_bytes = getattr(uploaded_file, "size", 0)

    # 1. Parse raw
    if file_ext == ".csv":
        try:
            raw_df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    elif file_ext in [".xlsx", ".xls"]:
        raw_df = pd.read_excel(uploaded_file)
    elif file_ext == ".pdf":
        reader = PdfReader(uploaded_file)
        all_lines = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_lines.extend(text.splitlines())

        rows = []
        header = None
        for line in all_lines:
            line_str = line.strip()
            if not line_str:
                continue
            if "," in line_str:
                parts = [p.strip() for p in line_str.split(",")]
            elif "\t" in line_str:
                parts = [p.strip() for p in line_str.split("\t")]
            else:
                parts = [p.strip() for p in re.split(r"\s{2,}", line_str)]

            if len(parts) >= 6:
                if header is None:
                    header = parts
                else:
                    rows.append(parts)

        if not rows or header is None:
            raise ValueError(
                f"Could not automatically detect tabular transaction data in '{filename}'. "
                f"Please ensure the PDF contains a structured table with InvoiceNo, StockCode, Quantity, etc."
            )
        cleaned_rows = []
        for r in rows:
            if len(r) == len(header):
                cleaned_rows.append(r)
            elif len(r) > len(header):
                cleaned_rows.append(r[:len(header)])
            else:
                cleaned_rows.append(r + [None] * (len(header) - len(r)))

        raw_df = pd.DataFrame(cleaned_rows, columns=header)
    else:
        raise ValueError(f"Unsupported file format '{file_ext}'. Supported: CSV, Excel, PDF.")

    original_cols = list(raw_df.columns)
    
    # 2. Normalize columns
    df = normalize_columns(raw_df)
    valid, missing = validate_columns(df)

    # Track column mapping
    mapping_records = []
    for orig, norm in zip(original_cols, df.columns):
        mapping_records.append({
            "Original File Header": orig,
            "Mapped Platform Attribute": norm,
            "Status": "✅ Mapped" if norm in REQUIRED_COLUMNS else "ℹ️ Preserved"
        })

    # Basic stats
    n_rows = len(df)
    diagnostics = {
        "filename": filename,
        "format": file_ext.upper().replace(".", ""),
        "size_bytes": file_size_bytes,
        "total_rows": n_rows,
        "total_columns": len(df.columns),
        "column_mapping": mapping_records,
        "is_valid": valid,
        "missing_required": missing,
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }

    if not valid:
        diagnostics["error_message"] = (
            f"Missing required columns: {missing}. Expected columns: {REQUIRED_COLUMNS}"
        )

    return df, diagnostics


def load_raw_data(
    source_preference: str = "excel",
    custom_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Loads raw transaction data from the preferred source or specified path.
    """
    selected_path: Optional[Path] = None

    if custom_path:
        selected_path = Path(custom_path)
        if not selected_path.exists():
            raise FileNotFoundError(f"Custom dataset file not found at: {selected_path.resolve()}")
    else:
        # Default priority: check CSV first for fast loads if available, or excel
        if source_preference.lower() == "csv" and CSV_DATA_PATH.exists():
            selected_path = CSV_DATA_PATH
        elif EXCEL_DATA_PATH.exists():
            selected_path = EXCEL_DATA_PATH
        elif CSV_DATA_PATH.exists():
            selected_path = CSV_DATA_PATH
        else:
            raise FileNotFoundError(
                f"Neither Excel ({EXCEL_DATA_PATH}) nor CSV ({CSV_DATA_PATH}) could be located."
            )

    logger.info(f"Loading data from: {selected_path.name}")

    if selected_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(selected_path)
    elif selected_path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(selected_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(selected_path, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unsupported file format: {selected_path.suffix}")

    df = normalize_columns(df)
    valid, missing_cols = validate_columns(df)
    if not valid:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    source_info = f"{selected_path.name} ({len(df):,} rows)"
    return df, source_info


def get_raw_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts summary statistics and data hygiene diagnostics from the raw dataset.
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / n_rows * 100).round(2).to_dict()
    
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C", na=False)
    n_cancels = int(is_cancel.sum())
    
    n_neg_qty = int((pd.to_numeric(df["Quantity"], errors="coerce") < 0).sum())
    n_zero_or_neg_price = int((pd.to_numeric(df["UnitPrice"], errors="coerce") <= 0).sum())
    n_duplicates = int(df.duplicated().sum())
    
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    min_date = dates.min()
    max_date = dates.max()

    return {
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "cancellations_count": n_cancels,
        "negative_quantity_count": n_neg_qty,
        "non_positive_price_count": n_zero_or_neg_price,
        "duplicate_count": n_duplicates,
        "duplicate_pct": round(n_duplicates / n_rows * 100, 2),
        "date_min": min_date,
        "date_max": max_date,
        "unique_customers": int(df["CustomerID"].dropna().nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "unique_countries": int(df["Country"].nunique())
    }
