"""
Unit tests for src.data_loader
Validates column normalization, format parsing, and uploaded file inspection.
"""

import io
import pandas as pd
import pytest
from src.data_loader import (
    normalize_columns,
    validate_columns,
    parse_uploaded_file,
    inspect_and_parse_uploaded_file,
    COLUMN_ALIASES
)


class MockUploadedFile:
    def __init__(self, name: str, data_bytes: bytes):
        self.name = name
        self._bio = io.BytesIO(data_bytes)
        self.size = len(data_bytes)

    def read(self, *args, **kwargs):
        return self._bio.read(*args, **kwargs)

    def seek(self, offset, whence=0):
        return self._bio.seek(offset, whence)

    def getvalue(self):
        return self._bio.getvalue()


def test_normalize_columns():
    df = pd.DataFrame(columns=["Order ID", "item_code", "QTY", "Item Price", "Client ID", "Date", "Market", "Title"])
    normalized = normalize_columns(df)
    expected = ["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID", "InvoiceDate", "Country", "Description"]
    assert list(normalized.columns) == expected


def test_validate_columns():
    valid_df = pd.DataFrame(columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    is_valid, missing = validate_columns(valid_df)
    assert is_valid is True
    assert len(missing) == 0

    invalid_df = pd.DataFrame(columns=["InvoiceNo", "Quantity"])
    is_valid, missing = validate_columns(invalid_df)
    assert is_valid is False
    assert "StockCode" in missing


def test_inspect_and_parse_csv():
    csv_content = (
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
        "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01 08:26:00,2.55,17850,United Kingdom\n"
        "C536366,71053,WHITE METAL LANTERN,-2,2010-12-01 08:28:00,3.39,17850,United Kingdom\n"
    ).encode("utf-8")
    
    mock_file = MockUploadedFile("sample_transactions.csv", csv_content)
    df, meta = inspect_and_parse_uploaded_file(mock_file)

    assert len(df) == 2
    assert meta["total_rows"] == 2
    assert meta["total_columns"] == 8
    assert meta["is_valid"] is True
    assert meta["duplicate_rows"] == 0
