"""
Unit and integration tests for data loading and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import validate_columns
from src.preprocessing import clean_retail_data


@pytest.fixture
def sample_raw_data():
    """Generates synthetic transactions covering all edge cases to test pipeline integrity."""
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "C536366", "536367", "536368", "536369"],
        "StockCode": ["85123A", "85123A", "85123A", "71053", "84406B", "POST"],
        "Description": ["WHITE HEART", "WHITE HEART", "WHITE HEART", "METAL", None, "Manual"],
        "Quantity": [6, 6, -1, 10, -5, 1],
        "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:26:00", "2010-12-01 08:28:00", 
                        "2010-12-01 08:34:00", "2010-12-01 08:40:00", "2010-12-01 09:00:00"],
        "UnitPrice": [2.55, 2.55, 2.55, 3.39, 0.0, 18.00],
        "CustomerID": [17850.0, 17850.0, 17850.0, 17850.0, None, None],
        "Country": ["United Kingdom"] * 6
    })


def test_validate_columns():
    valid_df = pd.DataFrame(columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    ])
    is_valid, missing = validate_columns(valid_df)
    assert is_valid is True
    assert len(missing) == 0

    invalid_df = pd.DataFrame(columns=["InvoiceNo", "Quantity"])
    is_valid, missing = validate_columns(invalid_df)
    assert is_valid is False
    assert "UnitPrice" in missing


def test_clean_retail_data(sample_raw_data):
    views, metrics = clean_retail_data(sample_raw_data, save_processed=False)

    # 1. Duplicates: Row 0 and Row 1 are exact duplicates; one should be dropped
    assert metrics["duplicate_count"] == 1
    assert metrics["initial_rows"] == 6
    assert metrics["deduplicated_rows"] == 5

    # 2. Cancellations: Row 2 has 'C536366'
    assert metrics["cancellations_count"] == 1
    assert any(views["returns"]["InvoiceNo"] == "C536366")

    # 3. Clean sales view should only contain valid positive sales with identified CustomerID
    clean_sales = views["clean_sales"]
    assert len(clean_sales) == 2  # Row 0 (deduped), Row 3
    assert (clean_sales["Quantity"] > 0).all()
    assert (clean_sales["UnitPrice"] > 0).all()
    assert clean_sales["CustomerID"].notnull().all()

    # 4. Revenue calculation check
    expected_rev = 6 * 2.55
    assert clean_sales.iloc[0]["Revenue"] == pytest.approx(expected_rev)


def test_parse_uploaded_file_csv():
    import io
    from src.data_loader import parse_uploaded_file

    csv_content = b"InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n1001,ITEM1,Desc,2,2011-01-01,15.0,12345,UK\n"
    file_obj = io.BytesIO(csv_content)
    file_obj.name = "customer_test.csv"

    df, summary = parse_uploaded_file(file_obj)
    assert len(df) == 1
    assert "customer_test.csv" in summary
    assert "InvoiceNo" in df.columns
    assert "UnitPrice" in df.columns


def test_parse_uploaded_file_alias_normalization():
    import io
    from src.data_loader import parse_uploaded_file

    # Test with lowercase / spaced column headers from real customer files
    csv_content = b"invoice_no,sku,item_description,qty,order_date,price,customer_id,market\n1002,ITEM2,Desc2,4,2011-01-02,25.0,12346,USA\n"
    file_obj = io.BytesIO(csv_content)
    file_obj.name = "custom_orders.csv"

    df, summary = parse_uploaded_file(file_obj)
    assert len(df) == 1
    assert "InvoiceNo" in df.columns
    assert "StockCode" in df.columns
    assert "Quantity" in df.columns
    assert "UnitPrice" in df.columns
    assert "CustomerID" in df.columns

