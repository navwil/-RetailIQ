"""
RetailIQ - Exploratory Data Analysis (EDA) Module
Calculates core e-commerce metrics, aggregations, product performance,
and geographic distributions for enterprise analytics and reporting.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


def compute_kpis(df_sales: pd.DataFrame, df_returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Computes top-level executive KPI metrics from sales transactions.
    """
    total_rev = float(df_sales["Revenue"].sum())
    total_orders = int(df_sales["InvoiceNo"].nunique())
    unique_customers = int(df_sales["CustomerID"].nunique())
    unique_products = int(df_sales["StockCode"].nunique())
    total_units = int(df_sales["Quantity"].sum())
    avg_order_value = float(total_rev / total_orders) if total_orders > 0 else 0.0

    return_rate = 0.0
    returned_value = 0.0
    if df_returns is not None and len(df_returns) > 0:
        returned_value = float(abs(df_returns["Revenue"].sum()))
        total_gross = total_rev + returned_value
        return_rate = round((returned_value / total_gross) * 100, 2) if total_gross > 0 else 0.0

    return {
        "total_revenue": total_rev,
        "total_orders": total_orders,
        "unique_customers": unique_customers,
        "unique_products": unique_products,
        "total_units_sold": total_units,
        "avg_order_value": avg_order_value,
        "returned_value": returned_value,
        "return_rate_pct": return_rate
    }



def get_monthly_sales_summary(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sales by Year-Month with month-over-month revenue growth rates.
    """
    monthly = (
        df_sales.groupby("YearMonth")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Units=("Quantity", "sum"),
            ActiveCustomers=("CustomerID", "nunique")
        )
        .reset_index()
    )
    monthly = monthly.sort_values("YearMonth").reset_index(drop=True)
    monthly["MoM_Growth_Pct"] = monthly["Revenue"].pct_change() * 100
    monthly["AOV"] = monthly["Revenue"] / monthly["Orders"]
    return monthly


def get_daily_sales_series(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily sales revenue, orders, and units sold.
    """
    daily = (
        df_sales.groupby("Date")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Units=("Quantity", "sum")
        )
        .reset_index()
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)
    daily["7D_MA"] = daily["Revenue"].rolling(window=7, min_periods=1).mean()
    daily["30D_MA"] = daily["Revenue"].rolling(window=30, min_periods=1).mean()
    return daily


def get_top_products(df_sales: pd.DataFrame, top_n: int = 10, by: str = "Revenue") -> pd.DataFrame:
    """
    Retrieves top selling products ranked by Revenue or Quantity.
    """
    prod_summary = (
        df_sales.groupby(["StockCode", "Description"])
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
            AvgPrice=("UnitPrice", "mean")
        )
        .reset_index()
    )
    ranked = prod_summary.sort_values(by=by, ascending=False).head(top_n).reset_index(drop=True)
    ranked["RevenueShare_Pct"] = (ranked["Revenue"] / df_sales["Revenue"].sum()) * 100
    return ranked


def get_country_sales_summary(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates geographic sales performance by country.
    """
    country_summary = (
        df_sales.groupby("Country")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
            Units=("Quantity", "sum")
        )
        .reset_index()
    )
    total_rev = country_summary["Revenue"].sum()
    country_summary["RevenueShare_Pct"] = (country_summary["Revenue"] / total_rev) * 100
    country_summary = country_summary.sort_values("Revenue", ascending=False).reset_index(drop=True)
    return country_summary


def get_hourly_and_weekday_patterns(df_sales: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes temporal purchasing patterns by Day of Week and Hour of Day.
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_summary = (
        df_sales.groupby("DayOfWeek")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique")
        )
        .reindex(day_order)
        .dropna(how="all")
        .reset_index()
    )

    hourly_summary = (
        df_sales.groupby("Hour")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique")
        )
        .reset_index()
        .sort_values("Hour")
    )

    return weekday_summary, hourly_summary
