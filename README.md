# RetailIQ – Online Retail Analytics & Intelligence System

**RetailIQ** is an enterprise-grade, end-to-end Data Science and Machine Learning web application designed for comprehensive e-commerce transaction intelligence. Built on real transactional data from the classic UCI Online Retail dataset, RetailIQ bridges the gap between retrospective business intelligence dashboards and predictive algorithmic modeling.

---

## 1. Project Overview & System Capabilities

RetailIQ ingests multi-million pound retail transaction logs, executes an audited data hygiene pipeline, and delivers two primary analytical capabilities:
1. **Unsupervised Customer Segmentation**: Formulates behavioral Recency, Frequency, and Monetary (RFM) metrics, corrects severe positive skewness through logarithmic transformations, and identifies optimal customer cohorts using K-Means clustering evaluated via Elbow Inertia and Silhouette analysis.
2. **Chronological Time-Series Sales Forecasting**: Transforms discrete sales logs into continuous regular time-series to project future demand (7, 30, and 90-day horizons) using statistical ARIMA/SARIMAX and Holt-Winters Exponential Smoothing models, strictly validated via out-of-sample chronological splits to prevent lookahead data leakage.

The system is delivered as a modern, high-contrast SaaS-style interactive web application built with **Streamlit**, featuring interactive **Plotly** visualizations, automated **ReportLab** PDF generation, and tabular data exports.

---

## 2. Problem Statement

Retail organizations processing hundreds of thousands of transactions confront two critical operational bottlenecks:
* **Customer Heterogeneity**: Treating all customers identically leads to wasted marketing capital, blanket discounting that erodes gross margins, and customer attrition. Businesses need to isolate high-value VIP buyers from lapsing customers.
* **Demand Uncertainty**: Inaccurate sales forecasting causes inventory stockouts or capital lockup in unsold inventory. Demand models must respect weekly retail cyclicality while accounting for holiday demand surges.
* **Noisy Transaction Records**: Raw transactional systems log cancellations, refunds, damages, and missing customer identities that distort analytics if not handled properly.

---

## 3. Core Objectives

* Build a reusable data preprocessing module that systematically isolates genuine sales from cancellations and returns.
* Implement RFM modeling and K-Means clustering to discover distinct, interpretable marketing personas.
* Implement statistical time-series forecasting (ARIMA and Holt-Winters) with rigorous chronological holdout validation.
* Develop a modular, responsive 9-page Streamlit application with custom dark SaaS aesthetics.
* Provide downloadable publication-ready academic reports (PDF), executive briefs, and slide presentations (PPTX).
* Maintain complete reproducibility with zero mock data generation.

---

## 4. Key Features

* **Multi-View Data Architecture**: Separates clean sales (`df_clean_sales`), returns/cancellations (`df_returns`), and macro store sales (`df_macro_sales`).
* **Pareto Skewness Correction**: Applies $\tilde{x} = \ln(1 + x)$ transformations to RFM features prior to Euclidean distance calculations in K-Means.
* **Algorithmic Persona Mapping**: Derives human-interpretable segment labels (*Champions*, *Loyal Customers*, *Potential Loyalists*, *At-Risk Customers*, *Hibernating*) dynamically from cluster centroids.
* **95% Confidence Bounds**: Produces statistical confidence intervals for future sales forecasts.
* **Interactive Customer Drilldown**: Look up any customer ID to view their lifetime purchase history and RFM metrics.
* **Automated PDF & PPTX Generation**: Compiles a 10-chapter academic report and presentation slides with actual experimental results.

---

## 5. Technology Stack

* **Programming Language**: Python 3.10+ (tested on Python 3.13)
* **Application Framework**: Streamlit (v1.35+)
* **Data Processing & Manipulation**: Pandas, NumPy, PyArrow
* **Machine Learning**: Scikit-Learn (KMeans, StandardScaler, Metrics)
* **Statistical Modeling & Forecasting**: Statsmodels (ARIMA, SARIMAX, Holt-Winters Exponential Smoothing)
* **Interactive Visualizations**: Plotly Express & Plotly Graph Objects
* **Document & Slide Generation**: ReportLab (PDF), Python-PPTX (PowerPoint)
* **Testing & Quality Assurance**: Pytest

---

## 6. Dataset Description & Hygiene

The system is built upon the UCI Online Retail transaction dataset:
* **Transaction Records**: 541,909 rows × 8 attributes
* **Date Span**: December 1, 2010 to December 9, 2011 (374 calendar days)
* **Product Catalog**: 4,070 unique inventory SKUs
* **Customer Accounts**: 4,372 distinct identified customers
* **Geographic Coverage**: 38 global markets (dominated by the United Kingdom)

### Transaction Attributes
| Column Name | Type | Description & Preprocessing Treatment |
| :--- | :--- | :--- |
| `InvoiceNo` | String | 6-digit invoice code. Cancellations start with `'C'` (9,288 records) and are isolated into returns. |
| `StockCode` | String | Unique product code. |
| `Description` | String | Product name. 1,454 missing descriptions imputed via historical StockCode lookup. |
| `Quantity` | Integer | Units purchased. 10,624 negative values isolated as cancellations or inventory adjustments. |
| `InvoiceDate` | Datetime | Timestamp of transaction. Converted to datetime for time-series aggregation. |
| `UnitPrice` | Float | Unit price in GBP (£). 2,517 non-positive prices filtered from sales views. |
| `CustomerID` | Float/Str | Customer identifier. Missing in 135,080 rows (24.9% guest checkouts); excluded from RFM. |
| `Country` | String | Destination country of purchase. |

---

## 7. System Architecture

```
RetailIQ Architecture
├── Data Ingestion Layer: Dual Excel / CSV reader with column validation
├── Preprocessing Pipeline: Deduplication, cancellation tagging, revenue derivation
├── Analytical Core:
│   ├── RFM Engine: Log-transformation, scaling, KMeans clustering
│   └── Forecasting Engine: Daily resampling, chronological split, ARIMA & HW
├── Application Presentation: Streamlit multi-page dashboard with custom CSS
└── Reporting Engine: Automated ReportLab PDF and python-pptx generation
```

---

## 8. Project Folder Structure

```
RetailIQ/
│
├── app.py                      # Main application entrypoint and system overview
├── config.py                   # Centralized configuration, paths, and constants
├── requirements.txt            # Project dependencies
├── README.md                   # Complete system documentation
├── .gitignore                  # Git exclusions
│
├── data/
│   ├── raw/                    # Raw dataset directory
│   └── processed/              # Cleaned Parquet & CSV cached files
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── data_loader.py          # Data ingestion and column validation
│   ├── preprocessing.py        # Hygiene pipeline and view partitioning
│   ├── eda.py                  # KPI calculations and commercial aggregations
│   ├── customer_segmentation.py# RFM analysis and KMeans clustering
│   ├── sales_forecasting.py    # Time-series models (ARIMA & Exponential Smoothing)
│   ├── model_evaluation.py     # Evaluation metrics (MAE, RMSE, MAPE, Silhouette)
│   ├── report_generator.py     # PDF report compiler (ReportLab)
│   └── utils.py                # SaaS CSS styling, Plotly themes, and caching
│
├── pages/
│   ├── 1_Overview.py           # Page 1: Executive KPI overview & trends
│   ├── 2_Sales_Analytics.py    # Page 2: Granular sales & growth rates
│   ├── 3_Customer_Segmentation.py # Page 3: RFM clusters & personas
│   ├── 4_Sales_Forecasting.py  # Page 4: ARIMA & Holt-Winters forecasting
│   ├── 5_Product_Analytics.py  # Page 5: SKU-level velocity & catalog
│   ├── 6_Customer_Insights.py  # Page 6: Individual customer audit & history
│   ├── 7_Data_Explorer.py      # Page 7: Data hygiene audit & raw exports
│   ├── 8_Model_Performance.py  # Page 8: Algorithmic metrics & Elbow curves
│   └── 9_Reports.py            # Page 9: PDF & CSV downloadable reports
│
├── models/                     # Serialized models (.joblib)
│   ├── kmeans_rfm_model.joblib
│   ├── rfm_scaler.joblib
│   └── segmentation_metadata.joblib
│
├── outputs/
│   ├── charts/                 # Visual artifacts
│   └── reports/                # Generated PDF reports & PPTX slides
│       ├── RetailIQ_Academic_Project_Report.pdf
│       └── RetailIQ_Project_Presentation.pptx
│
├── notebooks/
│   └── Online_Retail_Analysis.ipynb # Complete 11-step Jupyter Notebook
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # Hygiene and partition unit tests
    ├── test_segmentation.py    # RFM and clustering unit tests
    └── test_forecasting.py     # Time-series and model evaluation tests
```

---

## 9. Installation & Windows Setup Instructions

### Step 1: Clone or Open the Workspace
Open a terminal in the project root directory:
```powershell
cd "E:\ml project"
```

### Step 2: Create and Activate a Python Virtual Environment
On Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Dataset Path Configuration
The dataset paths are configured in `config.py`:
```python
EXCEL_DATA_PATH = BASE_DIR / "Online Retail.xlsx"
CSV_DATA_PATH = BASE_DIR / "OnlineRetail.csv"
```
Ensure either `Online Retail.xlsx` or `OnlineRetail.csv` exists in the project root. The system automatically prefers Excel and falls back to CSV if needed.

### Step 5: Run the Streamlit Application
```powershell
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

### Step 6: Run the Jupyter Notebook
```powershell
jupyter notebook notebooks/Online_Retail_Analysis.ipynb
```

### Step 7: Run Automated Unit Tests
```powershell
python -m pytest tests/
```

---

## 10. Analytical Methodologies

### A. RFM Customer Segmentation
1. **Recency ($R$)**: Number of days elapsed between customer's most recent purchase and the reference snapshot date ($T_{max} + 1\text{ day}$).
2. **Frequency ($F$)**: Count of unique completed invoice orders placed by the customer.
3. **Monetary ($M$)**: Total GBP spending across all valid non-cancelled transactions.
4. **Skewness Treatment**: Commercial spend and frequency follow an exponential Pareto distribution. We apply $\tilde{x} = \ln(1 + x)$ followed by `StandardScaler` to ensure Euclidean distance in K-Means is not dominated by extreme outliers.
5. **Optimization**:
   * **Elbow Method**: Minimizes within-cluster sum of squares (Inertia).
   * **Silhouette Coefficient**: Measures cluster cohesion vs separation. Optimal partition is $k = 4$ clusters ($\text{Score} \approx 0.32 - 0.35$).

### Derived Customer Personas
* **Champions**: High monetary value, high order frequency, purchased recently. VIP loyalty target.
* **Loyal Customers**: Consistent purchasing cadence and steady monetary contribution.
* **Potential Loyalists**: Recent buyers with moderate frequency and basket value.
* **At-Risk Customers**: High historical spending, but inactive for >90 days. Primary targets for win-back promotions.
* **Hibernating**: Low spend, low frequency, dormant for >200 days.

---

### B. Time-Series Sales Forecasting
1. **Continuous Resampling**: Discrete sales are resampled to daily frequency (`'D'`), zero-filling non-operating calendar days (store closed on Saturdays).
2. **Chronological Splitting**: A strict holdout split (the final 30 calendar days) is used for out-of-sample testing to prevent data leakage.
3. **Algorithms**:
   * **ARIMA / SARIMAX**: Captures autoregression, non-stationary differencing ($d=1$), moving averages, and 7-day weekly seasonality.
   * **Holt-Winters Exponential Smoothing**: Decomposes level, additive trend, and 7-day seasonal components.
4. **Evaluation Metrics**:
   * **MAE**: Mean Absolute Error in £.
   * **RMSE**: Root Mean Squared Error, penalizing large outlier misses.
   * **sMAPE**: Symmetric Mean Absolute Percentage Error, providing bounded percentage metrics when sales fluctuate near zero.
   * **R²**: Proportion of out-of-sample variance explained.

---

## 11. Dashboard Page Descriptions

1. **Executive Overview**: High-level KPIs, 7-day and 30-day moving average revenue trends, top 10 SKUs, geographic pie charts, and interactive date/country filters.
2. **Sales Analytics**: Deep dive into daily revenue line charts, monthly revenue bars, MoM growth rates, hourly purchasing heat, and order value distributions.
3. **Customer Segmentation**: 2D and 3D cluster visualizations, segment revenue share, RFM diagnostic matrices, and interactive customer roster filtered by segment persona.
4. **Sales Forecasting**: Interactive selection of frequency, algorithm, and 7/30/90-day forecast horizons with actual vs predicted curves and 95% confidence intervals.
5. **Product Analytics**: SKU search engine, ranking by Revenue/Quantity/Orders, catalog pricing scatter plot, and cross-border popularity.
6. **Customer Insights**: Individual customer purchase history lookup, observed lifetime spend diagnostics, repeat purchase rates, and lapsing buyer tables.
7. **Data Explorer**: Raw data audit, schema data types, null percentage breakdowns, cancellation analysis, and cleaned CSV export.
8. **Model Performance**: Elbow inertia curves, Silhouette score comparisons across $k \in [2..8]$, comparative forecasting benchmark table, and technical interpretations.
9. **Reports**: Instant generation and download of the 10-chapter Academic Project Report (PDF), Executive Sales Brief (PDF), and CSV manifests.

---

## 12. Methodological Limitations

1. **Historical Horizon Constraint**: The UCI Online Retail dataset spans approximately 12 months (Dec 2010 – Dec 2011). Because only a single annual cycle is observed, true year-over-year annual seasonality cannot be rigorously separated from macroeconomic holiday surges. Projections beyond 30 days should be treated as strategic directional guidance.
2. **Observed Spend vs Predictive CLV**: Customer lifetime values reported represent **Observed Historical Lifetime Spend** within the dataset window. True forward-looking predictive CLV requires probabilistic repeat-purchase models (such as BG/NBD and Gamma-Gamma).
3. **Saturday Closure Pattern**: The physical business did not process transactions on Saturdays, creating an inherent 7-day weekly oscillation.

---

## 13. Future Enhancements

* Integration of deep learning forecasting models (Temporal Fusion Transformers, N-BEATS).
* Probabilistic predictive Customer Lifetime Value modeling via BG/NBD.
* Real-time streaming transaction ingestion via Apache Kafka.
* Automated dynamic pricing engine using reinforcement learning.

---

## 14. Academic References

1. Chen, D., Sain, S.L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208.
2. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control*. 5th Edition, John Wiley & Sons.
3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proc. 5th Berkeley Symp. Math. Statist. Prob.*, 1, 281–297.
4. Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
5. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. 3rd Edition, OTexts.

---

## 15. Author & Project Metadata

* **Project**: RetailIQ – Online Retail Analytics & Intelligence System

* **Environment**: Windows 11 / Python 3.13 / Streamlit 1.35+
* **License**: Educational & Academic Use Only
