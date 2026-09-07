# ✈️ Air Passengers — Time Series Forecasting

A complete beginner-friendly **Time Series Forecasting** project on the classic Air Passengers dataset.
This notebook walks through all 15 steps of Time Series analysis — from raw data to future predictions — with clear code, definitions, and observations.

---

## 📋 Project Overview

| | |
|---|---|
| **Dataset** | Air Passengers Dataset |
| **Total Rows** | 144 months |
| **Period** | January 1949 to December 1960 |
| **Goal** | Predict future monthly airline passengers |
| **Type** | Time Series Forecasting |
| **Algorithm** | ARIMA (1, 1, 1) |
| **Language** | Python |
| **Tools** | Google Colab, Pandas, Statsmodels, Scikit-learn |

---

## 🗺️ All 15 Steps Covered

| # | Step | Description |
|---|---|---|
| 1 | Problem Definition | What are we predicting and why? |
| 2 | Data Collection | Load and inspect the dataset |
| 3 | Data Cleaning | Fix missing values, duplicates, datetime format |
| 4 | Visualization | Plot the raw time series |
| 5 | Trend Analysis | Find long-term direction using rolling mean |
| 6 | Seasonality Analysis | Find repeating yearly patterns |
| 7 | Stationarity Check | ADF Test — is data stable over time? |
| 8 | Differencing | Make data stationary for ARIMA |
| 9 | Train-Test Split | 1949-1958 train, 1959-1960 test |
| 10 | Model Selection | Why ARIMA was chosen |
| 11 | Model Training | Train ARIMA(1,1,1) on training data |
| 12 | Forecasting | Predict 24 months on test period |
| 13 | Evaluation | MAE, MSE, RMSE, MAPE metrics |
| 14 | Visualization | Actual vs Predicted plot |
| 15 | Final Prediction | Forecast future passengers 1961-1962 |

---

## 📁 Project Structure

```
AirPassengers_TimeSeries/
│
├── AirPassengers_TimeSeries.ipynb    ← Main notebook
├── AirPassengers.csv                 ← Dataset
├── README.md                         ← This file
└── LICENSE                           ← MIT License
```

---

## 📊 Dataset Description

| Column | Type | Description |
|---|---|---|
| Month | datetime | Year and month — e.g. 1949-01 |
| Passengers | int | Number of airline passengers that month |

---

## 🎯 Step 1 — Problem Definition

- **Goal:** Predict future monthly airline passenger counts
- **Type:** Time Series Forecasting
- **Target Variable:** Number of Passengers per month
- **Data Period:** January 1949 to December 1960
- **Forecast Period:** 1961 to 1962
- **Success Metric:** MAE, RMSE, MAPE

---

## 📦 Step 2 — Data Collection

- Dataset: Classic Air Passengers dataset
- **144 rows** — one row per month
- **2 columns** — Month and Passengers
- Source: Kaggle — kaggle.com/datasets/rakannimer/air-passengers

---

## 🧹 Step 3 — Data Cleaning

| Check | Result |
|---|---|
| Missing values | 0 — data is complete ✅ |
| Duplicate rows | 0 — no duplicates ✅ |
| Month column type | Converted to datetime ✅ |
| Index | Month set as index ✅ |

---

## 📊 Step 4 — Visualization

- Clear **upward trend** visible from 1949 to 1960
- Clear **seasonal spikes** every summer (July-August)
- Variation increases over time — **multiplicative seasonality**

---

## 📈 Step 5 — Trend Analysis

- Used **12-month Rolling Mean** to identify trend
- Average passengers grew from **~126/month in 1949** to **~476/month in 1960**
- Total growth over 12 years: **~278%**
- Trend is **consistent and accelerating**

---

## 🌊 Step 6 — Seasonality Analysis

- Used **Seasonal Decomposition** to separate Trend + Seasonality + Residual
- **Peak months:** July and August every year
- **Low months:** November and January every year
- Pattern type: **Multiplicative** — seasonal swings grow with trend

---

## 🔍 Step 7 — Stationarity Check (ADF Test)

- **ADF Test p-value > 0.05** — Original data is NOT stationary ❌
- Mean is increasing over time — not constant
- Differencing required before ARIMA modeling

---

## ⚙️ Step 8 — Differencing

- Applied **1st order differencing** (d=1)
- After differencing: **p-value < 0.05** — data became stationary ✅
- Flat plot with no trend — ready for modeling

---

## ✂️ Step 9 — Train-Test Split

| Set | Period | Months |
|---|---|---|
| Training | January 1949 to December 1958 | 120 months |
| Testing | January 1959 to December 1960 | 24 months |

- Data split **in time order** — no shuffling ✅

---

## 🤖 Step 10 — Model Selection

**Why ARIMA?**
- Data has clear trend ✅
- Data has clear seasonality ✅
- Dataset is small (144 rows) ✅
- ARIMA is simple, fast, and reliable ✅

**ARIMA Parameters:**
| Parameter | Value | Meaning |
|---|---|---|
| p | 1 | Autoregression order |
| d | 1 | Differencing order |
| q | 1 | Moving average order |

---

## 🏋️ Step 11 — Model Training

- ARIMA(1,1,1) trained on 120 months of data
- Model learned trend and patterns from 1949-1958
- AIC score calculated for model quality

---

## 🔮 Step 12 — Forecasting

- Model predicted **24 months** on test period (1959-1960)
- Forecast captures upward trend correctly
- Seasonal patterns partially captured

---

## 📏 Step 13 — Model Evaluation

| Metric | Meaning |
|---|---|
| MAE | Average error in passenger count |
| MSE | Penalizes large errors more |
| RMSE | Same unit as passengers — most interpretable |
| MAPE | Error as percentage — MAPE < 10% = excellent |

---

## 📊 Step 14 — Visualization of Results

- Actual vs Predicted plot for 1959-1960
- Error bar chart showing prediction mistakes
- ±10% error band plotted around forecast

---

## 🔮 Step 15 — Final Prediction

- Model retrained on **full 144 months** of data
- Forecasted **24 months into future** (1961-1962)
- Predicted passengers are higher than 1960 — consistent with growth trend
- Output useful for airline **capacity planning and staffing**

---

## 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

---

## ▶️ How to Run on Google Colab

**Step 1** — Upload AirPassengers.csv to Google Drive:
```
drive.google.com → Machine_learning folder → Upload AirPassengers.csv
```

**Step 2** — Open notebook in Google Colab

**Step 3** — Mount Google Drive first (run this before everything):
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 4** — Load dataset:
```python
df = pd.read_csv('/content/drive/MyDrive/Machine_learning/AirPassengers.csv')
```

**Step 5** — Run all cells from top to bottom ✅

---

## 👩‍💻 Author

**Iqra**
Beginner ML Student | Pakistan 🇵🇰
Learning Machine Learning step by step 🚀

---

## 📚 References

- Kaggle — Air Passengers Dataset: kaggle.com/datasets/rakannimer/air-passengers
- Statsmodels Documentation: statsmodels.org
- Pandas Documentation: pandas.pydata.org
- Scikit-learn Documentation: scikit-learn.org
- GeeksforGeeks — Time Series Forecasting with ARIMA (2026)

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
