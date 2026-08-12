# 🏬 Store Sales — Time Series Forecasting

![Python](https.img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python)
![LightGBM](https.img.shields.io/badge/Model-LightGBM-green.svg?style=flat-square)
![Kaggle](https.img.shields.io/badge/Kaggle-Competition-20BEFF.svg?style=flat-square&logo=kaggle)
![License](https.img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

A end-to-end Machine Learning solution for the [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

This project predicts 16 days of sales (August 16–31, 2017) for **54 stores** across **33 product families** (totaling **1,782 individual time series**) for Corporación Favorita, a large Ecuadorian-based grocery retailer.

---

## 📌 Competition Overview

* **Host**: Corporación Favorita (Kaggle)
* **Goal**: Build a time-series forecasting model to accurately predict grocery store sales.
* **Competition Link**: [Store Sales - Time Series Forecasting on Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
* **Dataset Download**: [Kaggle Competition Data Page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

---

## 📊 Dataset Structure

The dataset contains historical daily unit sales data across multiple stores and product families in Ecuador, along with economic and calendar supplementary data:

| File | Description |
| :--- | :--- |
| `train.csv` | Historical daily sales per store & family (Jan 1, 2013 – Aug 15, 2017). |
| `test.csv` | Target period to forecast (Aug 16, 2017 – Aug 31, 2017). |
| `stores.csv` | Store metadata including city, state, store type, and cluster assignment. |
| `oil.csv` | Daily WTI crude oil prices (Ecuador's economy is highly dependent on oil exports). |
| `holidays_events.csv` | National, regional, and local holidays & special events. |
| `transactions.csv` | Daily total customer transaction count per store. |
| `sample_submission.csv` | Submission format sample with `id` and `sales` target. |

> **Note**: Due to GitHub file size limits (e.g. `train.csv` is ~121 MB), the `Dataset/` folder is listed in `.gitignore`. Please download the dataset directly from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and place the CSV files inside the `Dataset/` directory.

---

## 📐 Evaluation Metric

The competition evaluates submissions using **Root Mean Squared Logarithmic Error (RMSLE)**:

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\log(1+\hat{y}_i) - \log(1+y_i)\right)^2}$$

### 💡 Key Target Transformation Strategy
Rather than using a complex custom loss function, target values are transformed during training using `log1p(sales)`:
$$y_{\text{train}} = \log(1 + \text{sales})$$

Optimizing standard **Root Mean Squared Error (RMSE)** on $y_{\text{train}}$ is mathematically equivalent to minimizing **RMSLE** on raw sales values. After inference, predictions are inverse-transformed back using `expm1()` and clipped to zero to ensure non-negative predictions.

---

## ⚙️ Methodology & Feature Engineering

The forecasting pipeline incorporates domain-specific feature engineering tailored to retail sales in Ecuador:

1. **Calendar & Temporal Features**:
   * Basic calendar markers (`dayofweek`, `month`, `dayofyear`, `is_weekend`).
   * **Quincena Effect (Payday Indicator)**: In Ecuador, salaries are traditionally paid on the 15th and the last day of each month (`is_payday`), causing consistent demand spikes.
   * Cyclical temporal encoding using sine/cosine transformations.

2. **Lag Features ($\ge 16$ days)**:
   * To prevent **data leakage** across the 16-day prediction horizon, historical sales lag features start at minimum lag 16 (`sales_lag_16`, `sales_lag_21`, `sales_lag_28`, etc.).

3. **Rolling Window Statistics**:
   * Moving averages (`sales_roll_mean_7`, `sales_roll_mean_14`, `sales_roll_mean_28`), standard deviation, and rolling min/max over pre-shifted lag windows.

4. **Exogenous Economic Data**:
   * **Crude Oil Prices**: Interpolated daily WTI crude oil prices (`dcoilwtico`) and smoothed rolling averages.
   * **National Holidays**: Binary indicators for non-transferred national holidays.
   * **Promotional Data**: Lagged and rolling promotion intensity (`onpromotion`).

5. **Categorical Encodings**:
   * Label encoding for store metadata (`store_nbr`, `family`, `city`, `state`, `type`, `cluster`).

---

## 🤖 Model Architecture & Validation Strategy

* **Algorithm**: **LightGBM** (Gradient Boosted Decision Trees).
* **Validation Strategy**: Time-based validation split using the last 16 days of training data (August 1–15, 2017) to mirror the exact test set scenario.
* **Early Stopping**: Monitored on validation RMSE (log-scale) to prevent overfitting.

---

## 📁 Repository Structure

```text
Store Sales Forecasting/
├── Dataset/               # Place Kaggle CSV files here (ignored by git)
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   ├── transactions.csv
│   └── sample_submission.csv
├── Model/
│   └── store_sales_forecasting.ipynb   # Main Jupyter Notebook
├── .gitignore             # Excludes Dataset/ and large CSV files
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/NumiKun/Store-Sales-Forecasting.git
cd Store-Sales-Forecasting
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed. Install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn lightgbm scikit-learn
```

### 3. Download Data
Download all CSV files from the [Kaggle Competition Data Page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and place them in the `Dataset/` directory:
```text
Dataset/
  ├── train.csv
  ├── test.csv
  ├── stores.csv
  ├── oil.csv
  ├── holidays_events.csv
  ├── transactions.csv
  └── sample_submission.csv
```

### 4. Run the Jupyter Notebook
Open and run the notebook cell-by-cell in VS Code or Jupyter Lab:
```bash
jupyter notebook Model/store_sales_forecasting.ipynb
```

Running all cells will generate the final predictions saved to `submission.csv`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
