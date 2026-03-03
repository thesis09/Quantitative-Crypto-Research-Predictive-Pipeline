# Quantitative Crypto Research & Predictive Pipeline

An end-to-end quantitative research pipeline designed to fetch historical cryptocurrency market data, engineer advanced technical features, and train dual predictive machine learning models to forecast localized price action (high/low targets). 

This pipeline emphasizes rigorous financial modeling standards, specifically utilizing `TimeSeriesSplit` cross-validation to eliminate look-ahead bias in temporal financial data.

## 🚀 Key Features

* **Automated Data Ingestion:** Utilizes the Binance REST API to reliably fetch multi-year, high-resolution candlestick data (e.g., 15m intervals).
* **Advanced Feature Engineering:** Calculates critical quantitative market indicators including:
  * Volume Weighted Average Price (VWAP)
  * Time Weighted Average Price (TWAP)
  * Relative Strength Index (RSI)
  * Exponential & Simple Moving Averages (EMA_20, SMA_50)
  * Custom Historical High/Low percentage differentials.
* **Dual Predictive Modeling:** Trains two distinct `XGBRegressor` models to independently predict the percentage difference of future highs and future lows over a specified look-forward window.
* **Robust Cross-Validation:** Implements `RandomizedSearchCV` paired with `TimeSeriesSplit` to ensure the hyperparameter tuning process respects the temporal order of market data, preventing data leakage and overfitting.

## 📂 Project Structure

* `data_retrieval.py`: Handles the Binance API connection, raw data structuring, and the complex calculation of all technical and rolling-window metrics. Exports intermediate data to `.xlsx`.
* `ml_model.py`: Contains the core machine learning logic, including data preprocessing, imputation of missing values, and the setup of the XGBoost models and Time-Series cross-validation grids.
* `main.py`: The main execution script. Orchestrates the flow from data fetching to model training, evaluation, and finally saving the serialized models.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/thesis09/Crypto-Historical-Data-Retrieval-Assessment.git](https://github.com/thesis09/Crypto-Historical-Data-Retrieval-Assessment.git)
   cd Crypto-Historical-Data-Retrieval-Assessment
