# 📈 Quantitative Crypto Research & Predictive Pipeline

[![Model](https://img.shields.io/badge/Model-XGBoost_Dual_Regressor-blue)](https://github.com/thesis09/Quantitative-Crypto-Research-Predictive-Pipeline)
[![RMSE](https://img.shields.io/badge/RMSE-3.01-brightgreen)](https://github.com/thesis09/Quantitative-Crypto-Research-Predictive-Pipeline)
[![Validation](https://img.shields.io/badge/Validation-TimeSeriesSplit_(No_Lookahead)-success)](https://github.com/thesis09/Quantitative-Crypto-Research-Predictive-Pipeline)
[![Data](https://img.shields.io/badge/Data-Binance_REST_API-yellow)](https://www.binance.com/en/binance-api)
[![Language](https://img.shields.io/badge/Language-Python-blue)](https://python.org)

> An end-to-end quantitative research pipeline for forecasting **localized price action (High/Low targets)** across cryptocurrency markets. Built with rigorous financial ML methodology: strict **TimeSeriesSplit cross-validation**, explicit **look-ahead bias elimination**, and dual **XGBoost Regressors** with hyperparameter optimization — achieving **RMSE of 3.01** on out-of-sample test data.

---

## 📊 Results

| Metric | Value | Notes |
|---|---|---|
| **RMSE (High target)** | **3.01** | Out-of-sample test set |
| **Validation Method** | TimeSeriesSplit | No data leakage, no look-ahead |
| **Hyperparameter Search** | RandomizedSearchCV | Cross-validated on time-ordered folds |
| **Features** | VWAP, TWAP, RSI, EMA + custom | Full indicator engineering suite |
| **Data Source** | Binance REST API | Historical OHLCV |

> **On look-ahead bias**: The majority of published retail algo strategies are invalidated by data leakage — using future information to predict the past. This pipeline uses strict `TimeSeriesSplit` from scikit-learn, ensuring every training fold contains only data that precedes the validation fold in time. No standard K-Fold. No shuffled splits. This is the correct methodology.

---

## 🏗️ Pipeline Architecture

```
Binance REST API
      │
      ▼
┌─────────────────────────┐
│  Historical Data Ingest  │  OHLCV at multiple timeframes
│  (binance-python client) │  Configurable lookback window
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Feature Engineering Module              │
│                                         │
│  Price-derived:                         │
│    • VWAP  (Volume-Weighted Avg Price)  │
│    • TWAP  (Time-Weighted Avg Price)    │
│    • EMA   (Exponential Moving Average) │
│    • RSI   (Relative Strength Index)    │
│    • Custom momentum / volatility feats │
│                                         │
│  All features computed causally —       │
│  no future data at any step             │
└──────────┬──────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Dual XGBoost Regressor Pipeline     │
│                                      │
│  Model A: Forecast next-bar HIGH     │
│  Model B: Forecast next-bar LOW      │
│                                      │
│  Tuning: RandomizedSearchCV          │
│  CV:     TimeSeriesSplit (n=5)       │
│  Metric: RMSE on held-out fold       │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Out-of-Sample Evaluation    │
│  RMSE: 3.01 (High target)    │
│  Prediction vs Actual plot   │
└──────────────────────────────┘
```

---

## 🔬 Methodology: Why TimeSeriesSplit Matters

Standard ML pipelines use random K-Fold cross-validation. For financial time-series, this is a critical error: a randomly shuffled fold can include data from the future relative to the training window, creating an artificially low validation error that doesn't generalize to live trading.

This pipeline enforces **walk-forward validation**:

```
Fold 1:  [Train: t=0..100]  [Val: t=101..120]
Fold 2:  [Train: t=0..120]  [Val: t=121..140]
Fold 3:  [Train: t=0..140]  [Val: t=141..160]
...
```

Every validation fold only contains data that the model could not have seen during training. This is the methodology used by quantitative researchers at professional trading firms.

---

## 📐 Feature Engineering

All features are computed using only information available at prediction time (causal computation):

| Feature | Description | Quant Relevance |
|---|---|---|
| **VWAP** | Volume-Weighted Average Price | Institutional fair value reference; mean reversion anchor |
| **TWAP** | Time-Weighted Average Price | Execution benchmark; algo order sizing |
| **EMA** | Exponential Moving Average (multi-period) | Trend direction and momentum |
| **RSI** | Relative Strength Index | Overbought/oversold regime detection |
| **Custom features** | Volatility, spread, price momentum | Regime-dependent signal generation |

---

## 🚀 Quick Start

### Install

```bash
pip install xgboost pandas numpy scikit-learn python-binance matplotlib
```

### Run Pipeline

```bash
python pipeline.py --symbol BTCUSDT --interval 1h --lookback 500
```

### Evaluate Model

```bash
python evaluate.py --model models/xgb_high.pkl --test_data data/test.csv
```

---

## 📁 Project Structure

```
Quantitative-Crypto-Research-Predictive-Pipeline/
├── data_ingest.py          # Binance API historical data fetcher
├── feature_engineering.py  # VWAP, TWAP, RSI, EMA + custom indicators
├── model.py                # Dual XGBoost with RandomizedSearchCV + TimeSeriesSplit
├── evaluate.py             # Out-of-sample evaluation + RMSE reporting
├── pipeline.py             # End-to-end runner
├── notebooks/
│   └── research.ipynb      # EDA, feature importance, prediction plots
└── README.md
```

---

## 📈 Roadmap

- [ ] Add directional accuracy metric (binary up/down classification layer)
- [ ] Extend to multi-asset portfolio with correlation-adjusted position sizing
- [ ] Live paper trading integration via Binance WebSocket
- [ ] Regime detection layer (HMM or clustering on volatility) to switch model weights

---

## 📎 Related

- [ViT+PPO CartPole](https://github.com/thesis09/Cartpole-) — RL agent, H100 → RTX 3060 compression
- [LunarLander Transformer-XL PPO](https://github.com/thesis09/Lunar-Lander) — 280+ reward, 35k steps

---

## ⚠️ Disclaimer

This project is for research and educational purposes only. Nothing here constitutes financial advice or a trading recommendation.

---

## License

MIT
