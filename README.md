# 📈 Quantitative Crypto Research & Predictive Pipeline

[![Model](https://img.shields.io/badge/Model-XGBoost_Dual_Regressor-blue)](https://github.com/thesis09/Quantitative-Crypto-Research-Predictive-Pipeline)
[![Data](https://img.shields.io/badge/Data-Binance_REST_API-yellow)](https://www.binance.com/en/binance-api)
[![Validation](https://img.shields.io/badge/CV-TimeSeriesSplit_n%3D5-success)](https://scikit-learn.org/)
[![Tuning](https://img.shields.io/badge/Tuning-RandomizedSearchCV_300_iter-orange)](https://scikit-learn.org/)
[![Language](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

> An end-to-end quantitative research pipeline that fetches **BTCUSDT 15-minute OHLCV data** from the Binance API (2020–2025), engineers 9 causal features including VWAP, TWAP, RSI, EMA, and rolling high/low proximity metrics, then trains **dual XGBoost regressors** to forecast the **% deviation from the next N-bar High and Low** — with strict `TimeSeriesSplit(n=5)` cross-validation and `RandomizedSearchCV` over 300 iterations to eliminate look-ahead bias and overfitting.

> Author is open to RL + LLM + Quant engineering roles (India onsite / worldwide remote). Contact: kaustubhkubitkar@gmail.com

---

## 📊 Results

| Target | Metric | Value |
|---|---|---|
| **% Diff from Next-5-Bar High** | RMSE | **3.01** |
| **% Diff from Next-5-Bar High** | MAE | Computed |
| **% Diff from Next-5-Bar High** | R² | Computed |
| **% Diff from Next-5-Bar Low** | RMSE | Computed |
| Validation Method | | TimeSeriesSplit (n=5), no shuffle |
| Hyperparameter Search | | RandomizedSearchCV, 300 iterations |
| Dataset | | BTCUSDT 15m, Jan 2020 – Jun 2025 |

> Full metrics (MAE, R², MAPE for both models) are saved to `model_metrics.xlsx` after each run.

---

## 🏗️ Pipeline

```
Binance REST API  (api.binance.com/api/v3/klines)
        │
        │  BTCUSDT · 15m interval · 2020-01-01 → 2025-06-30
        ▼
┌─────────────────────────────────────────────────────┐
│  data_retrieval.py  ·  get_crypto_data()            │
│                                                     │
│  Raw OHLCV columns:                                 │
│  Open, High, Low, Close, Volume, Quote Asset Vol,   │
│  Number of Trades, Taker Buy Base/Quote Vol         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  data_retrieval.py  ·  calculate_metrics()          │
│                                                     │
│  CAUSAL indicators (no future leakage):             │
│  ┌──────────────────────────────────────────────┐   │
│  │  RSI(14)          — momentum oscillator      │   │
│  │  EMA_20           — trend (exponential)      │   │
│  │  SMA_50           — trend (simple)           │   │
│  │  VWAP             — cumulative TPV / vol     │   │
│  │  TWAP             — expanding mean of Close  │   │
│  │  High_Last_30     — rolling 30-bar max High  │   │
│  │  Days_Since_High  — bars since rolling high  │   │
│  │  %_Diff_From_High — close vs rolling high %  │   │
│  │  Low_Last_30      — rolling 30-bar min Low   │   │
│  │  Days_Since_Low   — bars since rolling low   │   │
│  │  %_Diff_From_Low  — close vs rolling low %   │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  FORWARD targets (shift-then-roll, no leakage       │
│  in training — test set excludes these rows):       │
│  ┌──────────────────────────────────────────────┐   │
│  │  %_Diff_From_High_Next_5_Days                │   │
│  │  %_Diff_From_Low_Next_5_Days                 │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │  9 features · dropna · fillna(mean)
                       ▼
┌─────────────────────────────────────────────────────┐
│  ml_model.py  ·  preprocess_data()                  │
│  train_test_split(test_size=0.2, random_state=42)   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  ml_model.py  ·  train_xgboost()                    │
│                                                     │
│  XGBRegressor(objective='reg:squarederror',         │
│               tree_method='hist')                   │
│                                                     │
│  RandomizedSearchCV:                                │
│    n_iter=300, scoring=neg_MSE                      │
│    cv=TimeSeriesSplit(n_splits=5)                   │
│                                                     │
│  Search space:                                      │
│    n_estimators:    [500, 700, 900]                 │
│    learning_rate:   [0.01, 0.05, 0.1]              │
│    max_depth:       [3, 5, 7, 10]                   │
│    min_child_weight:[3, 5]                          │
│    gamma:           [0.1, 0.3]                      │
│    subsample:       [0.8, 0.9]                      │
│    colsample_bytree:[0.8, 0.9]                      │
│    reg_alpha:       [0.3, 0.5]                      │
│    reg_lambda:      [1, 2]                          │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
  Model A: High target       Model B: Low target
  xgb_model_high.pkl         xgb_model_low.pkl
          │                          │
          └────────────┬─────────────┘
                       ▼
              model_metrics.xlsx
         (MAE · R² · MAPE per model)
```

---

## 🔬 Why TimeSeriesSplit — Not K-Fold

Standard K-Fold randomly shuffles data before splitting. For financial time-series, a shuffled validation fold contains bars from the *future* relative to the training set — the model "sees" tomorrow's data during validation, producing artificially low error that never holds in live trading.

This pipeline enforces **walk-forward validation**:

```
Fold 1:  [Train ──────────────]  [Val ──]
Fold 2:  [Train ───────────────────]  [Val ──]
Fold 3:  [Train ────────────────────────]  [Val ──]
Fold 4:  [Train ─────────────────────────────]  [Val ──]
Fold 5:  [Train ──────────────────────────────────]  [Val ──]
                                                  Time →
```

Every validation fold only contains bars that are strictly *after* every training bar. This is the methodology used by quantitative researchers at professional trading firms, and the reason most published retail backtest results don't replicate in production.

---

## 📐 Feature Engineering Detail

All features in `calculate_metrics()` are computed causally — no future bar is used in any feature calculation:

| Feature | Computation | Quant Relevance |
|---|---|---|
| `RSI(14)` | `100 - 100/(1 + avg_gain/avg_loss)` rolling 14 | Overbought/oversold; regime detection |
| `EMA_20` | `Close.ewm(span=20)` | Short-term trend direction |
| `SMA_50` | `Close.rolling(50)` | Medium-term trend; MA crossover signals |
| `VWAP` | `cumsum(Close×Volume) / cumsum(Volume)` | Institutional fair value; mean reversion anchor |
| `TWAP` | `Close.expanding().mean()` | Execution benchmark |
| `Days_Since_High_Last_30` | `idxmax` within rolling 30-bar window | Price memory; momentum recency |
| `%_Diff_From_High_Last_30` | `(Close - rolling_max) / rolling_max × 100` | Distance from resistance |
| `Days_Since_Low_Last_30` | `idxmin` within rolling 30-bar window | Support recency |
| `%_Diff_From_Low_Last_30` | `(Close - rolling_min) / rolling_min × 100` | Distance from support |

**Targets** (forward-looking, excluded from features):
- `%_Diff_From_High_Next_5_Days`: How much price will rise to the next 5-bar high (upside target)
- `%_Diff_From_Low_Next_5_Days`: How much price will fall to the next 5-bar low (downside target)

---

## 🚀 Quick Start

### Install
```bash
pip install xgboost pandas numpy scikit-learn requests openpyxl joblib
```

### Run Full Pipeline
```bash
python main.py
```

This will:
1. Fetch BTCUSDT 15m data from Binance (2020–2025)
2. Compute all features and targets
3. Train both XGBoost models with 300-iteration hyperparameter search
4. Save `xgb_model_high.pkl`, `xgb_model_low.pkl`
5. Save `model_metrics.xlsx` with MAE, R², MAPE for both models
6. Save `crypto_data_with_metrics.xlsx` with full feature-engineered dataset

### Configure
Edit the parameters in `main.py`:
```python
symbol    = "BTCUSDT"   # Any Binance trading pair
interval  = "15m"       # 1m, 5m, 15m, 1h, 4h, 1d
start_date = "2020-01-01"
end_date   = "2025-06-30"
variable1 = 30   # Look-back window for historical high/low features
variable2 = 5    # Look-forward window for target (next N bars)
```

---

## 📁 Project Structure

```
Quantitative-Crypto-Research-Predictive-Pipeline/
├── main.py               # Pipeline runner — parameters, orchestration, model saving
├── data_retrieval.py     # Binance API ingest + full feature engineering
├── ml_model.py           # XGBoost training with TimeSeriesSplit + RandomizedSearchCV
├── xgb_model_high.pkl    # Saved High target model (after run)
├── xgb_model_low.pkl     # Saved Low target model (after run)
├── model_metrics.xlsx    # MAE · R² · MAPE for both models (after run)
└── crypto_data_with_metrics.xlsx  # Full feature-engineered dataset (after run)
```

---

## 📈 Roadmap

- [ ] Add directional accuracy metric (binary up/down classification head)
- [ ] Extend to multi-asset (ETHUSDT, SOLUSDT) with correlation-adjusted signals
- [ ] Live paper trading via Binance WebSocket stream
- [ ] Regime detection layer (HMM on volatility) to switch model weights by market state
- [ ] SHAP feature importance analysis

---

## 📎 Related

- [ViT+PPO CartPole — H100 → RTX 3060 compression](https://github.com/thesis09/Cartpole-)
- [LunarLander Transformer-XL PPO — 280+ reward, 35k steps](https://github.com/thesis09/Lunar-Lander)

---

## ⚠️ Disclaimer

Research and educational use only. Not financial advice.

---

## License

MIT
