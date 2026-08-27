# =============================================================================
# 3_ml_predictor.py — ML Predictor
# For every ticker in filtered_data.json:
#   1. Downloads 1yr OHLCV from yfinance
#   2. Engineers technical indicator features
#   3. Trains 5 regression models
#   4. Evaluates on chronological hold-out (RMSE, MAE, R²)
#   5. Saves the best model as models/{TICKER}_best_model.pkl
#   6. Saves metadata + next-day prediction to models/model_metadata.json
#
# Run: python scripts/3_ml_predictor.py
# =============================================================================

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FILTERED_DATA_PATH, MODELS_DIR, MODEL_METADATA_PATH,
    YFINANCE_PERIOD, TEST_SPLIT_RATIO, MIN_TRADING_DAYS,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_PERIOD, SMA_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    MODELS_TO_TRAIN
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI without external dependency."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (MACD line, Signal line)."""
    ema_fast   = series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow   = series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal     = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line, signal


def compute_bollinger(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper band, middle band, lower band)."""
    mid   = series.rolling(window=BOLLINGER_PERIOD).mean()
    std   = series.rolling(window=BOLLINGER_PERIOD).std()
    upper = mid + BOLLINGER_STD * std
    lower = mid - BOLLINGER_STD * std
    return upper, mid, lower


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw OHLCV DataFrame from yfinance and return a feature matrix.
    Target column: 'target' = next day's closing price (shifted -1).
    All features are based on current and past data only (no lookahead).
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # --- Price-based features ---
    df["ema_20"]  = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    df["sma_50"]  = close.rolling(window=SMA_PERIOD).mean()
    df["rsi"]     = compute_rsi(close)

    macd_line, macd_signal   = compute_macd(close)
    df["macd"]               = macd_line
    df["macd_signal"]        = macd_signal
    df["macd_hist"]          = macd_line - macd_signal

    bb_upper, bb_mid, bb_lower = compute_bollinger(close)
    df["bb_upper"]    = bb_upper
    df["bb_lower"]    = bb_lower
    df["bb_width"]    = (bb_upper - bb_lower) / (bb_mid + 1e-10)  # Normalised width
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)  # 0-1 price in band

    # --- Price ratios (scale-invariant) ---
    df["close_to_ema20"] = close / (df["ema_20"] + 1e-10)
    df["close_to_sma50"] = close / (df["sma_50"] + 1e-10)
    df["high_low_range"] = (high - low) / (close + 1e-10)

    # --- Return features ---
    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    # --- Volume features ---
    df["volume_sma20"]   = volume.rolling(20).mean()
    df["volume_ratio"]   = volume / (df["volume_sma20"] + 1e-10)

    # --- Lag features (yesterday and 2 days ago closing price change) ---
    df["lag_1"] = close.shift(1)
    df["lag_2"] = close.shift(2)
    df["lag_5"] = close.shift(5)

    # --- Target: next day's close ---
    df["target"] = close.shift(-1)

    # Drop rows with NaN from rolling windows or shifts
    df.dropna(inplace=True)

    return df


FEATURE_COLUMNS = [
    "ema_20", "sma_50", "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_position",
    "close_to_ema20", "close_to_sma50", "high_low_range",
    "return_1d", "return_5d", "return_10d",
    "volume_ratio", "lag_1", "lag_2", "lag_5",
    # Raw OHLCV also included as features
    "Open", "High", "Low", "Close", "Volume",
]

# -----------------------------------------------------------------------------
# Model Definitions
# All models that need scaling are wrapped in a Pipeline with StandardScaler.
# Tree-based models (XGBoost, RF, GB) don't need scaling but it doesn't hurt.
# -----------------------------------------------------------------------------

def get_models() -> dict:
    """Return dict of model name → sklearn-compatible regressor/pipeline."""
    models = {}

    if "XGBoost" in MODELS_TO_TRAIN:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    if "RandomForest" in MODELS_TO_TRAIN:
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )

    if "GradientBoosting" in MODELS_TO_TRAIN:
        models["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )

    if "Ridge" in MODELS_TO_TRAIN:
        # Ridge needs scaling — wrap in Pipeline
        models["Ridge"] = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ])

    if "SVR" in MODELS_TO_TRAIN:
        # SVR is sensitive to scale — wrap in Pipeline
        models["SVR"] = Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel="rbf", C=100, epsilon=0.1)),
        ])

    return models

# -----------------------------------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, R², and MAPE metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    # Mean Absolute Percentage Error — intuitive business metric
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4), "mape": round(mape, 4)}


def pick_best_model(results: dict) -> str:
    """
    Pick the best model name from a {model_name: metrics} dict.
    Scoring: lowest RMSE is primary, highest R² as tiebreaker.
    """
    return min(results.keys(), key=lambda m: (results[m]["rmse"], -results[m]["r2"]))

# -----------------------------------------------------------------------------
# Per-Ticker Pipeline
# -----------------------------------------------------------------------------

def train_ticker(ticker: str) -> dict | None:
    """
    Full pipeline for one ticker:
    download → feature engineer → split → train all models → evaluate → save best.
    Returns metadata dict or None if ticker should be skipped.
    """
    log.info(f"\n{'─'*50}")
    log.info(f"Processing: {ticker}")

    # 1. Download data
    try:
        raw = yf.download(ticker, period=YFINANCE_PERIOD, auto_adjust=True, progress=False)
    except Exception as e:
        log.warning(f"  yfinance download failed for {ticker}: {e}")
        return None

    if raw.empty or len(raw) < MIN_TRADING_DAYS:
        log.warning(f"  Insufficient data for {ticker} ({len(raw)} rows). Skipping.")
        return None

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # 2. Engineer features
    df = engineer_features(raw)

    if len(df) < MIN_TRADING_DAYS:
        log.warning(f"  After feature engineering, {ticker} has {len(df)} rows. Skipping.")
        return None

    # 3. Chronological train/test split (no shuffling — this is time series data)
    split_idx = int(len(df) * (1 - TEST_SPLIT_RATIO))
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    X      = df[feature_cols].values
    y      = df["target"].values
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {len(feature_cols)}")

    # 4. Train all models and evaluate
    models  = get_models()
    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate(y_test, y_pred)
            results[name] = metrics
            log.info(
                f"  {name:<20} RMSE={metrics['rmse']:.2f}  "
                f"MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}  MAPE={metrics['mape']:.2f}%"
            )
        except Exception as e:
            log.warning(f"  {name} failed: {e}")

    if not results:
        log.warning(f"  All models failed for {ticker}. Skipping.")
        return None

    # 5. Pick and save the best model
    best_name  = pick_best_model(results)
    best_model = models[best_name]
    model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.pkl")
    joblib.dump(best_model, model_path)
    log.info(f"  ✅ Best model: {best_name} → saved to {model_path}")

    # 6. Predict tomorrow's price using the last row of ALL data (most recent)
    # Re-engineer features on full dataset to get the latest feature vector
    df_full = engineer_features(raw)
    last_features = df_full[feature_cols].iloc[-1].values.reshape(1, -1)

    try:
        tomorrow_prediction = float(best_model.predict(last_features)[0])
    except Exception as e:
        log.warning(f"  Prediction failed for {ticker}: {e}")
        tomorrow_prediction = None

    current_price = float(df["Close"].iloc[-1])

    return {
        "ticker":              ticker,
        "best_model":          best_name,
        "model_path":          model_path,
        "current_price":       round(current_price, 4),
        "predicted_next_close": round(tomorrow_prediction, 4) if tomorrow_prediction else None,
        "predicted_direction": (
            "up" if tomorrow_prediction and tomorrow_prediction > current_price
            else "down" if tomorrow_prediction and tomorrow_prediction < current_price
            else "flat"
        ),
        "all_model_metrics":   results,
        "best_model_metrics":  results[best_name],
        "train_rows":          len(X_train),
        "test_rows":           len(X_test),
        "feature_columns":     feature_cols,
        "trained_at":          datetime.now(timezone.utc).isoformat(),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("STEP 3 — ML Predictor")
    log.info("=" * 60)

    # Load tickers from filtered_data.json
    if not os.path.exists(FILTERED_DATA_PATH):
        log.error(f"filtered_data.json not found. Run 2_sentiment.py first.")
        sys.exit(1)

    with open(FILTERED_DATA_PATH, "r", encoding="utf-8") as f:
        filtered = json.load(f)

    tickers = filtered.get("unique_tickers", [])

    if not tickers:
        log.error("No tickers found in filtered_data.json. Check your scraper output.")
        sys.exit(1)

    log.info(f"Tickers to train: {tickers}")

    # Train models for each ticker
    all_metadata = {}
    failed = []

    for ticker in tqdm(tickers, desc="Training models", unit="ticker"):
        result = train_ticker(ticker)
        if result:
            all_metadata[ticker] = result
        else:
            failed.append(ticker)

    # Save all metadata to a single JSON
    metadata_output = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "tickers_trained": sorted(all_metadata.keys()),
        "tickers_failed":  failed,
        "predictions":     all_metadata,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_output, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info(f"✅ Models trained: {len(all_metadata)} | Failed: {len(failed)}")
    log.info(f"Metadata saved → {MODEL_METADATA_PATH}")

    if failed:
        log.warning(f"Failed tickers: {failed}")

    # Final prediction summary table
    log.info("\n--- Prediction Summary ---")
    for ticker, meta in sorted(all_metadata.items()):
        direction_arrow = "▲" if meta["predicted_direction"] == "up" else ("▼" if meta["predicted_direction"] == "down" else "→")
        log.info(
            f"  {ticker:<6} | Current: ${meta['current_price']:>8.2f} | "
            f"Predicted: ${meta['predicted_next_close']:>8.2f} {direction_arrow} | "
            f"Best: {meta['best_model']:<20} | R²={meta['best_model_metrics']['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
