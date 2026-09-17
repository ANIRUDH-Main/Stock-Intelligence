#!/usr/bin/env python3
"""
Reproducible null-calibration experiment for Stock Intelligence.

This script is designed to reproduce the synthetic-data protocol described
in Section V and Appendix A of the accompanying research paper.

It intentionally reads the audited starting prices from:
    ../models/model_metadata.json

Run from the repository root:
    python experiments/null_calibration.py

Outputs:
    data/null_calibration/gbm_paths.csv
    data/null_calibration/null_results.csv
    data/null_calibration/run_manifest.json

The experiment uses:
    - 200 synthetic GBM paths
    - 251 daily observations/path
    - seed 7 for simulation
    - seed 0 reserved for permutation work
    - annualized volatility sampled from {0.20, 0.35, 0.50, 0.80}
    - annualized drift ~ N(0.08, 0.25^2)
    - 160/41 chronological train/test split
    - five regressors with the hyperparameters recorded in Appendix A
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "xgboost is required. Install dependencies with: pip install xgboost"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "models" / "model_metadata.json"
OUT_DIR = ROOT / "data" / "null_calibration"

N_PATHS = 200
N_OBS = 251
TRAIN_ROWS = 160
TEST_ROWS = 41
SIM_SEED = 7
PERMUTATION_SEED = 0
VOLATILITIES = np.array([0.20, 0.35, 0.50, 0.80])
DRIFT_MEAN = 0.08
DRIFT_SD = 0.25

FEATURE_COLUMNS = [
    "ema_20", "sma_50", "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_position",
    "close_to_ema20", "close_to_sma50", "high_low_range",
    "return_1d", "return_5d", "return_10d", "volume_ratio",
    "lag_1", "lag_2", "lag_5",
    "Open", "High", "Low", "Close", "Volume",
]


def load_starting_prices() -> dict[str, float]:
    """Read audited ticker prices from the repository's model metadata."""
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {META_PATH}. Run this script from the complete "
            "Stock-Intelligence repository containing model_metadata.json."
        )

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    prices = {}
    for ticker, record in meta.get("predictions", {}).items():
        price = record.get("current_price")
        if isinstance(price, (int, float)) and price > 0:
            prices[ticker] = float(price)

    if not prices:
        raise ValueError("No positive current_price values found in model_metadata.json.")

    return prices


def generate_gbm_paths(start_prices: dict[str, float]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Generate 200 geometric-Brownian-motion paths.

    Each path samples its starting price from the audited ticker prices and
    volatility uniformly from the four values specified in Appendix A.
    Drift is sampled from N(0.08, 0.25^2).
    """
    rng = np.random.default_rng(SIM_SEED)
    tickers = np.array(sorted(start_prices))
    prices = np.array([start_prices[t] for t in tickers], dtype=float)

    records = []
    manifest_rows = []

    for path_id in range(1, N_PATHS + 1):
        ticker = str(rng.choice(tickers))
        s0 = float(start_prices[ticker])
        sigma = float(rng.choice(VOLATILITIES))
        mu = float(rng.normal(DRIFT_MEAN, DRIFT_SD))

        # Daily GBM with 252 trading days/year.
        dt = 1.0 / 252.0
        z = rng.normal(0.0, 1.0, N_OBS - 1)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
        close = np.empty(N_OBS)
        close[0] = s0
        close[1:] = s0 * np.exp(np.cumsum(log_returns))

        # Synthetic OHLCV construction. OHLCV is only a feature container;
        # the unpredictable component is the close process itself.
        overnight = rng.normal(0.0, 0.002, N_OBS)
        open_px = close * np.exp(overnight)
        intraday_scale = np.abs(rng.normal(0.0, 0.008, N_OBS))
        high_px = np.maximum(open_px, close) * (1.0 + intraday_scale)
        low_px = np.minimum(open_px, close) * (1.0 - intraday_scale)
        volume = np.exp(rng.normal(np.log(1_000_000), 0.35, N_OBS))

        for day in range(N_OBS):
            records.append({
                "path_id": path_id,
                "day": day,
                "ticker_source": ticker,
                "Open": open_px[day],
                "High": high_px[day],
                "Low": low_px[day],
                "Close": close[day],
                "Volume": volume[day],
            })

        manifest_rows.append({
            "path_id": path_id,
            "ticker_source": ticker,
            "initial_price": s0,
            "annualized_volatility": sigma,
            "annualized_drift": mu,
        })

    return pd.DataFrame(records), manifest_rows


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 25-feature specification recorded in Appendix A."""
    out = df.copy().sort_values(["path_id", "day"]).reset_index(drop=True)
    g = out.groupby("path_id", group_keys=False)

    out["ema_20"] = g["Close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    out["sma_50"] = g["Close"].transform(lambda s: s.rolling(50).mean())

    delta = g["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(out["path_id"]).transform(lambda s: s.rolling(14).mean())
    avg_loss = loss.groupby(out["path_id"]).transform(lambda s: s.rolling(14).mean())
    rs = avg_gain / (avg_loss + 1e-10)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = g["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out.groupby("path_id")["macd"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    mid = g["Close"].transform(lambda s: s.rolling(20).mean())
    sd = g["Close"].transform(lambda s: s.rolling(20).std())
    out["bb_upper"] = mid + 2 * sd
    out["bb_lower"] = mid - 2 * sd
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (mid + 1e-10)
    out["bb_position"] = (out["Close"] - out["bb_lower"]) / (
        out["bb_upper"] - out["bb_lower"] + 1e-10
    )

    out["close_to_ema20"] = out["Close"] / out["ema_20"]
    out["close_to_sma50"] = out["Close"] / out["sma_50"]
    out["high_low_range"] = (out["High"] - out["Low"]) / out["Close"]

    out["return_1d"] = g["Close"].pct_change(1)
    out["return_5d"] = g["Close"].pct_change(5)
    out["return_10d"] = g["Close"].pct_change(10)

    vol_mean20 = g["Volume"].transform(lambda s: s.rolling(20).mean())
    out["volume_ratio"] = out["Volume"] / vol_mean20

    out["lag_1"] = g["Close"].shift(1)
    out["lag_2"] = g["Close"].shift(2)
    out["lag_5"] = g["Close"].shift(5)

    return out


def build_models():
    return {
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=0,
            objective="reg:squarederror", n_jobs=1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=0, n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=0,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, epsilon=0.1)),
        ]),
    }


def evaluate_path(path_df: pd.DataFrame) -> list[dict]:
    df = add_features(path_df)
    df["target_next_close"] = df.groupby("path_id")["Close"].shift(-1)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_next_close"]).reset_index(drop=True)

    # The 50-period feature causes the first 49 rows to be removed, yielding
    # exactly 201 usable rows from 251 observations.
    if len(df) != 201:
        raise RuntimeError(f"Expected 201 usable rows, got {len(df)}.")

    train = df.iloc[:TRAIN_ROWS]
    test = df.iloc[TRAIN_ROWS:TRAIN_ROWS + TEST_ROWS]

    X_train, y_train = train[FEATURE_COLUMNS], train["target_next_close"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_next_close"]

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append({
            "model": name,
            "rmse": rmse,
            "mae": float(np.mean(np.abs(y_test - pred))),
            "r2": float(r2_score(y_test, pred)),
            "mape": float(mean_absolute_percentage_error(y_test, pred) * 100),
        })

    persistence = test["Close"].to_numpy()
    p_rmse = float(np.sqrt(mean_squared_error(y_test, persistence)))
    p_mape = float(mean_absolute_percentage_error(y_test, persistence) * 100)
    p_r2 = float(r2_score(y_test, persistence))

    for row in rows:
        row["theil_u"] = row["rmse"] / p_rmse if p_rmse else np.nan
        row["beats_persistence"] = bool(row["rmse"] < p_rmse)

    best = min(rows, key=lambda x: x["rmse"])
    rows.append({
        "model": "Persistence",
        "rmse": p_rmse,
        "mae": float(np.mean(np.abs(y_test - persistence))),
        "r2": p_r2,
        "mape": p_mape,
        "theil_u": 1.0,
        "beats_persistence": False,
    })
    rows.append({
        "model": "Selected (best of 5)",
        "rmse": best["rmse"],
        "mae": best["mae"],
        "r2": best["r2"],
        "mape": best["mape"],
        "theil_u": best["theil_u"],
        "beats_persistence": best["beats_persistence"],
    })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start_prices = load_starting_prices()
    paths, path_manifest = generate_gbm_paths(start_prices)
    paths.to_csv(OUT_DIR / "gbm_paths.csv", index=False)

    all_results = []
    for path_id, path_df in paths.groupby("path_id"):
        for row in evaluate_path(path_df):
            row["path_id"] = int(path_id)
            all_results.append(row)

    results = pd.DataFrame(all_results)
    results = results[
        ["path_id", "model", "rmse", "mae", "r2", "mape", "theil_u", "beats_persistence"]
    ]
    results.to_csv(OUT_DIR / "null_results.csv", index=False)

    selected = results[results["model"] == "Selected (best of 5)"]
    summary = (
        results.groupby("model", as_index=False)
        .agg(
            median_R2=("r2", "median"),
            median_MAPE=("mape", "median"),
            median_Theil_U=("theil_u", "median"),
            beats_persistence_pct=("beats_persistence", "mean"),
        )
    )
    summary["beats_persistence_pct"] *= 100

    manifest = {
        "experiment": "Null calibration of Stock Intelligence forecasting protocol",
        "n_paths": N_PATHS,
        "observations_per_path": N_OBS,
        "usable_rows_per_path": 201,
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "simulation_seed": SIM_SEED,
        "permutation_seed": PERMUTATION_SEED,
        "volatility_values": VOLATILITIES.tolist(),
        "drift_distribution": "Normal(mean=0.08, sd=0.25)",
        "initial_price_source": "models/model_metadata.json",
        "initial_price_tickers": sorted(start_prices),
        "path_parameters": path_manifest,
        "summary": summary.to_dict(orient="records"),
        "note": (
            "This script provides a reproducible implementation of the protocol "
            "specified in the paper. Its output should replace any earlier "
            "unarchived synthetic results if exact historical provenance cannot "
            "be established."
        ),
    }

    with (OUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(summary.to_string(index=False))
    print(f"\nWrote outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
