"""
modeling.py
- Lightweight training skeleton using RandomForestRegressor
- Prepares features from TA-enriched dataframe and persists models
- NOTE: This is a starting point for experimentation and offline training
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger("modeling")
logger.setLevel(logging.INFO)

MODEL_DIR = Path("data") / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    # Ensure expected TA columns exist — if not, attempt safe defaults
    for col in ["rsi_14", "macd_diff", "mom_5", "mom_20", "sma20", "sma50"]:
        if col not in df.columns:
            df[col] = 0.0
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["sma20_slope"] = df["sma20"].pct_change(5).fillna(0)
    df["sma50_slope"] = df["sma50"].pct_change(10).fillna(0)
    df["vol_20"] = df["Close"].pct_change().rolling(20).std().fillna(0)
    features = ["rsi_14", "macd_diff", "mom_5", "mom_20", "sma20_slope", "sma50_slope", "vol_20"]
    X = df[features].dropna()
    y = df["ret_5"].shift(-5).reindex(X.index)
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def train_for_symbol(symbol: str, df: pd.DataFrame, n_estimators: int = 100):
    X, y = prepare_features(df)
    if X.empty:
        logger.warning("No training data for %s", symbol)
        return None
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_mean_squared_error")
    logger.info("Symbol %s CV MSE: %.6f (mean)", symbol, -scores.mean())
    pipeline.fit(X, y)
    out = MODEL_DIR / f"{symbol.replace('/', '_')}_rf.joblib"
    joblib.dump(pipeline, out)
    logger.info("Saved model: %s", out)
    return out
