"""
technical_analysis.py
- Adds SMA(20,50), EMA(12,26), RSI(14), MACD, Bollinger Bands
- Identifies basic support/resistance (simple pivots), trend direction and volume analysis
- Generates basic buy/sell/hold signals
"""
import logging
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

logger = logging.getLogger("technical_analysis")


class TechnicalAnalyzer:
    def __init__(self):
        pass

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input df: index=Date, columns include Open/High/Low/Close/Volume
        Adds columns for indicators and returns the augmented dataframe.
        """
        df = df.copy()
        if "Close" not in df.columns:
            raise ValueError("DataFrame missing Close column")

        # Ensure numeric
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0)

        # SMA
        df["sma_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()

        # EMA
        df["ema_12"] = EMAIndicator(close=df["Close"], window=12).ema_indicator()
        df["ema_26"] = EMAIndicator(close=df["Close"], window=26).ema_indicator()

        # RSI
        rsi = RSIIndicator(close=df["Close"], window=14)
        df["rsi_14"] = rsi.rsi()

        # MACD
        macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["bb_hband"] = bb.bollinger_hband()
        df["bb_lband"] = bb.bollinger_lband()
        df["bb_mavg"] = bb.bollinger_mavg()
        # Position within bands: 0 (lower) to 1 (upper)
        df["bb_pos"] = (df["Close"] - df["bb_lband"]) / (df["bb_hband"] - df["bb_lband"] + 1e-9)

        # Price momentum: percent change over 5/20 days
        df["mom_5"] = df["Close"].pct_change(5)
        df["mom_20"] = df["Close"].pct_change(20)

        # Simple support/resistance: local mins/maxs over rolling window
        df["spt_20"] = df["Low"].rolling(window=20, min_periods=5).min()
        df["res_20"] = df["High"].rolling(window=20, min_periods=5).max()

        # Trend direction: based on SMA slopes
        df["sma20_slope"] = df["sma_20"].diff()
        df["sma50_slope"] = df["sma_50"].diff()
        df["trend"] = df.apply(self._detect_trend, axis=1)

        return df

    def _detect_trend(self, row):
        try:
            if pd.isna(row["sma_20"]) or pd.isna(row["sma_50"]):
                return "unknown"
            if row["sma_20"] > row["sma_50"] and row["sma20_slope"] > 0:
                return "up"
            if row["sma_20"] < row["sma_50"] and row["sma20_slope"] < 0:
                return "down"
            return "sideways"
        except Exception:
            return "unknown"

    def generate_signals(self, df: pd.DataFrame) -> dict:
        """
        Returns a dict with the latest signals and indicator summary for the most recent row.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        signals = {}
        # RSI
        rsi = latest.get("rsi_14", None)
        signals["rsi"] = rsi
        signals["rsi_signal"] = "neutral"
        if pd.notna(rsi):
            if rsi < 30:
                signals["rsi_signal"] = "oversold"
            elif rsi > 70:
                signals["rsi_signal"] = "overbought"
            elif rsi < 45:
                signals["rsi_signal"] = "weak"
            elif rsi > 55:
                signals["rsi_signal"] = "strong"

        # MACD crossover
        macd_diff = latest.get("macd_diff", None)
        prev_diff = prev.get("macd_diff", None)
        signals["macd_diff"] = macd_diff
        if pd.notna(macd_diff) and pd.notna(prev_diff):
            if prev_diff < 0 and macd_diff > 0:
                signals["macd_signal"] = "bullish_crossover"
            elif prev_diff > 0 and macd_diff < 0:
                signals["macd_signal"] = "bearish_crossover"
            else:
                signals["macd_signal"] = "neutral"
        else:
            signals["macd_signal"] = "unknown"

        # Moving averages status
        signals["price"] = latest.get("Close", None)
        signals["above_sma20"] = bool(latest.get("Close", 0) > latest.get("sma_20", float("inf")))
        signals["above_sma50"] = bool(latest.get("Close", 0) > latest.get("sma_50", float("inf")))
        signals["trend"] = latest.get("trend", "unknown")

        # Volume: compare to 20-day average
        vol = latest.get("Volume", 0)
        vol_avg = df["Volume"].rolling(window=20, min_periods=5).mean().iloc[-1]
        signals["volume"] = vol
        signals["vol_avg_20"] = vol_avg
        signals["volume_surge"] = bool(vol_avg > 0 and vol > vol_avg * 1.5)

        # Bollinger position
        signals["bb_pos"] = latest.get("bb_pos", None)

        # Support/resistance
        signals["support_20"] = latest.get("spt_20", None)
        signals["resistance_20"] = latest.get("res_20", None)

        # Buy/Sell/Hold heuristic
        score_buy = 0
        # simple rules
        if signals.get("macd_signal") == "bullish_crossover":
            score_buy += 1
        if signals.get("rsi_signal") in ("oversold", "strong"):
            score_buy += 1
        if signals.get("trend") == "up":
            score_buy += 1
        if signals.get("volume_surge"):
            score_buy += 1
        # map to recommendation
        if score_buy >= 3:
            signals["recommendation"] = "STRONG BUY"
        elif score_buy == 2:
            signals["recommendation"] = "BUY"
        elif score_buy == 1:
            signals["recommendation"] = "HOLD"
        else:
            signals["recommendation"] = "SELL"

        return signals
