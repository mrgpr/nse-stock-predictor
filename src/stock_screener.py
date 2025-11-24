"""
stock_screener.py
- Scoring stocks 0-100 using the existing weighted heuristic
- Adds timeframe-aware volatility lookbacks:
    daily -> 20
    weekly -> 40
    monthly -> 60
    quarterly -> 120 (3 months x 20? see note)
    biquarterly -> 120 (6 months -> 120)  # we'll use 120 for 6 months
    yearly -> 240
- Volatility computed as the conservative max of:
    * ATR(14) / price  (short-term true range)
    * rolling std of returns over the timeframe window
- Targets and target ranges sized using the volatility measure appropriate to the mode.
- Returns sector, risk, rationale, expected_return_pct, target_low/high
"""
import logging
from typing import Dict, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

logger = logging.getLogger("stock_screener")
logger.setLevel(logging.INFO)


# Default mapping (trading days)
DEFAULT_VOL_WINDOW = {
    "daily": 20,         # ~1 month
    "weekly": 40,        # ~2 months
    "monthly": 60,       # ~3 months
    "quarterly": 60,     # quarter (3 months). use 60 to align with monthly medium-term.
    "biquarterly": 120,  # 6 months -> 120 trading days approx
    "yearly": 240        # ~1 year trading days (approx)
}


class StockScreener:
    def __init__(self, config_path: Path = Path("config") / "stocks_list.json", vol_window_map: Dict[str, int] = None):
        self.weights = {
            "rsi": 20,
            "macd": 20,
            "ma_trend": 20,
            "volume": 15,
            "momentum": 15,
            "bollinger": 10
        }
        self.config_path = Path(config_path)
        self.sectors = {}
        self.vol_window_map = vol_window_map or DEFAULT_VOL_WINDOW
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.sectors = config.get("sectors", {})
        except Exception as e:
            logger.warning("Failed to load sectors from config: %s", e)
            self.sectors = {}

    def _compute_atr_pct(self, df: pd.DataFrame):
        """
        ATR(14)/price as short-term volatility proxy.
        """
        try:
            if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
                return 0.0
            atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
            atr_series = atr.average_true_range()
            if atr_series is None or atr_series.empty:
                return 0.0
            atr_latest = float(atr_series.iloc[-1])
            price = float(df["Close"].iloc[-1])
            if price <= 0:
                return 0.0
            return max(0.0, atr_latest / price)
        except Exception:
            logger.exception("ATR computation failed")
            return 0.0

    def _compute_return_std(self, df: pd.DataFrame, window: int):
        """
        Rolling std of daily returns over `window` days. Fallback safe returns.
        """
        try:
            if df is None or df.empty or "Close" not in df.columns:
                return 0.0
            returns = df["Close"].pct_change().dropna()
            if returns.empty:
                return 0.0
            if window < 2:
                window = 2
            val = returns.rolling(window=window, min_periods=5).std().iloc[-1]
            if pd.isna(val):
                val = returns.std()
            return float(val) if not pd.isna(val) else 0.0
        except Exception:
            logger.exception("Return std computation failed")
            return 0.0

    def _volatility_for_mode(self, df: pd.DataFrame, mode: str):
        """
        Determine volatility based on mode's window:
         - compute ATR_pct (short-term)
         - compute returns std over vol_window
         - return the conservative max(atr_pct, ret_std)
        """
        vol_window = int(self.vol_window_map.get(mode, DEFAULT_VOL_WINDOW.get(mode, 20)))
        atr_pct = self._compute_atr_pct(df)
        ret_std = self._compute_return_std(df, vol_window)
        # Conservative: pick the larger (more risk-aware)
        vol = max(atr_pct, ret_std)
        return vol, atr_pct, ret_std, vol_window

    def _rationale_from_signals(self, signals: Dict[str, Any]):
        parts = []
        macd = signals.get("macd_signal")
        rsi_sig = signals.get("rsi_signal")
        trend = signals.get("trend")
        vol_surge = signals.get("volume_surge")
        if macd == "bullish_crossover":
            parts.append("MACD bullish crossover")
        if rsi_sig in ("oversold", "strong"):
            parts.append(f"RSI {rsi_sig}")
        if trend and trend != "unknown":
            parts.append(f"Trend: {trend}")
        if vol_surge:
            parts.append("Volume surge")
        if not parts:
            return "No strong technical signals"
        return "; ".join(parts[:2])

    def _risk_rating(self, score: float, vol: float):
        """
        Tuned thresholds — adjust as needed per your distribution.
        vol expected as a decimal (e.g., 0.03 ~ 3%)
        """
        try:
            if score < 40 or vol > 0.08:
                return "High"
            if vol > 0.04 or score < 65:
                return "Medium"
            return "Low"
        except Exception:
            return "Unknown"

    def score_universe(self, ta_results: Dict[str, Dict[str, Any]], mode: str = "daily"):
        """
        ta_results: { symbol: {"df": df, "signals": signals} }
        mode: one of daily/weekly/monthly/quarterly/biquarterly/yearly
        Returns dict {all: [...], top: [...]}
        """
        scored = {}
        mode = mode.lower()
        for sym, info in ta_results.items():
            signals = info.get("signals", {}) or {}
            df = info.get("df")
            score = 0.0

            # RSI scoring
            rsi = signals.get("rsi", 50)
            if rsi < 30:
                score += 1.0 * self.weights["rsi"]
            elif rsi < 45:
                score += 0.8 * self.weights["rsi"]
            elif rsi < 55:
                score += 0.6 * self.weights["rsi"]
            elif rsi < 70:
                score += 0.3 * self.weights["rsi"]
            else:
                score += 0.1 * self.weights["rsi"]

            # MACD
            macd_sig = signals.get("macd_signal", "neutral")
            if macd_sig == "bullish_crossover":
                score += 1.0 * self.weights["macd"]
            elif macd_sig == "bearish_crossover":
                score += 0.0 * self.weights["macd"]
            else:
                score += 0.4 * self.weights["macd"]

            # MA trend
            trend = signals.get("trend", "unknown")
            if trend == "up":
                score += 1.0 * self.weights["ma_trend"]
            elif trend == "down":
                score += 0.0 * self.weights["ma_trend"]
            else:
                score += 0.4 * self.weights["ma_trend"]

            # Volume surge
            score += (1.0 if signals.get("volume_surge") else 0.3) * self.weights["volume"]

            # Momentum using mom_5/mom_20 if available
            mom5 = df["mom_5"].iloc[-1] if (df is not None and "mom_5" in df.columns and len(df) > 0) else 0.0
            mom20 = df["mom_20"].iloc[-1] if (df is not None and "mom_20" in df.columns and len(df) > 0) else 0.0
            mom = np.nanmean([mom5, mom20])
            if np.isnan(mom):
                mom = 0.0
            if mom > 0.05:
                score += 1.0 * self.weights["momentum"]
            elif mom > 0.01:
                score += 0.7 * self.weights["momentum"]
            elif mom > -0.01:
                score += 0.4 * self.weights["momentum"]
            else:
                score += 0.1 * self.weights["momentum"]

            # Bollinger position
            bbpos = signals.get("bb_pos", 0.5) or 0.5
            if bbpos < 0.2:
                score += 1.0 * self.weights["bollinger"]
            elif bbpos < 0.4:
                score += 0.8 * self.weights["bollinger"]
            elif bbpos < 0.6:
                score += 0.5 * self.weights["bollinger"]
            elif bbpos < 0.8:
                score += 0.2 * self.weights["bollinger"]
            else:
                score += 0.0 * self.weights["bollinger"]

            final_score = max(0.0, min(100.0, score))

            # Last price
            last_price = None
            try:
                last_price = float(signals.get("price") if signals.get("price") is not None else (df["Close"].iloc[-1] if (df is not None and "Close" in df.columns and len(df) > 0) else None))
            except Exception:
                last_price = None

            # Volatility selection depending on mode
            vol, atr_pct, ret_std, vol_window = self._volatility_for_mode(df, mode)

            # Basic target (resistance or +5%)
            resistance = signals.get("resistance_20")
            if last_price is not None:
                if resistance and resistance > last_price:
                    base_target = float(resistance * 1.03)
                else:
                    base_target = float(last_price * (1 + 0.05))
            else:
                base_target = None

            target = base_target
            target_low = None
            target_high = None
            expected_return_pct = None
            if target is not None and last_price is not None:
                # Range: +/- vol * multiplier
                mult = 1.0
                vol_used = max(vol, 0.01)  # floor to 1%
                target_low = float(target * (1 - vol_used * mult))
                target_high = float(target * (1 + vol_used * mult))
                expected_return_pct = float((target / last_price - 1) * 100)

            risk = self._risk_rating(final_score, vol)
            sector = self.sectors.get(sym, "Unknown")
            rationale = self._rationale_from_signals(signals)

            scored[sym] = {
                "symbol": sym,
                "score": round(final_score, 1),
                "signals": signals,
                "last_price": last_price,
                "target": target,
                "target_low": target_low,
                "target_high": target_high,
                "expected_return_pct": None if expected_return_pct is None else round(expected_return_pct, 2),
                "volatility": round(float(vol), 4),
                "atr_pct": round(float(atr_pct), 4),
                "ret_std": round(float(ret_std), 4),
                "vol_window": vol_window,
                "risk": risk,
                "sector": sector,
                "rationale": rationale
            }

        # Sort & pick top
        items = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        top_n = {"daily": 5, "weekly": 10, "monthly": 20, "quarterly": 20, "biquarterly": 30, "yearly": 50}.get(mode, 5)
        return {"all": items, "top": items[:top_n]}
