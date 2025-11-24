"""
stock_screener.py
- Scoring as before with ATR-based volatility
- Computes target range using ATR fraction
- Adds sector, risk, rationale, expected_return_pct
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


class StockScreener:
    def __init__(self, config_path: Path = Path("config") / "stocks_list.json"):
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
        ATR(14) / current price — a volatility proxy
        """
        try:
            if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
                return 0.0
            atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
            atr_val = atr.average_true_range()
            if atr_val is None or atr_val.empty:
                return 0.0
            atr_latest = float(atr_val.iloc[-1])
            price = float(df["Close"].iloc[-1])
            if price <= 0:
                return 0.0
            return max(0.0, atr_latest / price)
        except Exception:
            logger.exception("ATR computation failed")
            return 0.0

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

    def _risk_rating(self, score: float, atr_pct: float):
        try:
            if score < 40 or atr_pct > 0.08:
                return "High"
            if atr_pct > 0.04 or score < 65:
                return "Medium"
            return "Low"
        except Exception:
            return "Unknown"

    def score_universe(self, ta_results: Dict[str, Dict[str, Any]], mode: str = "daily"):
        scored = {}
        for sym, info in ta_results.items():
            signals = info.get("signals", {}) or {}
            df = info.get("df")
            score = 0.0

            # RSI
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

            # ATR-based volatility
            atr_pct = self._compute_atr_pct(df)

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
                # target range: +/- atr_pct * multiplier (use 1.0 by default)
                mult = 1.0
                atr_used = max(atr_pct, 0.01)
                target_low = float(target * (1 - atr_used * mult))
                target_high = float(target * (1 + atr_used * mult))
                expected_return_pct = float((target / last_price - 1) * 100)

            risk = self._risk_rating(final_score, atr_pct)
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
                "volatility": round(float(atr_pct), 4),
                "risk": risk,
                "sector": sector,
                "rationale": rationale
            }

        items = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        top_n = {"daily": 5, "weekly": 10, "monthly": 20}.get(mode, 5)
        return {"all": items, "top": items[:top_n]}
