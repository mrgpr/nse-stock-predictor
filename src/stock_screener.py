"""
stock_screener.py
- Scores stocks 0-100 using described weights
- Adds:
  * volatility (20-day std of returns)
  * target_range_low / high based on volatility
  * expected_return_pct
  * sector (from config)
  * risk rating (Low/Medium/High)
  * short rationale 1-2 lines
- Filters top N for each mode
"""
import logging
from typing import Dict, Any
from pathlib import Path
import json
import numpy as np
import math
import pandas as pd

logger = logging.getLogger("stock_screener")


class StockScreener:
    def __init__(self, config_path: Path = Path("config") / "stocks_list.json"):
        # Weights (sum 100)
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

    def _format_price(self, p):
        try:
            return float(p) if p is not None else None
        except Exception:
            return None

    def _compute_volatility(self, df: pd.DataFrame):
        """
        Compute recent volatility as 20-day std dev of daily returns.
        """
        try:
            if df is None or df.empty or "Close" not in df.columns:
                return 0.0
            returns = df["Close"].pct_change().dropna()
            vol20 = returns.rolling(window=20, min_periods=5).std().iloc[-1] if not returns.empty else 0.0
            # fallback
            if vol20 is None or np.isnan(vol20):
                vol20 = returns.std() if not returns.empty else 0.0
            return float(vol20) if vol20 is not None else 0.0
        except Exception as e:
            logger.exception("Volatility computation failed: %s", e)
            return 0.0

    def _risk_rating(self, score: float, volatility: float):
        """
        Simple risk heuristic:
         - High: score < 40 or volatility > 0.06
         - Medium: volatility between 0.03 and 0.06 or score < 65
         - Low: otherwise
        """
        try:
            if score < 40 or volatility > 0.06:
                return "High"
            if volatility > 0.03 or score < 65:
                return "Medium"
            return "Low"
        except Exception:
            return "Unknown"

    def _rationale_from_signals(self, signals: Dict[str, Any]):
        """
        Build 1-2 line rationale from signals:
        - mention MACD, RSI, trend, volume
        """
        parts = []
        macd = signals.get("macd_signal")
        rsi_sig = signals.get("rsi_signal")
        trend = signals.get("trend")
        vol_surge = signals.get("volume_surge")

        if macd in ("bullish_crossover",):
            parts.append("MACD bullish crossover")
        elif macd in ("bearish_crossover",):
            parts.append("MACD bearish crossover")

        if rsi_sig in ("oversold", "strong"):
            parts.append(f"RSI {rsi_sig}")

        if trend and trend != "unknown":
            parts.append(f"Trend: {trend}")

        if vol_surge:
            parts.append("Volume surge")

        # limit to 2 short phrases
        if not parts:
            return "No strong technical signals"
        return "; ".join(parts[:2])

    def score_universe(self, ta_results: Dict[str, Dict[str, Any]], mode="daily"):
        """
        ta_results: { symbol: {"df": df, "signals": signals} }
        Returns a dict with symbol -> scoring metadata including sector, target range, expected return, risk, rationale
        """
        scored = {}
        for sym, info in ta_results.items():
            signals = info.get("signals", {}) or {}
            df = info.get("df")
            score = 0.0

            # RSI scoring
            rsi = signals.get("rsi", None) or 50.0
            if rsi < 30:
                rsi_score = 1.0
            elif rsi < 45:
                rsi_score = 0.8
            elif rsi < 55:
                rsi_score = 0.6
            elif rsi < 70:
                rsi_score = 0.3
            else:
                rsi_score = 0.1
            score += rsi_score * self.weights["rsi"]

            # MACD crossover
            macd_sig = signals.get("macd_signal", "neutral")
            macd_score = 1.0 if macd_sig == "bullish_crossover" else (0.0 if macd_sig == "bearish_crossover" else 0.4)
            score += macd_score * self.weights["macd"]

            # MA trend
            trend = signals.get("trend", "unknown")
            ma_score = 1.0 if trend == "up" else (0.0 if trend == "down" else 0.4)
            score += ma_score * self.weights["ma_trend"]

            # Volume surge
            vol_score = 1.0 if signals.get("volume_surge") else 0.3
            score += vol_score * self.weights["volume"]

            # Momentum
            mom5 = df["mom_5"].iloc[-1] if (df is not None and "mom_5" in df.columns and len(df) > 0) else 0.0
            mom20 = df["mom_20"].iloc[-1] if (df is not None and "mom_20" in df.columns and len(df) > 0) else 0.0
            mom = np.nanmean([mom5, mom20])
            if np.isnan(mom):
                mom = 0.0
            if mom > 0.05:
                mom_score = 1.0
            elif mom > 0.01:
                mom_score = 0.7
            elif mom > -0.01:
                mom_score = 0.4
            else:
                mom_score = 0.1
            score += mom_score * self.weights["momentum"]

            # Bollinger
            bbpos = signals.get("bb_pos", 0.5) or 0.5
            if bbpos < 0.2:
                bb_score = 1.0
            elif bbpos < 0.4:
                bb_score = 0.8
            elif bbpos < 0.6:
                bb_score = 0.5
            elif bbpos < 0.8:
                bb_score = 0.2
            else:
                bb_score = 0.0
            score += bb_score * self.weights["bollinger"]

            final_score = max(0.0, min(100.0, score))

            # Price & target
            last_price = None
            try:
                last_price = float(signals.get("price") if signals.get("price") is not None else (df["Close"].iloc[-1] if (df is not None and "Close" in df.columns and len(df) > 0) else None))
            except Exception:
                last_price = None

            # compute volatility (20-day)
            volatility = self._compute_volatility(df)

            # baseline target (same logic as before)
            support = signals.get("support_20")
            resistance = signals.get("resistance_20")
            target = None
            if last_price is not None:
                if resistance and resistance > last_price:
                    target = float(resistance * 1.03)
                else:
                    target = float(last_price * (1 + 0.05))

            # target range based on volatility: +/- vol (20d) as simple uncertainty
            target_low = None
            target_high = None
            expected_return_pct = None
            if target is not None and last_price is not None:
                # use volatility fraction to define range
                # ensure vol isn't unreasonably small
                vol_factor = max(volatility, 0.01)  # min 1% to avoid 0
                target_low = float(target * (1 - vol_factor))
                target_high = float(target * (1 + vol_factor))
                expected_return_pct = float((target / last_price - 1) * 100)

            # risk rating
            risk = self._risk_rating(final_score, volatility)

            # sector
            sector = self.sectors.get(sym, "Unknown")

            # short rationale
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
                "volatility": round(float(volatility), 4),
                "risk": risk,
                "sector": sector,
                "rationale": rationale
            }

        # Sort all and pick top N
        items = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        top_n = {"daily": 5, "weekly": 10, "monthly": 20}.get(mode, 5)
        return {"all": items, "top": items[:top_n]}
