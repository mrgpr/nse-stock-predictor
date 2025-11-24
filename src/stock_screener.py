"""
stock_screener.py
- Scores stocks 0-100 using described weights:
  * RSI (oversold/overbought) - 20 pts
  * MACD crossover - 20 pts
  * Moving average trends - 20 pts
  * Volume surge - 15 pts
  * Price momentum - 15 pts
  * Bollinger band position - 10 pts
- Filters top N for each mode
- Calculates simple target price and stop loss based on volatility and support/resistance
"""
import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger("stock_screener")


class StockScreener:
    def __init__(self):
        # Weights (sum 100)
        self.weights = {
            "rsi": 20,
            "macd": 20,
            "ma_trend": 20,
            "volume": 15,
            "momentum": 15,
            "bollinger": 10
        }

    def score_universe(self, ta_results: Dict[str, Dict[str, Any]], mode="daily"):
        """
        ta_results: { symbol: {"df": df, "signals": signals} }
        Returns a dict with symbol -> scoring metadata
        """
        scored = {}
        for sym, info in ta_results.items():
            signals = info.get("signals", {})
            df = info.get("df")
            score = 0.0

            # RSI scoring: prefer RSI between 30-60 (buyers prefer < 60). Oversold gives higher score
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

            # Moving average trend
            trend = signals.get("trend", "unknown")
            ma_score = 1.0 if trend == "up" else (0.0 if trend == "down" else 0.4)
            score += ma_score * self.weights["ma_trend"]

            # Volume surge
            vol_score = 1.0 if signals.get("volume_surge") else 0.3
            score += vol_score * self.weights["volume"]

            # Momentum: combine mom_5 & mom_20
            mom5 = df["mom_5"].iloc[-1] if "mom_5" in df.columns and len(df) > 0 else 0.0
            mom20 = df["mom_20"].iloc[-1] if "mom_20" in df.columns and len(df) > 0 else 0.0
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

            # Bollinger position: near upper band is risk; middle is neutral; lower is good
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

            # Normalize to 0-100
            final_score = max(0.0, min(100.0, score))
            # Target price / stop loss
            last_price = signals.get("price") or (df["Close"].iloc[-1] if "Close" in df.columns and len(df) > 0 else None)
            support = signals.get("support_20")
            resistance = signals.get("resistance_20")
            target = None
            stop_loss = None
            if last_price is not None:
                # Target: towards resistance + 5%
                if resistance and resistance > last_price:
                    target = float(resistance * 1.03)
                else:
                    target = float(last_price * (1 + 0.05))  # 5% target by default
                # Stop loss: near support - 2% or 3% of price
                if support and support < last_price:
                    stop_loss = float(max(support * 0.99, last_price * (1 - 0.05)))
                else:
                    stop_loss = float(last_price * (1 - 0.03))

            scored[sym] = {
                "symbol": sym,
                "score": round(final_score, 1),
                "signals": signals,
                "last_price": float(last_price) if last_price is not None else None,
                "target": target,
                "stop_loss": stop_loss,
            }

        # Sort top N by mode
        items = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        top_n = {"daily": 5, "weekly": 10, "monthly": 20}.get(mode, 5)
        return {"all": items, "top": items[:top_n]}
