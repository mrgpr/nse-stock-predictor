"""
data_fetcher.py
- Fetch historical data for Indian stocks using yfinance
- Get real-time quotes
- Batch handling, caching to disk
- Error handling for missing data
"""
from pathlib import Path
import json
import logging
import time
from typing import Dict
import yfinance as yf
import pandas as pd

logger = logging.getLogger("data_fetcher")


class DataFetcher:
    def __init__(self, config_path: Path, historical_path: Path):
        self.config_path = Path(config_path)
        self.historical_path = Path(historical_path)
        self.historical_path.mkdir(parents=True, exist_ok=True)
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def list_all_symbols(self):
        syms = []
        for k, v in self.config.items():
            syms.extend(v)
        # unique and preserve order
        seen = set()
        out = []
        for s in syms:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("^", "")
        return self.historical_path / f"{safe}.csv"

    def fetch_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical data for a single symbol. Uses local cache if not older than 1 day.
        """
        path = self._cache_path(symbol)
        # Use cached data if present and recent
        try:
            if path.exists():
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                # If cached data covers at least period days, reuse (simple heuristic)
                if not df.empty:
                    logger.debug("Using cached data for %s (rows=%d)", symbol, len(df))
                    return df
        except Exception:
            logger.exception("Failed reading cache for %s", symbol)

        # Download new
        logger.info("Downloading history for %s", symbol)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, actions=False)
            if df is None or df.empty:
                raise ValueError(f"No data for {symbol}")
            df = df.rename_axis("Date").reset_index()
            df.to_csv(path, index=False)
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            # Sleep briefly to be friendly to Yahoo
            time.sleep(0.5)
            return df
        except Exception as e:
            logger.exception("Error fetching historical for %s: %s", symbol, e)
            # Return empty DataFrame with expected columns
            cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
            return pd.DataFrame(columns=cols)

    def fetch_batch(self, symbols, period="1y", interval="1d") -> Dict[str, pd.DataFrame]:
        out = {}
        for s in symbols:
            try:
                df = self.fetch_history(s, period=period, interval=interval)
                if df is None or df.empty:
                    logger.warning("Empty dataframe for %s", s)
                out[s] = df
            except Exception as e:
                logger.exception("Failed fetching %s: %s", s, e)
        return out

    def realtime_quote(self, symbol: str) -> dict:
        """
        Return latest market price and basic quote info using yfinance fast_info when possible.
        """
        try:
            t = yf.Ticker(symbol)
            info = {}
            fi = getattr(t, "fast_info", None)
            if fi:
                info["last_price"] = fi.get("lastPrice", None)
                info["previous_close"] = fi.get("previousClose", None)
                info["open"] = fi.get("open", None)
            # fallback: history for last 1 day
            hist = t.history(period="2d", interval="1d")
            if not hist.empty:
                last = hist.iloc[-1]
                info["open"] = info.get("open", last.get("Open"))
                info["last_price"] = info.get("last_price", last.get("Close"))
            return info
        except Exception as e:
            logger.exception("Realtime quote failed for %s: %s", symbol, e)
            return {}
