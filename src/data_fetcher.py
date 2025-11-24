"""
data_fetcher.py
- Uses yfinance.download for bulk downloads (fast)
- Caches per-symbol CSVs in data/historical/
- Falls back to threaded single-ticker fetch if bulk fails
- Includes simple retry/backoff
"""
import logging
from pathlib import Path
import time
import pandas as pd
import yfinance as yf
import os
from typing import Dict

logger = logging.getLogger("data_fetcher")
logger.setLevel(logging.INFO)


class DataFetcher:
    def __init__(self, config_path: Path = Path("config") / "stocks_list.json", historical_path: Path = Path("data") / "historical"):
        self.config_path = Path(config_path)
        self.historical_path = Path(historical_path)
        self.historical_path.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("^", "")
        return self.historical_path / f"{safe}.csv"

    def list_all_symbols(self):
        import json
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                syms = []
                syms.extend(config.get("nifty50", []))
                syms.extend(config.get("midcap", []))
                syms.extend(config.get("indices", []))
                # remove duplicates preserving order
                seen = set()
                out = []
                for s in syms:
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                return out
        except Exception as e:
            logger.exception("Failed to load config symbols: %s", e)
            return []

    def fetch_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch single symbol with cache and retry.
        """
        path = self._cache_path(symbol)
        # try cache first
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                if not df.empty:
                    return df
            except Exception:
                logger.debug("Cache read failed for %s, refetching", symbol)

        # fetch with retry
        attempts = 3
        for attempt in range(attempts):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval, actions=False)
                if df is None or df.empty:
                    raise ValueError("No data returned")
                # normalize index and save
                df = df.rename_axis("Date").reset_index()
                df.to_csv(path, index=False)
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                return df
            except Exception as e:
                wait = 1 + attempt * 2
                logger.warning("Fetch attempt %d failed for %s: %s — retrying in %ss", attempt + 1, symbol, e, wait)
                time.sleep(wait)
        logger.error("All fetch attempts failed for %s — returning empty frame", symbol)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def fetch_batch(self, symbols, period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Attempt bulk download using yfinance.download, write caches.
        Fallback: threaded fetch_history per symbol.
        """
        out = {}
        symbols = list(symbols)
        if not symbols:
            return out

        try:
            logger.info("Attempting bulk download for %d symbols", len(symbols))
            # yfinance allows space-separated tickers
            joined = " ".join(symbols)
            df_all = yf.download(tickers=joined, period=period, interval=interval, group_by="ticker", threads=True, auto_adjust=False, progress=False)
            # If MultiIndex columns -> multiple tickers
            if hasattr(df_all.columns, "levels") and len(df_all.columns.levels) > 0:
                for sym in symbols:
                    try:
                        sub = df_all[sym].dropna(how="all")
                        if sub.empty:
                            logger.warning("Bulk download gave empty for %s; fallback to per-symbol", sym)
                            out[sym] = self.fetch_history(sym, period=period, interval=interval)
                            continue
                        # ensure date index, save cache
                        sub = sub.rename_axis("Date").reset_index().set_index("Date")
                        sub.to_csv(self._cache_path(sym), index=True)
                        out[sym] = sub
                    except Exception:
                        logger.exception("Error reading bulk data for %s; falling back", sym)
                        out[sym] = self.fetch_history(sym, period=period, interval=interval)
            else:
                # single ticker result (or unknown structure)
                # Apply the same df to each symbol conservatively (likely only one symbol requested)
                for sym in symbols:
                    out[sym] = df_all.copy()
            logger.info("Bulk download completed")
            return out
        except Exception as e:
            logger.warning("Bulk download failed (%s). Falling back to threaded per-symbol fetch.", e)

        # Threaded fallback
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(self.fetch_history, s, period, interval): s for s in symbols}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    out[s] = fut.result()
                except Exception:
                    logger.exception("Threaded fetch failed for %s", s)
                    out[s] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        return out
