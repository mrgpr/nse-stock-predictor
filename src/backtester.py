"""
backtester.py
- A very small backtest to evaluate strategy picks produced by screener/model.
- Assumes you have price series (Close) and picks with date, symbol, recommendation, predicted return or target.
- This is illustrative and not a production engine.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger("backtester")
logger.setLevel(logging.INFO)


def run_simple_long_backtest(price_series: pd.Series, entry_dates: List[pd.Timestamp], hold_days: int = 5):
    """
    price_series: pd.Series indexed by Timestamp (Close)
    entry_dates: list of entry timestamps
    Returns simple cumulative return assuming buy at next open/close and exit after hold_days.
    """
    returns = []
    for d in entry_dates:
        try:
            entry_idx = price_series.index.get_loc(d, method="nearest")
            entry_price = price_series.iloc[entry_idx]
            exit_idx = min(entry_idx + hold_days, len(price_series) - 1)
            exit_price = price_series.iloc[exit_idx]
            rtn = (exit_price / entry_price) - 1.0
            returns.append(rtn)
        except Exception:
            continue
    if not returns:
        return {"cagr": 0.0, "total_return": 0.0, "trades": 0}
    total = np.prod([1 + r for r in returns]) - 1
    # approximate CAGR assuming one trade per hold_days (very rough)
    years = (len(price_series) / 252)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else total
    return {"cagr": cagr, "total_return": total, "trades": len(returns), "avg_return": np.mean(returns)}
