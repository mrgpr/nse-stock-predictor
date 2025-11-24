#!/usr/bin/env python3
"""
Main CLI to run predictions.
Modes: daily, weekly, monthly, quarterly, biquarterly, yearly
This script orchestrates data fetch -> TA -> scoring -> report -> email (optional)
It selects historical periods appropriate to each mode so the screener's volatility windows and targets are meaningful.
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.data_fetcher import DataFetcher
from src.technical_analysis import TechnicalAnalyzer
from src.stock_screener import StockScreener
from src.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

ROOT = Path(__file__).parent

# Formatting helpers
def fmt_price(p):
    try:
        return f"₹{float(p):,.2f}"
    except Exception:
        return "N/A"

def fmt_pct(x):
    try:
        sign = "+" if x >= 0 else ""
        return f"{sign}{float(x):.2f}%"
    except Exception:
        return "N/A"

def print_console_report(report_meta):
    """
    Nicely print the report to console using report_meta produced by ReportGenerator.
    """
    all_items = report_meta.get("all", [])
    if not all_items:
        print("No items to display.")
        return

    # Group
    groups = {"STRONG BUY": [], "BUY": [], "HOLD": [], "SELL": []}
    for it in all_items:
        rec = (it.get("signals") or {}).get("recommendation", "HOLD")
        if rec == "STRONG BUY":
            groups["STRONG BUY"].append(it)
        elif rec == "BUY":
            groups["BUY"].append(it)
        elif rec == "SELL":
            groups["SELL"].append(it)
        else:
            groups["HOLD"].append(it)

    timestamp = report_meta.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d"))
    print()
    print(f"📅 {report_meta.get('folder').split('/')[-1].replace('_',' ')} — {timestamp}")
    # Summary
    def avg_expected(items):
        vals = [it.get("expected_return_pct") for it in items if it.get("expected_return_pct") is not None]
        return round(sum(vals)/len(vals),2) if vals else None

    total_items = len(all_items)
    print("Summary:")
    print(f"- Total scanned: {total_items}")
    print(f"- Strong Buy: {len(groups['STRONG BUY'])} | Buy: {len(groups['BUY'])} | Hold: {len(groups['HOLD'])} | Sell: {len(groups['SELL'])}")
    overall_avg = avg_expected(all_items)
    if overall_avg is not None:
        print(f"- Avg Expected Return: {fmt_pct(overall_avg)}")
    print()

    # Helper to print table-like rows
    def print_section(title, items):
        print(title)
        if not items:
            print("  (none)\n")
            return
        header = f"{'Stock':<16} {'Price':>12} {'Target (range)':>28} {'Return':>10} {'Risk':>8} {'Sector':>15} {'Notes':>30}"
        print(header)
        print("-" * len(header))
        for it in items:
            sym = it.get("symbol", "")
            price = fmt_price(it.get("last_price"))
            # target range
            t_low = it.get("target_low")
            t_high = it.get("target_high")
            tgt_s = "N/A"
            if t_low is not None and t_high is not None:
                tgt_s = f"{fmt_price(t_low)} – {fmt_price(t_high)}"
            elif it.get("target") is not None:
                tgt_s = fmt_price(it.get("target"))
            ret = fmt_pct(it.get("expected_return_pct")) if it.get("expected_return_pct") is not None else "N/A"
            risk = it.get("risk", "N/A")
            sector = it.get("sector", "N/A")
            rationale = it.get("rationale", "")
            # limit rationale length
            if rationale and len(rationale) > 60:
                rationale = rationale[:57] + "..."
            print(f"{sym:<16} {price:>12} {tgt_s:>28} {ret:>10} {risk:>8} {sector:>15} {rationale:>30}")
        print()

    # Print in preferred order
    print_section("🔥 Strong Buy", groups["STRONG BUY"])
    print_section("🟩 Buy", groups["BUY"])
    print_section("🟨 Hold", groups["HOLD"])
    print_section("🔴 Sell", groups["SELL"])


def run(mode: str):
    mode = mode.lower()
    allowed_modes = ["daily", "weekly", "monthly", "quarterly", "biquarterly", "yearly"]
    if mode not in allowed_modes:
        logger.error("Invalid mode '%s'. Allowed: %s", mode, ", ".join(allowed_modes))
        return

    logger.info("Starting Indian Stock Predictor - mode=%s", mode)

    # Choose history lengths (period strings understood by yfinance)
    # We'll pick conservative choices:
    # - daily/weekly: 1 year of daily data
    # - monthly/quarterly: 3 years (gives enough history for 60/120-day windows)
    # - biquarterly/yearly: 5 years
    period_map = {
        "daily": "1y",
        "weekly": "1y",
        "monthly": "3y",
        "quarterly": "3y",
        "biquarterly": "5y",
        "yearly": "5y"
    }
    interval_map = {
        # Keep daily granularity; weekly/monthly analysis still benefits from daily history
        "daily": "1d",
        "weekly": "1d",
        "monthly": "1d",
        "quarterly": "1d",
        "biquarterly": "1d",
        "yearly": "1d"
    }

    period = period_map.get(mode, "1y")
    interval = interval_map.get(mode, "1d")

    # Initialize components
    config_path = ROOT / "config" / "stocks_list.json"
    fetcher = DataFetcher(config_path=config_path, historical_path=ROOT / "data" / "historical")
    analyzer = TechnicalAnalyzer()
    screener = StockScreener(config_path=config_path)
    reports = ReportGenerator(root_reports=ROOT / "reports")

    symbols = fetcher.list_all_symbols()
    logger.info("Symbols count: %d", len(symbols))

    # Fetch historical data appropriate for the mode
    logger.info("Downloading history for period=%s interval=%s", period, interval)
    hist_data = fetcher.fetch_batch(symbols, period=period, interval=interval)
    logger.info("Downloaded historical data for %d symbols", len(hist_data))

    # Compute indicators and signals
    ta_results = {}
    for sym, df in hist_data.items():
        try:
            if df is None or df.empty:
                logger.warning("Empty data for %s, skipping TA", sym)
                continue
            ta_df = analyzer.add_indicators(df.copy())
            signals = analyzer.generate_signals(ta_df)
            ta_results[sym] = {"df": ta_df, "signals": signals}
        except Exception as e:
            logger.exception("Failed to compute TA for %s: %s", sym, e)

    # Score & rank (screener uses mode to pick volatility window)
    scoring_results = screener.score_universe(ta_results, mode=mode)

    # Generate report (timestamped)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    report_meta = reports.generate_report(scoring_results, mode=mode, timestamp=timestamp)

    logger.info("Report generated: %s", report_meta["report_md"])
    logger.info("CSV saved: %s", report_meta["report_csv"])

    # Print console summary for immediate visibility
    try:
        print_console_report(report_meta)
    except Exception:
        logger.exception("Failed to print console report")

    # Attempt to send email if configured (CI-aware)
    try:
        reports.send_email_with_report(report_meta, top_n=5)
        logger.info("Email send attempted (see logs).")
    except Exception as e:
        logger.exception("Email sending failed: %s", e)

    # Create GitHub issue in CI if applicable (report body uses markdown)
    reports.create_github_issue_if_ci(report_meta, mode=mode)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Stock Predictor - run scans")
    parser.add_argument("--mode", choices=["daily", "weekly", "monthly", "quarterly", "biquarterly", "yearly"], default="daily", help="Mode to run")
    args = parser.parse_args()
    run(args.mode)
