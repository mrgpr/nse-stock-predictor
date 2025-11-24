#!/usr/bin/env python3
"""
Main CLI to run predictions.
Modes: daily, weekly, monthly
This script orchestrates data fetch -> TA -> scoring -> report -> email (optional)
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

def run(mode: str):
    logger.info("Starting Indian Stock Predictor - mode=%s", mode)
    # load symbols
    config_path = ROOT / "config" / "stocks_list.json"

    fetcher = DataFetcher(config_path=config_path, historical_path=ROOT / "data" / "historical")
    analyzer = TechnicalAnalyzer()
    screener = StockScreener()
    reports = ReportGenerator(root_reports=ROOT / "reports")

    # Determine stocks to analyze
    symbols = fetcher.list_all_symbols()
    logger.info("Symbols count: %d", len(symbols))

    # Fetch data (batch)
    hist_data = fetcher.fetch_batch(symbols, period="1y", interval="1d")  # 1 year daily
    logger.info("Downloaded historical data for %d symbols", len(hist_data))

    # Compute indicators and signals
    ta_results = {}
    for sym, df in hist_data.items():
        try:
            ta_df = analyzer.add_indicators(df.copy())
            signals = analyzer.generate_signals(ta_df)
            ta_results[sym] = {"df": ta_df, "signals": signals}
        except Exception as e:
            logger.exception("Failed to compute TA for %s: %s", sym, e)

    # Score & rank
    scoring_results = screener.score_universe(ta_results, mode=mode)

    # Generate report
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    report_meta = reports.generate_report(scoring_results, mode=mode, timestamp=timestamp)

    logger.info("Report generated: %s", report_meta["report_md"])
    logger.info("CSV saved: %s", report_meta["report_csv"])

    # Optional: Send email if running in CI (env vars present) or prompt interactively
    try:
        # ReportGenerator has a helper to send email, which will check for GitHub Actions secrets or prompt locally
        reports.send_email_with_report(report_meta, top_n=5)
        logger.info("Email send attempted (see logs).")
    except Exception as e:
        logger.exception("Email sending failed: %s", e)

    # Create GitHub issue if GITHUB_TOKEN is available (useful in Actions)
    reports.create_github_issue_if_ci(report_meta, mode=mode)

    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Stock Predictor - run daily/weekly/monthly scans")
    parser.add_argument("--mode", choices=["daily", "weekly", "monthly"], default="daily", help="Mode to run")
    args = parser.parse_args()
    run(args.mode)
