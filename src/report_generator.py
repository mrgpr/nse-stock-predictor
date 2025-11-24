"""
report_generator.py
- Generates improved Markdown and HTML reports and CSV exports
- Adds:
  * Summary (counts, avg expected return)
  * Grouped sections (STRONG BUY, BUY, HOLD, SELL)
  * Target ranges, expected return %, volatility, sector, risk, short rationale
- CI-safe email sending:
  * If running in GitHub Actions and EMAIL_* secrets missing -> skip email (no interactive prompt)
  * Locally: will prompt interactively for Gmail + App Password if EMAIL_* env vars absent
- Creates GitHub Issue in CI (assigns repo owner if available) with links to reports
"""
from pathlib import Path
from datetime import datetime
import logging
import csv
import os
import smtplib
from email.message import EmailMessage
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr
import requests
from typing import Dict, Any

logger = logging.getLogger("report_generator")
logger.setLevel(logging.INFO)

# Helper formatters
def fmt_price(p):
    try:
        return f"₹{p:,.2f}"
    except Exception:
        return "N/A"


def fmt_pct(x):
    try:
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.2f}%"
    except Exception:
        return "N/A"


class ReportGenerator:
    def __init__(self, root_reports: Path):
        self.root_reports = Path(root_reports)
        self.root_reports.mkdir(parents=True, exist_ok=True)

    def _ensure_dir(self, mode: str, timestamp: str):
        folder = self.root_reports / mode / timestamp
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def generate_report(self, scoring_results: dict, mode: str = "daily", timestamp: str = None) -> dict:
        """
        Create:
          - Markdown report with summary and grouped sections
          - HTML wrapper
          - CSV export (flat)
        Returns metadata dict with paths and lists.
        """
        timestamp = timestamp or datetime.utcnow().strftime("%Y-%m-%d")
        folder = self._ensure_dir(mode, timestamp)

        top = scoring_results.get("top", [])
        all_items = scoring_results.get("all", [])

        # CSV export (flat)
        csv_path = folder / f"{mode}_predictions_{timestamp}.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "rank", "symbol", "score", "last_price", "target", "target_low", "target_high",
                    "expected_return_pct", "volatility", "risk", "sector", "recommendation", "rationale"
                ])
                for i, item in enumerate(all_items, start=1):
                    s = item.get("signals", {})
                    writer.writerow([
                        i,
                        item.get("symbol"),
                        item.get("score"),
                        item.get("last_price"),
                        item.get("target"),
                        item.get("target_low"),
                        item.get("target_high"),
                        item.get("expected_return_pct"),
                        item.get("volatility"),
                        item.get("risk"),
                        item.get("sector"),
                        s.get("recommendation") if isinstance(s, dict) else None,
                        item.get("rationale")
                    ])
        except Exception:
            logger.exception("Failed to write CSV to %s", csv_path)

        # Build grouped lists by recommendation
        groups = {"STRONG BUY": [], "BUY": [], "HOLD": [], "SELL": []}
        for item in all_items:
            rec = item.get("signals", {}).get("recommendation", "HOLD")
            if rec == "STRONG BUY":
                groups["STRONG BUY"].append(item)
            elif rec == "BUY":
                groups["BUY"].append(item)
            elif rec == "SELL":
                groups["SELL"].append(item)
            else:
                groups["HOLD"].append(item)

        # Summary lines
        def avg_expected(items):
            vals = [it.get("expected_return_pct") for it in items if it.get("expected_return_pct") is not None]
            if not vals:
                return None
            return round(sum(vals) / len(vals), 2)

        summary_lines = [
            f"**Date:** {timestamp}",
            f"- Strong Buy: {len(groups['STRONG BUY'])}",
            f"- Buy: {len(groups['BUY'])}",
            f"- Hold: {len(groups['HOLD'])}",
            f"- Sell: {len(groups['SELL'])}"
        ]
        overall_avg = avg_expected(all_items)
        if overall_avg is not None:
            summary_lines.append(f"- Avg Expected Return: {fmt_pct(overall_avg)}")

        # Markdown generation
        md_path = folder / f"{mode}_report_{timestamp}.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# 📈 {mode.capitalize()} Stock Predictions — {timestamp}\n\n")
                f.write("## Summary\n")
                for line in summary_lines:
                    f.write(f"- {line}\n")
                f.write("\n---\n\n")

                # Helper to write a table-like block
                def write_section(title: str, items: list):
                    f.write(f"## {title}\n\n")
                    if not items:
                        f.write("_No picks in this category_\n\n")
                        return
                    # Header
                    f.write("| Stock | Price | Target (range) | Return | Risk | Sector | Notes |\n")
                    f.write("|---|---:|---|---:|---|---|---|\n")
                    for it in items:
                        sym = it.get("symbol")
                        price = it.get("last_price")
                        tgt = it.get("target")
                        t_low = it.get("target_low")
                        t_high = it.get("target_high")
                        exp = it.get("expected_return_pct")
                        risk = it.get("risk")
                        sector = it.get("sector")
                        rationale = it.get("rationale") or ""
                        price_s = fmt_price(price) if price is not None else "N/A"
                        if t_low is not None and t_high is not None:
                            tgt_s = f"{fmt_price(t_low)} – {fmt_price(t_high)}"
                        elif tgt is not None:
                            tgt_s = fmt_price(tgt)
                        else:
                            tgt_s = "N/A"
                        exp_s = fmt_pct(exp) if exp is not None else "N/A"
                        f.write(f"| **{sym}** | {price_s} | {tgt_s} | {exp_s} | {risk} | {sector} | {rationale} |\n")
                    f.write("\n")

                # Order: Strong Buy, Buy, Hold, Sell
                write_section("🔥 Strong Buy", groups["STRONG BUY"])
                write_section("🟩 Buy", groups["BUY"])
                write_section("🟨 Hold", groups["HOLD"])
                write_section("🔴 Sell", groups["SELL"])

                # Appendix with top details
                f.write("---\n\n")
                f.write("## 🧾 Detailed Top Picks (Top 20)\n\n")
                for idx, it in enumerate(all_items[:20], start=1):
                    f.write(f"### {idx}. {it.get('symbol')} — Score: {it.get('score')}\n")
                    f.write(f"- **Price:** {fmt_price(it.get('last_price'))}\n")
                    if it.get('target_low') is not None and it.get('target_high') is not None:
                        f.write(f"- **Target Range:** {fmt_price(it.get('target_low'))} – {fmt_price(it.get('target_high'))}\n")
                    elif it.get('target') is not None:
                        f.write(f"- **Target:** {fmt_price(it.get('target'))}\n")
                    f.write(f"- **Expected Return:** {fmt_pct(it.get('expected_return_pct'))}\n")
                    f.write(f"- **Volatility (20d):** {it.get('volatility')}\n")
                    f.write(f"- **Risk:** {it.get('risk')}\n")
                    f.write(f"- **Sector:** {it.get('sector')}\n")
                    f.write(f"- **Rationale:** {it.get('rationale')}\n\n")
        except Exception:
            logger.exception("Failed to write markdown report to %s", md_path)

        # HTML wrapper (safe showing of markdown)
        html_path = folder / f"{mode}_report_{timestamp}.html"
        try:
            with open(md_path, "r", encoding="utf-8") as fh:
                md_text = fh.read()
            html_content = self._render_html(md_text, title=f"{mode.capitalize()} Report {timestamp}")
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(html_content)
        except Exception:
            logger.exception("Failed to write HTML report to %s", html_path)

        return {
            "report_md": str(md_path),
            "report_html": str(html_path),
            "report_csv": str(csv_path),
            "folder": str(folder),
            "top": top,
            "all": all_items,
            "timestamp": timestamp
        }

    def _render_html(self, markdown_text: str, title: str = "Report") -> str:
        css = """
        body { font-family: Arial, Helvetica, sans-serif; padding: 20px; color: #111; }
        h1 { color: #0b5; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        pre { white-space: pre-wrap; font-family: monospace; }
        """
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{title}</title><style>{css}</style></head>
<body><h1>{title}</h1><div><pre>{markdown_text}</pre></div></body></html>"""
        return html

    def send_email_with_report(self, report_meta: dict, top_n: int = 5):
        """
        Send HTML email with CSV attachment.

        Behavior:
          - If running in GitHub Actions and EMAIL_* env vars are missing, SKIP email and log an instruction.
          - Locally, prompt interactively for credentials if EMAIL_* env vars are absent.
        """
        # Paths
        md_path = Path(report_meta.get("report_md", ""))
        html_path = Path(report_meta.get("report_html", ""))
        csv_path = Path(report_meta.get("report_csv", ""))

        # compose HTML body from markdown for email (reuse renderer)
        md_text = ""
        try:
            if md_path.exists():
                md_text = md_path.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed reading markdown for email body: %s", md_path)

        html_body = self._render_html(md_text, title=f"Stock Picks {report_meta.get('timestamp')}")

        # Get credentials from environment
        email_user = os.environ.get("EMAIL_USERNAME")
        email_pass = os.environ.get("EMAIL_PASSWORD")
        email_to = os.environ.get("EMAIL_TO")

        running_ci = os.environ.get("GITHUB_ACTIONS") == "true"

        # CI behavior: do not prompt if missing — just skip email
        if running_ci and not (email_user and email_pass and email_to):
            logger.info(
                "Running in CI and EMAIL_* secrets not set. Skipping email send. "
                "To enable emails from Actions, set repository secrets: EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO."
            )
            return

        # Local interactive behavior: prompt if missing
        interactive = False
        if not (email_user and email_pass and email_to):
            interactive = True
            try:
                print("Email credentials not found in environment. For local testing, enter Gmail credentials (App Password recommended).")
                email_user = input("Gmail address (from): ").strip()
                email_to = input("Recipient email (to): ").strip()
                import getpass
                email_pass = getpass.getpass("Gmail App Password (16 chars): ")
            except Exception:
                logger.exception("Interactive prompt failed or not available; skipping email.")
                return

        if not (email_user and email_pass and email_to):
            logger.warning("Email credentials incomplete; skipping email.")
            return

        # Build message
        msg = EmailMessage()
        msg["Subject"] = f"🔔 {report_meta.get('timestamp')} - {Path(report_meta.get('folder')).name} - Stock Picks"
        msg["From"] = formataddr(("Indian Stock Predictor", email_user))
        msg["To"] = email_to
        msg.set_content("This email contains HTML content. If you see this message, your client may not support HTML.")
        msg.add_alternative(html_body, subtype="html")

        # Attach CSV if exists
        try:
            if csv_path.exists():
                with open(csv_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={csv_path.name}")
                    msg.attach(part)
        except Exception:
            logger.exception("Failed attaching CSV: %s", csv_path)

        # Send using Gmail SMTP SSL
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
                server.login(email_user, email_pass)
                server.send_message(msg)
                logger.info("Email sent to %s", email_to)
        except Exception:
            logger.exception("Failed to send email. If running in Actions, ensure EMAIL_USERNAME and EMAIL_PASSWORD are set as secrets.")

        def create_github_issue_if_ci(self, report_meta: dict, mode: str = "daily"):
        """
        If running in GitHub Actions, create an issue using the generated Markdown report as the body.
        Uses GITHUB_TOKEN and GITHUB_REPOSITORY (both available in Actions).
        Assigns issue to GITHUB_REPOSITORY_OWNER if available to trigger notifications.
        """
        if os.environ.get("GITHUB_ACTIONS") != "true":
            logger.debug("Not running in GitHub Actions; skipping issue creation.")
            return

        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY")  # owner/repo
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        if not token or not repo:
            logger.warning("GITHUB_TOKEN or GITHUB_REPOSITORY missing; cannot create issue.")
            return

        md_path = Path(report_meta.get("report_md", ""))
        html_path = Path(report_meta.get("report_html", ""))
        csv_path = Path(report_meta.get("report_csv", ""))
        timestamp = report_meta.get("timestamp", "")

        # Read markdown report content (prefer the generated Markdown for the issue body)
        body_md = ""
        try:
            if md_path.exists():
                body_md = md_path.read_text(encoding="utf-8")
            else:
                # fallback: generate a short summary from report_meta['top']
                top = report_meta.get("top", [])[:10]
                lines = [f"Automated **{mode}** picks for **{timestamp}**\n"]
                for i, item in enumerate(top, start=1):
                    s = item.get("signals", {})
                    lines.append(f"{i}. **{item.get('symbol')}** — Score: {item.get('score')} — {s.get('recommendation')} — Price: ₹{item.get('last_price')} — Target: {fmt_price(item.get('target')) if item.get('target') else 'N/A'}")
                body_md = "\n".join(lines)
        except Exception:
            logger.exception("Failed to read markdown report; falling back to short summary.")
            top = report_meta.get("top", [])[:10]
            lines = [f"Automated **{mode}** picks for **{timestamp}**\n"]
            for i, item in enumerate(top, start=1):
                s = item.get("signals", {})
                lines.append(f"{i}. **{item.get('symbol')}** — Score: {item.get('score')} — {s.get('recommendation')} — Price: ₹{item.get('last_price')} — Target: {fmt_price(item.get('target')) if item.get('target') else 'N/A'}")
            body_md = "\n".join(lines)

        # Prepend a small header/summary (counts)
        try:
            all_items = report_meta.get("all", [])
            total = len(all_items)
            counts = {
                "STRONG BUY": sum(1 for it in all_items if (it.get("signals") or {}).get("recommendation") == "STRONG BUY"),
                "BUY": sum(1 for it in all_items if (it.get("signals") or {}).get("recommendation") == "BUY"),
                "HOLD": sum(1 for it in all_items if (it.get("signals") or {}).get("recommendation") == "HOLD"),
                "SELL": sum(1 for it in all_items if (it.get("signals") or {}).get("recommendation") == "SELL"),
            }
            header = [f"Automated **{mode}** picks for **{timestamp}**", "", f"- Total scanned: {total}", f"- Strong Buy: {counts['STRONG BUY']}", f"- Buy: {counts['BUY']}", f"- Hold: {counts['HOLD']}", f"- Sell: {counts['SELL']}", ""]
            full_body = "\n".join(header) + body_md
        except Exception:
            full_body = f"Automated **{mode}** picks for **{timestamp}**\n\n" + body_md

        # Append links to reports in repo (if folder exists)
        try:
            folder = Path(report_meta.get("folder", ""))
            if folder.exists():
                base_url = f"{server}/{repo}/blob/HEAD/{folder.as_posix()}"
                md_name = md_path.name if md_path.exists() else None
                html_name = html_path.name if html_path.exists() else None
                csv_name = csv_path.name if csv_path.exists() else None
                link_lines = ["\n\n**Reports:**"]
                if md_name:
                    link_lines.append(f"- [Markdown report]({base_url}/{md_name})")
                if html_name:
                    link_lines.append(f"- [HTML report]({base_url}/{html_name})")
                if csv_name:
                    link_lines.append(f"- [CSV export]({base_url}/{csv_name})")
                full_body += "\n".join(link_lines)
            else:
                full_body += "\n\n(Report files are in the runner workspace and will be committed to the repo if configured.)"
        except Exception:
            logger.exception("Failed to append report links to issue body.")

        # Truncate if too long (safety)
        MAX_LEN = 60000  # conservative
        if len(full_body) > MAX_LEN:
            full_body = full_body[:MAX_LEN - 200] + "\n\n*(report truncated)*\n"

        # Assign to repo owner if possible
        owner = os.environ.get("GITHUB_REPOSITORY_OWNER")
        assignees = [owner] if owner else []

        payload = {
            "title": f"[Auto] {mode.capitalize()} Picks - {timestamp}",
            "body": full_body,
            "labels": ["automation", f"{mode}-report"],
            "assignees": assignees
        }

        url = f"https://api.github.com/repos/{repo}/issues"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code in (200, 201):
                logger.info("GitHub issue created (%s).", r.json().get("html_url"))
            else:
                logger.warning("GitHub issue creation failed: %s %s", r.status_code, r.text)
        except Exception:
            logger.exception("Error creating GitHub issue")
