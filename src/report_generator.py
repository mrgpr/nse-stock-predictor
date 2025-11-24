"""
report_generator.py
- Generates Markdown and HTML reports and CSV exports
- Sends email via Gmail SMTP (interactive locally or using GitHub secrets in CI)
- Creates GitHub Issue if running in GitHub Actions
- Updated to assign created issues to repo owner when available (to trigger notifications)
"""
import logging
from pathlib import Path
from datetime import datetime
import csv
import os
import smtplib
from email.message import EmailMessage
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr
import json
import sys
import requests

logger = logging.getLogger("report_generator")
logger.setLevel(logging.INFO)


class ReportGenerator:
    def __init__(self, root_reports: Path):
        self.root_reports = Path(root_reports)
        self.root_reports.mkdir(parents=True, exist_ok=True)

    def _ensure_dir(self, mode, timestamp):
        folder = self.root_reports / mode / timestamp
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def generate_report(self, scoring_results: dict, mode="daily", timestamp=None):
        """
        Creates:
          - Markdown report
          - HTML report
          - CSV export
        Returns paths and metadata
        """
        timestamp = timestamp or datetime.utcnow().strftime("%Y-%m-%d")
        folder = self._ensure_dir(mode, timestamp)

        top = scoring_results.get("top", [])
        all_items = scoring_results.get("all", [])

        # CSV
        csv_path = folder / f"{mode}_predictions_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "symbol", "score", "last_price", "target", "stop_loss", "recommendation", "rsi", "macd_signal", "trend", "volume", "vol_avg_20", "bb_pos"])
            for i, item in enumerate(top, start=1):
                s = item["signals"]
                writer.writerow([
                    i,
                    item["symbol"],
                    item["score"],
                    item.get("last_price"),
                    item.get("target"),
                    item.get("stop_loss"),
                    s.get("recommendation"),
                    s.get("rsi"),
                    s.get("macd_signal"),
                    s.get("trend"),
                    s.get("volume"),
                    s.get("vol_avg_20"),
                    s.get("bb_pos")
                ])

        # Markdown
        md_path = folder / f"{mode}_report_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 📈 {mode.capitalize()} Stock Predictions - {timestamp}\n\n")
            f.write(f"## 🎯 Top {len(top)} Stocks\n\n")
            for idx, item in enumerate(top, start=1):
                s = item["signals"]
                f.write(f"### 🥇 #{idx} - {item['symbol']}\n")
                f.write(f"- **Current Price:** ₹{item.get('last_price')}\n")
                # predicted return = (target/price -1)
                ret = None
                if item.get("last_price") and item.get("target"):
                    try:
                        ret = round((item["target"] / item["last_price"] - 1) * 100, 2)
                    except Exception:
                        ret = None
                f.write(f"- **Predicted Return:** {('+'+str(ret)+'%') if ret is not None else 'N/A'}\n")
                f.write(f"- **Confidence Score:** {item['score']}/100\n")
                f.write(f"- **Signal:** {s.get('recommendation')} \n")
                f.write(f"- **Target Price:** ₹{item.get('target')}\n")
                f.write(f"- **Stop Loss:** ₹{item.get('stop_loss')}\n")
                # Risk level heuristic
                risk = "Low"
                if item['score'] < 40:
                    risk = "High"
                elif item['score'] < 65:
                    risk = "Medium"
                f.write(f"- **Risk Level:** {risk}\n\n")
                f.write(f"**Technical Indicators:**\n\n")
                f.write(f"- RSI: {s.get('rsi')} ({s.get('rsi_signal')})\n")
                f.write(f"- MACD: {s.get('macd_signal')}\n")
                f.write(f"- Price: {'Above' if s.get('above_sma20') else 'Below'} 20-day & {'Above' if s.get('above_sma50') else 'Below'} 50-day MA\n")
                vol_pct = ''
                try:
                    vol = s.get('volume') or 0
                    vavg = s.get('vol_avg_20') or 0
                    if vavg:
                        vol_pct = f"{round((vol / vavg - 1) * 100, 1)}% above avg"
                except Exception:
                    vol_pct = ''
                f.write(f"- Volume: {vol_pct}\n")
                f.write(f"- Recommendation: Consider {'buying' if s.get('recommendation') in ('BUY', 'STRONG BUY') else 'avoiding'} based on technicals.\n\n")

            # Market overview stub
            f.write("## 📊 Market Overview\n")
            f.write("- Nifty 50: (fetched from Yahoo) see report CSV for index values\n\n")

        # HTML
        html_path = folder / f"{mode}_report_{timestamp}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._render_html(md_path.read_text(), title=f"{mode.capitalize()} Report {timestamp}"))

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
        # Very simple HTML wrapper and CSS to make the email pretty
        css = """
        body { font-family: Arial, Helvetica, sans-serif; padding: 20px; color: #111; }
        h1 { color: #0b5; }
        pre { background: #f6f6f6; padding: 10px; border-radius: 6px; }
        .stock { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 6px; }
        .buy { color: green; }
        .sell { color: red; }
        """
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{title}</title><style>{css}</style></head>
<body><h1>{title}</h1><div><pre>{markdown_text}</pre></div></body></html>"""
        return html

    # def send_email_with_report(self, report_meta: dict, top_n: int = 5):
    #     """
    #     Sends an HTML email with top N picks and attaches CSV.
    #     If running in GitHub Actions, expects EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO in env (set as secrets).
    #     Locally, prompts user for credentials interactively to avoid storing secrets.
    #     """
    #     # Read CSV and HTML
    #     html_path = report_meta["report_html"]
    #     csv_path = report_meta["report_csv"]

    #     # Compose email content
    #     with open(report_meta["report_md"], "r", encoding="utf-8") as f:
    #         md = f.read()
    #     html_content = self._render_html(md, title=f"Stock Picks {report_meta['timestamp']}")

    #     # Acquire credentials
    #     email_user = os.environ.get("EMAIL_USERNAME")
    #     email_pass = os.environ.get("EMAIL_PASSWORD")
    #     email_to = os.environ.get("EMAIL_TO")

    #     interactive = False
    #     if not (email_user and email_pass and email_to):
    #         # Interactive prompt for local runs
    #         interactive = True
    #         print("Email credentials not found in environment. For local testing, enter Gmail credentials (App Password recommended).")
    #         email_user = input("Gmail address (from): ").strip()
    #         email_to = input("Recipient email (to): ").strip()
    #         import getpass
    #         email_pass = getpass.getpass("Gmail App Password (16 chars): ")

    #     # Build email
    #     msg = EmailMessage()
    #     msg["Subject"] = f"🔔 {report_meta.get('timestamp')} - {report_meta.get('folder').split('/')[-1]} - Stock Picks"
    #     msg["From"] = formataddr(("Indian Stock Predictor", email_user))
    #     msg["To"] = email_to
    #     msg.set_content("This email contains HTML content. If you see this message, your email client may not support HTML.")
    #     msg.add_alternative(html_content, subtype="html")

    #     # Attach CSV
    #     with open(csv_path, "rb") as f:
    #         part = MIMEBase("application", "octet-stream")
    #         part.set_payload(f.read())
    #         encoders.encode_base64(part)
    #         part.add_header("Content-Disposition", f"attachment; filename={Path(csv_path).name}")
    #         msg.attach(part)

    #     # Send via Gmail SMTP
    #     try:
    #         # Gmail SSL
    #         with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
    #             server.login(email_user, email_pass)
    #             server.send_message(msg)
    #             logger.info("Email sent to %s", email_to)
    #     except Exception as e:
    #         logger.exception("Failed to send email: %s", e)
    #         if interactive:
    #             print("Email sending failed. Check your app password and network.")

    def send_email_with_report(self, report_meta: dict, top_n: int = 5):
        """
        Sends an HTML email with top N picks and attaches CSV.
        Behavior:
          - If running inside GitHub Actions and EMAIL_* env vars are NOT set, do NOT prompt (skip email).
          - Locally (not in CI), prompt interactively if EMAIL_* are not set.
        """
        # Read CSV and HTML
        html_path = report_meta["report_html"]
        csv_path = report_meta["report_csv"]

        # Compose email content
        try:
            with open(report_meta["report_md"], "r", encoding="utf-8") as f:
                md = f.read()
        except Exception as e:
            logger.exception("Failed reading markdown report: %s", e)
            md = ""
        html_content = self._render_html(md, title=f"Stock Picks {report_meta.get('timestamp')}")

        # Acquire credentials from environment
        email_user = os.environ.get("EMAIL_USERNAME")
        email_pass = os.environ.get("EMAIL_PASSWORD")
        email_to = os.environ.get("EMAIL_TO")

        running_ci = os.environ.get("GITHUB_ACTIONS") == "true"

        # If in CI and credentials missing, do not prompt: log and return gracefully
        if running_ci and not (email_user and email_pass and email_to):
            logger.info(
                "Running in CI and email credentials not found. Skipping email send. "
                "To enable emails from Actions, set repository secrets: EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO."
            )
            return

        # If not in CI and credentials missing, prompt interactively
        interactive = False
        if not (email_user and email_pass and email_to):
            interactive = True
            try:
                print("Email credentials not found in environment. For local testing, enter Gmail credentials (App Password recommended).")
                email_user = input("Gmail address (from): ").strip()
                email_to = input("Recipient email (to): ").strip()
                import getpass
                email_pass = getpass.getpass("Gmail App Password (16 chars): ")
            except Exception as e:
                logger.exception("Interactive prompt failed or not available: %s", e)
                # Do not raise in non-interactive environments
                return

        if not (email_user and email_pass and email_to):
            logger.warning("Email credentials incomplete; skipping email.")
            return

        # Build email
        msg = EmailMessage()
        msg["Subject"] = f"🔔 {report_meta.get('timestamp')} - {Path(report_meta.get('folder')).name} - Stock Picks"
        msg["From"] = formataddr(("Indian Stock Predictor", email_user))
        msg["To"] = email_to
        msg.set_content("This email contains HTML content. If you see this message, your email client may not support HTML.")
        msg.add_alternative(html_content, subtype="html")

        # Attach CSV (if present)
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={Path(csv_path).name}")
                msg.attach(part)
        except Exception as e:
            logger.exception("Failed to attach CSV: %s", e)

        # Send via Gmail SMTP
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
                server.login(email_user, email_pass)
                server.send_message(msg)
                logger.info("Email sent to %s", email_to)
        except Exception as e:
            logger.exception("Failed to send email: %s", e)
            if interactive:
                print("Email sending failed. Check your app password and network.")


    def create_github_issue_if_ci(self, report_meta: dict, mode="daily"):
        """
        If running inside GitHub Actions, create an issue summarizing the top picks.
        Uses GITHUB_TOKEN and github context env variables in actions.
        Assigns the issue to the repository owner (GITHUB_REPOSITORY_OWNER) if available to trigger notifications.
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

        # Build top picks summary
        top = report_meta.get("top", [])[:10]
        body_lines = [f"Automated **{mode}** picks for **{report_meta.get('timestamp')}**\n"]
        for i, item in enumerate(top, start=1):
            s = item["signals"]
            line = f"{i}. **{item['symbol']}** — Score: {item['score']} — {s.get('recommendation')} — Price: ₹{item.get('last_price')} — Target: ₹{item.get('target')}"
            body_lines.append(line)

        # Link to report files within the repository if they exist in the workspace
        folder = Path(report_meta.get("folder", ""))
        if folder.exists():
            # Compute a repo URL to the files assuming reports are committed to the default branch after workflow
            # This uses GITHUB_SERVER_URL and GITHUB_REPOSITORY to point at the path
            base_url = f"{server}/{repo}/blob/HEAD/{folder.as_posix()}"
            md_name = Path(report_meta.get("report_md")).name
            csv_name = Path(report_meta.get("report_csv")).name
            html_name = Path(report_meta.get("report_html")).name
            body_lines.append("\n**Reports:**")
            body_lines.append(f"- [Markdown report]({base_url}/{md_name})")
            body_lines.append(f"- [HTML report]({base_url}/{html_name})")
            body_lines.append(f"- [CSV export]({base_url}/{csv_name})")
        else:
            body_lines.append("\n(Report files are saved in the Action runner workspace. They will be committed to the repo if configured.)")

        body = "\n".join(body_lines)

        # Try to assign to repo owner to get explicit notification
        owner = os.environ.get("GITHUB_REPOSITORY_OWNER")
        assignees = [owner] if owner else []

        payload = {
            "title": f"[Auto] {mode.capitalize()} Picks - {report_meta.get('timestamp')}",
            "body": body,
            "labels": ["automation", f"{mode}-report"],
            "assignees": assignees
        }

        url = f"https://api.github.com/repos/{repo}/issues"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            if r.status_code in (200, 201):
                logger.info("GitHub issue created (%s).", r.json().get("html_url"))
            else:
                logger.warning("GitHub issue creation failed: %s %s", r.status_code, r.text)
        except Exception as e:
            logger.exception("Error creating GitHub issue: %s", e)
