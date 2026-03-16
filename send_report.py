"""
send_report.py — Email the nightly scan summary via Gmail
Reads results/ JSON files and sends a clean HTML email.

Setup (one-time):
  1. Go to myaccount.google.com → Security → 2-Step Verification → App passwords
  2. Create an app password for "Mail"
  3. Add to GitHub Secrets:
       GMAIL_USER   = your.email@gmail.com
       GMAIL_PASS   = your-16-char-app-password
       NOTIFY_EMAIL = recipient@email.com  (can be same as GMAIL_USER)
"""

import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def load(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def regime_color(regime: str) -> str:
    return {"BULL": "#16a34a", "NEUTRAL": "#d97706", "BEAR": "#dc2626"}.get(regime, "#6b7280")


def build_html(regime: dict, conviction: list, layer1: list, layer2: list) -> str:
    date_str   = datetime.utcnow().strftime("%d %b %Y")
    r          = regime.get("regime", "UNKNOWN")
    r_color    = regime_color(r)
    r_note     = regime.get("regime_note", "")
    sc         = regime.get("smallcap100") or {}
    n50        = regime.get("nifty50") or {}

    # ── Regime banner ──
    sc_line = ""
    if sc:
        sc_line = (f"Nifty SC100: {sc.get('last','?')} &nbsp;|&nbsp; "
                   f"1M: {sc.get('chg_1m_pct','?'):+.1f}% &nbsp;|&nbsp; "
                   f"3M: {sc.get('chg_3m_pct','?'):+.1f}%")

    # ── Conviction rows ──
    conviction_html = ""
    if conviction:
        rows = ""
        for s in conviction[:5]:
            t  = s.get("technical", {})
            pb = s.get("pre_breakout", {})
            rows += f"""
            <tr>
              <td style="padding:8px 12px;font-weight:600">{s['symbol'].replace('.NS','')}</td>
              <td style="padding:8px 12px">₹{s['price']}</td>
              <td style="padding:8px 12px">₹{s['market_cap_cr']:.0f} Cr</td>
              <td style="padding:8px 12px;font-weight:600;color:#16a34a">{s['combined_score']:.3f}</td>
              <td style="padding:8px 12px">{t.get('rsi','?')} RSI</td>
              <td style="padding:8px 12px">
                OBV {'✓' if t.get('obv_bullish') else '✗'} &nbsp;
                RS {t.get('rs_days','?')}d &nbsp;
                F-Score {pb.get('piotroski_score','?')} &nbsp;
                FCF {pb.get('fcf_yield_pct','N/A')}%
              </td>
              <td style="padding:8px 12px;color:#6b7280;font-size:12px">{s.get('price_stage','')}</td>
            </tr>"""
        conviction_html = f"""
        <h2 style="font-size:16px;margin:24px 0 8px;color:#111">★ Conviction List — Both Layers</h2>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:13px;background:#f9fafb;border-radius:8px;overflow:hidden">
          <thead>
            <tr style="background:#1e293b;color:#fff">
              <th style="padding:8px 12px;text-align:left">Symbol</th>
              <th style="padding:8px 12px;text-align:left">Price</th>
              <th style="padding:8px 12px;text-align:left">MCap</th>
              <th style="padding:8px 12px;text-align:left">Score</th>
              <th style="padding:8px 12px;text-align:left">RSI</th>
              <th style="padding:8px 12px;text-align:left">Signals</th>
              <th style="padding:8px 12px;text-align:left">Stage</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""
    else:
        conviction_html = """
        <div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;
                    padding:12px 16px;font-size:13px;color:#713f12;margin:16px 0">
          No conviction plays today — no overlap between Layer 1 and Layer 2.<br>
          Check Layer 2 watchlist for stocks building a thesis before the breakout.
        </div>"""

    # ── Layer 1 rows ──
    l1_rows = ""
    for s in layer1[:10]:
        fd = s.get("fundamentals", {})
        l1_rows += f"""
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:7px 10px;font-weight:500">{s['symbol'].replace('.NS','')}</td>
          <td style="padding:7px 10px">₹{s['last_close']}</td>
          <td style="padding:7px 10px">{s['composite_score']:.3f}</td>
          <td style="padding:7px 10px">{s.get('rsi','?')}</td>
          <td style="padding:7px 10px">{'✓' if s.get('obv_bullish') else '✗'}</td>
          <td style="padding:7px 10px">{s.get('rs_days','?')}d</td>
          <td style="padding:7px 10px">{s.get('price_chg_3m_pct',0):+.1f}%</td>
          <td style="padding:7px 10px">{fd.get('fcf_yield_pct','N/A')}%</td>
        </tr>"""

    l1_html = ""
    if l1_rows:
        l1_html = f"""
        <h2 style="font-size:16px;margin:24px 0 8px;color:#111">📈 Technical Breakouts (Layer 1)</h2>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:13px">
          <thead>
            <tr style="background:#e2e8f0;font-size:11px;color:#475569">
              <th style="padding:6px 10px;text-align:left">Symbol</th>
              <th style="padding:6px 10px;text-align:left">Price</th>
              <th style="padding:6px 10px;text-align:left">Score</th>
              <th style="padding:6px 10px;text-align:left">RSI</th>
              <th style="padding:6px 10px;text-align:left">OBV</th>
              <th style="padding:6px 10px;text-align:left">RS Days</th>
              <th style="padding:6px 10px;text-align:left">3M Ret</th>
              <th style="padding:6px 10px;text-align:left">FCF%</th>
            </tr>
          </thead>
          <tbody>{l1_rows}</tbody>
        </table>"""
    else:
        l1_html = """
        <div style="background:#f1f5f9;border-radius:8px;padding:12px 16px;
                    font-size:13px;color:#64748b;margin:16px 0">
          No breakout candidates today. Normal in bear/neutral market regimes.
        </div>"""

    # ── Layer 2 rows ──
    l2_rows = ""
    for s in layer2[:10]:
        p_score = s.get("piotroski", {}).get("piotroski_score")
        fcf_y   = s.get("fcf", {}).get("fcf_yield_pct")
        sc      = s.get("screener") or {}
        cats    = s.get("catalyst", {}).get("catalysts", [])
        cat_str = cats[0][:35] if cats else "–"
        grp     = s.get("group", {}).get("matched_group") or "–"
        roce    = sc.get("roce_pct")
        roce_display = f"{roce:.1f}%" if roce else ("N/A" if not fcf_y else f"FCF {fcf_y}%")
        l2_rows += f"""
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:7px 10px;font-weight:500">{s['symbol'].replace('.NS','')}</td>
          <td style="padding:7px 10px">₹{s['price']}</td>
          <td style="padding:7px 10px">₹{s['market_cap_cr']:.0f}Cr</td>
          <td style="padding:7px 10px">{s['composite_score']:.3f}</td>
          <td style="padding:7px 10px">{p_score if p_score is not None else 'N/A'}/9</td>
          <td style="padding:7px 10px">{roce_display}</td>
          <td style="padding:7px 10px;color:#6b7280;font-size:11px">{grp}</td>
          <td style="padding:7px 10px;font-size:11px;color:#374151">{cat_str}</td>
        </tr>"""

    l2_html = f"""
    <h2 style="font-size:16px;margin:24px 0 8px;color:#111">🔍 Pre-Breakout Watchlist (Layer 2)</h2>
    <table width="100%" cellpadding="0" cellspacing="0"
           style="border-collapse:collapse;font-size:13px">
      <thead>
        <tr style="background:#e2e8f0;font-size:11px;color:#475569">
          <th style="padding:6px 10px;text-align:left">Symbol</th>
          <th style="padding:6px 10px;text-align:left">Price</th>
          <th style="padding:6px 10px;text-align:left">MCap</th>
          <th style="padding:6px 10px;text-align:left">Score</th>
          <th style="padding:6px 10px;text-align:left">Piotroski</th>
          <th style="padding:6px 10px;text-align:left">ROCE/FCF</th>
          <th style="padding:6px 10px;text-align:left">Group</th>
          <th style="padding:6px 10px;text-align:left">Top Signal</th>
        </tr>
      </thead>
      <tbody>{l2_rows}</tbody>
    </table>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="max-width:700px;margin:0 auto;padding:20px">

    <!-- Header -->
    <div style="background:#0f172a;border-radius:12px 12px 0 0;padding:20px 24px">
      <div style="font-size:11px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase">
        Multibagger Discovery Engine
      </div>
      <div style="font-size:22px;font-weight:600;color:#fff;margin-top:4px">
        NSE Scan — {date_str}
      </div>
      <div style="font-size:12px;color:#64748b;margin-top:4px">
        Layer 1: Breakout + OBV + RS + FCF &nbsp;|&nbsp; Layer 2: Pre-Breakout + Piotroski
      </div>
    </div>

    <!-- Regime banner -->
    <div style="background:{r_color};padding:14px 24px">
      <span style="font-size:15px;font-weight:600;color:#fff">
        Market Regime: {r}
      </span>
      <span style="font-size:12px;color:rgba(255,255,255,0.85);margin-left:12px">
        {r_note}
      </span>
      <div style="font-size:12px;color:rgba(255,255,255,0.75);margin-top:4px">{sc_line}</div>
    </div>

    <!-- Content card -->
    <div style="background:#fff;border-radius:0 0 12px 12px;padding:20px 24px;
                box-shadow:0 1px 3px rgba(0,0,0,0.08)">

      {conviction_html}
      {l1_html}
      {l2_html}

      <!-- Footer -->
      <div style="margin-top:28px;padding-top:16px;border-top:1px solid #e5e7eb;
                  font-size:11px;color:#9ca3af">
        ⚠ Educational use only. Not financial advice. Do your own due diligence.<br>
        Generated by Multibagger Discovery Engine v3 — value-picks.blogspot.com methodology
      </div>
    </div>
  </div>
</body>
</html>"""


def send_via_gmail(html: str, subject: str, notify_email: str):
    """Send via Gmail SMTP using an App Password."""
    gmail_user = os.environ["GMAIL_USER"].strip()
    # Google displays app passwords with spaces/non-breaking spaces between groups
    # e.g. "xxxx xxxx xxxx xxxx" — strip ALL whitespace before SMTP login
    gmail_pass = "".join(os.environ["GMAIL_PASS"].split())

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Multibagger Engine <{gmail_user}>"
    msg["To"]      = notify_email
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_pass)
        server.sendmail(gmail_user, notify_email, msg.as_string())
    print(f"✅ Gmail sent to {notify_email}")


def send_via_sendgrid(html: str, subject: str, notify_email: str):
    """
    Send via SendGrid API — free tier, 100 emails/day.
    Requires only SENDGRID_API_KEY in GitHub Secrets.
    No Gmail App Password needed.

    Setup (3 minutes):
      1. Sign up at sendgrid.com (free)
      2. Settings → API Keys → Create API Key → Full Access → copy key
      3. Add to GitHub Secrets:
           SENDGRID_API_KEY  = SG.xxxxx...
           NOTIFY_EMAIL      = your@email.com
           SENDER_EMAIL      = your@email.com  (must be verified in SendGrid)
    """
    import urllib.request
    import urllib.error

    api_key     = os.environ["SENDGRID_API_KEY"]
    sender      = os.environ.get("SENDER_EMAIL", notify_email)

    payload = json.dumps({
        "personalizations": [{"to": [{"email": notify_email}]}],
        "from":    {"email": sender, "name": "Multibagger Engine"},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }).encode()

    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print(f"✅ SendGrid sent to {notify_email}  (status {resp.status})")


def send_email(html: str, subject: str):
    """
    Auto-selects the sending method based on which secrets are present.
    Priority: SendGrid (simpler) → Gmail SMTP (if App Password available)
    """
    notify_email = os.environ.get("NOTIFY_EMAIL", "")
    if not notify_email:
        print("⚠  NOTIFY_EMAIL not set — skipping email")
        return

    if os.environ.get("SENDGRID_API_KEY"):
        send_via_sendgrid(html, subject, notify_email)
    elif os.environ.get("GMAIL_USER") and os.environ.get("GMAIL_PASS"):
        send_via_gmail(html, subject, notify_email)
    else:
        print("⚠  No email credentials found.")
        print("   Set either SENDGRID_API_KEY  (recommended)")
        print("   or GMAIL_USER + GMAIL_PASS   (requires Gmail App Password)")


if __name__ == "__main__":
    regime     = load("results/regime.json",     {"regime": "UNKNOWN"})
    conviction = load("results/conviction.json", [])
    layer1     = load("results/latest.json",     [])
    layer2     = load("results/watchlist.json",  [])

    r         = regime.get("regime", "UNKNOWN")
    date_str  = datetime.utcnow().strftime("%d %b %Y")
    n_conv    = len(conviction)
    n_l1      = len(layer1)

    emoji     = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(r, "⚪")
    subject   = (
        f"{emoji} NSE Scan {date_str} — "
        f"{'★ ' + str(n_conv) + ' Conviction' if n_conv else 'No conviction'} | "
        f"{n_l1} Breakouts | Regime: {r}"
    )

    html = build_html(regime, conviction, layer1, layer2)
    send_email(html, subject)
