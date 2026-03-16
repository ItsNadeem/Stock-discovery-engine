"""
screener_fetcher.py — Screener.in Data Enrichment

Fetches accurate Indian fundamental data directly from Screener.in
to replace unreliable yfinance data for the final Layer 2 candidates.

Why this matters:
  yfinance gets NSE data from a third-party Yahoo feed. For small/microcaps
  it frequently returns stale, missing, or wrong fundamentals (e.g. INDSWFTLAB
  showing 1641% revenue growth in our first real run — a pure data artefact).
  Screener.in pulls directly from BSE/NSE regulatory filings (XBRL), making
  it the most accurate free source for Indian quarterly financials.

What Screener.in provides that yfinance often gets wrong:
  - Accurate quarterly revenue / PAT (from actual BSE filings)
  - Real promoter holding % (from NSE shareholding pattern)
  - ROCE (not available in yfinance at all)
  - Correct debt figures (yfinance often misclassifies)
  - Sales growth over 3/5 years (not just TTM)
  - Market cap updated to current price

Usage:
  from screener_fetcher import enrich_with_screener
  candidates = enrich_with_screener(candidates)  # adds 'screener' key to each

Rate limiting:
  Screener.in is a free service — be polite. We add a 1.5s sleep between
  requests, and only call it for the final top-N candidates (15–20 stocks),
  never for the full 1,800-symbol universe scan.

Note on the endpoint:
  https://www.screener.in/api/company/SYMBOL/?format=json
  This is the same endpoint their website uses internally. It is not
  an officially documented public API — use respectfully, don't abuse it.
  If it stops working, fall back gracefully (the code handles this).
"""

import logging
import time
import re
from typing import Optional

import requests

log = logging.getLogger(__name__)

SCREENER_BASE    = "https://www.screener.in"
SCREENER_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept":          "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.screener.in/",
}
SLEEP_BETWEEN_REQUESTS = 1.5   # seconds — be polite to Screener.in


# ──────────────────────────────────────────────────────────────
# SYMBOL NORMALISATION
# yfinance symbols are like "INFY.NS", Screener uses "INFY"
# Some BSE-listed companies need special handling
# ──────────────────────────────────────────────────────────────

def nse_to_screener_symbol(symbol: str) -> str:
    """Strip .NS / .BO suffix for Screener.in lookups."""
    return symbol.replace(".NS", "").replace(".BO", "").strip().upper()


# ──────────────────────────────────────────────────────────────
# SCREENER.IN JSON FETCHER
# ──────────────────────────────────────────────────────────────

def fetch_screener_data(symbol: str) -> Optional[dict]:
    """
    Fetch company data from Screener.in JSON endpoint.

    Returns a parsed dict with clean financials, or None on failure.

    The JSON response structure has these top-level keys:
      id, name, bse_code, nse_id, isin,
      peers, schedules, concalls,
      ratios  → list of {name, value} dicts (key ratios)
      [financial tables as arrays]
    """
    sc_symbol = nse_to_screener_symbol(symbol)
    url = f"{SCREENER_BASE}/api/company/{sc_symbol}/?format=json"

    try:
        r = requests.get(url, headers=SCREENER_HEADERS, timeout=15)

        if r.status_code == 404:
            # Try consolidated view (some companies only have consolidated)
            url_cons = f"{SCREENER_BASE}/api/company/{sc_symbol}/consolidated/?format=json"
            r = requests.get(url_cons, headers=SCREENER_HEADERS, timeout=15)

        if r.status_code != 200:
            log.debug(f"Screener {sc_symbol}: HTTP {r.status_code}")
            return None

        data = r.json()
        return parse_screener_response(data, sc_symbol)

    except requests.exceptions.Timeout:
        log.debug(f"Screener {sc_symbol}: timeout")
        return None
    except Exception as e:
        log.debug(f"Screener {sc_symbol}: {e}")
        return None


def parse_screener_response(data: dict, symbol: str) -> dict:
    """
    Parse the Screener.in JSON response into a clean, flat dict.
    Handles missing fields gracefully — returns None for unavailable data.
    """

    def safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            # Screener values come as strings like "1,234.56" or "12.5%"
            cleaned = str(val).replace(",", "").replace("%", "").strip()
            if cleaned in ("", "-", "N/A", "NA"):
                return None
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    # ── Key ratios (flat list of {name, value}) ──
    ratios = {}
    for r in data.get("ratios", []):
        name  = str(r.get("name", "")).strip()
        value = r.get("value")
        if name:
            ratios[name] = value

    # Standard Screener ratio names
    market_cap_cr      = safe_float(ratios.get("Market Cap"))
    current_price      = safe_float(ratios.get("Current Price"))
    high_52w           = safe_float(ratios.get("High / Low", "").split("/")[0] if "/" in str(ratios.get("High / Low", "")) else ratios.get("High / Low"))
    pe_ratio           = safe_float(ratios.get("Stock P/E"))
    book_value         = safe_float(ratios.get("Book Value"))
    dividend_yield     = safe_float(ratios.get("Dividend Yield"))
    roce               = safe_float(ratios.get("ROCE"))
    roe                = safe_float(ratios.get("ROE"))
    face_value         = safe_float(ratios.get("Face Value"))

    # ── Promoter holding from shareholding schedule ──
    promoter_pct = None
    for schedule in data.get("schedules", []):
        if "promoter" in str(schedule.get("name", "")).lower():
            rows = schedule.get("rows", [])
            for row in rows:
                cells = row.get("cells", [])
                if cells:
                    # Most recent quarter is first column
                    val = safe_float(cells[0]) if cells else None
                    if val is not None:
                        promoter_pct = val
                        break
            break

    # ── Quarterly financials ──
    # Screener returns P&L, Balance Sheet, Cash Flow as separate tables
    quarterly_sales    = _extract_table(data, "quarterly", "Sales")
    quarterly_profit   = _extract_table(data, "quarterly", "Net Profit")
    annual_sales       = _extract_table(data, "annual",    "Sales")
    annual_profit      = _extract_table(data, "annual",    "Net Profit")

    # Revenue growth (last quarter YoY)
    rev_growth_yoy = None
    if quarterly_sales and len(quarterly_sales) >= 5:
        latest = quarterly_sales[0]
        year_ago = quarterly_sales[4]   # 4 quarters back = same quarter last year
        if latest and year_ago and year_ago != 0:
            rev_growth_yoy = round((latest - year_ago) / abs(year_ago) * 100, 1)

    # 3-year revenue CAGR
    rev_cagr_3yr = None
    if annual_sales and len(annual_sales) >= 4:
        latest_yr = annual_sales[0]
        three_yr  = annual_sales[3]
        if latest_yr and three_yr and three_yr > 0:
            rev_cagr_3yr = round(((latest_yr / three_yr) ** (1/3) - 1) * 100, 1)

    # PAT growth (last quarter YoY)
    pat_growth_yoy = None
    if quarterly_profit and len(quarterly_profit) >= 5:
        latest = quarterly_profit[0]
        year_ago = quarterly_profit[4]
        if latest is not None and year_ago and year_ago != 0:
            pat_growth_yoy = round((latest - year_ago) / abs(year_ago) * 100, 1)

    # Debt / Equity from balance sheet
    de_ratio = _get_de_ratio(data)

    # ── Operating profit margin (OPM) from quarterly ──
    opm = None
    opm_data = _extract_table(data, "quarterly", "OPM")
    if not opm_data:
        opm_data = _extract_table(data, "quarterly", "Operating Profit Margin")
    if opm_data:
        opm = safe_float(opm_data[0])

    return {
        "source":            "screener.in",
        "symbol":            symbol,
        "market_cap_cr":     market_cap_cr,
        "current_price":     current_price,
        "pe_ratio":          pe_ratio,
        "book_value":        book_value,
        "roce_pct":          roce,
        "roe_pct":           roe,
        "promoter_pct":      promoter_pct,
        "de_ratio":          de_ratio,
        "rev_growth_yoy_pct": rev_growth_yoy,
        "rev_cagr_3yr_pct":  rev_cagr_3yr,
        "pat_growth_yoy_pct": pat_growth_yoy,
        "opm_pct":           opm,
        "dividend_yield_pct": dividend_yield,
        # Raw quarterly arrays for further analysis
        "_quarterly_sales":   quarterly_sales[:8]  if quarterly_sales  else [],
        "_quarterly_profit":  quarterly_profit[:8] if quarterly_profit else [],
    }


def _extract_table(data: dict, table_type: str, row_name: str) -> Optional[list]:
    """
    Extract a specific row from Screener's financial tables.
    table_type: "quarterly" or "annual"
    row_name:   "Sales", "Net Profit", "OPM", etc.

    Screener returns tables as:
      {"name": "Profit & Loss", "rows": [{"cells": [...values...]}, ...]}
    The first cell in each row is the label, rest are values newest-first.
    """
    tables = data.get(table_type, []) or []

    for table in tables:
        rows = table.get("rows", []) if isinstance(table, dict) else []
        for row in rows:
            if isinstance(row, dict):
                cells = row.get("cells", [])
                label = str(cells[0]).strip() if cells else ""
                if row_name.lower() in label.lower():
                    values = []
                    for cell in cells[1:]:
                        try:
                            cleaned = str(cell).replace(",", "").replace("%", "").strip()
                            values.append(float(cleaned) if cleaned not in ("", "-", "N/A") else None)
                        except (ValueError, TypeError):
                            values.append(None)
                    return values if values else None

    # Also check top-level keys that Screener sometimes uses
    for key in [table_type + "_results", table_type]:
        block = data.get(key)
        if isinstance(block, dict):
            for row_key, row_vals in block.items():
                if row_name.lower() in row_key.lower() and isinstance(row_vals, list):
                    return [float(v) if v is not None else None for v in row_vals]

    return None


def _get_de_ratio(data: dict) -> Optional[float]:
    """Extract Debt/Equity from Screener balance sheet data."""
    # Try ratios first
    for r in data.get("ratios", []):
        name = str(r.get("name", "")).lower()
        if "debt" in name and "equity" in name:
            val = str(r.get("value", "")).replace(",", "").strip()
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Try balance sheet tables
    debt   = _extract_table(data, "annual", "Borrowings")
    equity = _extract_table(data, "annual", "Equity Capital")
    reserves = _extract_table(data, "annual", "Reserves")

    if debt and equity and reserves:
        d = debt[0]
        e = equity[0]
        res = reserves[0]
        if d is not None and e is not None and res is not None:
            total_equity = (e or 0) + (res or 0)
            if total_equity > 0:
                return round(d / total_equity, 2)

    return None


# ──────────────────────────────────────────────────────────────
# ENRICHMENT FUNCTION — called after main scan on top candidates
# ──────────────────────────────────────────────────────────────

def enrich_with_screener(candidates: list[dict], max_candidates: int = 20) -> list[dict]:
    """
    Fetch Screener.in data for the top N candidates and merge it in.

    Called AFTER the main scan on the final shortlist — never on the
    full 1,800-symbol universe (too slow, too many requests).

    For each candidate:
      - Fetches Screener.in JSON
      - Merges accurate fundamentals into candidate dict under 'screener' key
      - Overrides growth/debt/valuation fields if Screener data is available
        and more complete than yfinance
      - Adds 'screener_flags' list for notable findings
    """
    enriched = candidates[:max_candidates]
    total    = len(enriched)

    log.info(f"Enriching {total} candidates with Screener.in data...")

    for i, cand in enumerate(enriched):
        symbol    = cand["symbol"]
        sym_clean = nse_to_screener_symbol(symbol)

        log.info(f"  [{i+1}/{total}] Screener fetch: {sym_clean}")
        sc = fetch_screener_data(symbol)

        if sc is None:
            log.debug(f"  {sym_clean}: Screener fetch failed — keeping yfinance data")
            cand["screener"] = None
            cand["screener_flags"] = []
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        # ── Merge: override yfinance with Screener where Screener has data ──

        # Promoter holding — Screener is more accurate (from NSE filings)
        if sc["promoter_pct"] is not None:
            cand["shareholding"]["insider_pct"] = sc["promoter_pct"]
            # Recalculate promoter score
            cand["shareholding"]["promoter_score"] = round(
                min(sc["promoter_pct"] / 75, 1.0), 3
            )

        # Revenue growth — use Screener's YoY quarterly (more reliable)
        if sc["rev_growth_yoy_pct"] is not None:
            cand["growth"]["revenue_growth_pct"] = sc["rev_growth_yoy_pct"]

        # PAT growth
        if sc["pat_growth_yoy_pct"] is not None:
            cand["growth"]["earnings_growth_pct"] = sc["pat_growth_yoy_pct"]

        # Operating margin
        if sc["opm_pct"] is not None:
            cand["growth"]["profit_margin_pct"] = sc["opm_pct"]

        # D/E ratio — override yfinance (frequently wrong for Indian cos)
        if sc["de_ratio"] is not None:
            cand["debt"]["current_de"] = sc["de_ratio"]
            # Recalculate debt score
            if sc["de_ratio"] <= 0.15:
                cand["debt"]["debt_score"]    = 0.8
                cand["debt"]["is_near_zero_debt"] = True
            else:
                cand["debt"]["debt_score"] = round(
                    max(1.0 - sc["de_ratio"] / 2.0, 0.0), 3
                )

        # ROCE — not in yfinance at all
        if sc["roce_pct"] is not None:
            cand["screener_roce"] = sc["roce_pct"]

        # ── Generate Screener-specific flags ──
        flags = []

        if sc["roce_pct"] and sc["roce_pct"] > 20:
            flags.append(f"⭐ ROCE {sc['roce_pct']:.1f}% (Screener)")
        elif sc["roce_pct"] and sc["roce_pct"] > 15:
            flags.append(f"✅ ROCE {sc['roce_pct']:.1f}% (Screener)")

        if sc["rev_cagr_3yr_pct"] and sc["rev_cagr_3yr_pct"] > 20:
            flags.append(f"📈 3Yr Rev CAGR {sc['rev_cagr_3yr_pct']:.1f}% (Screener)")

        if sc["promoter_pct"] and sc["promoter_pct"] > 70:
            flags.append(f"👥 Promoter {sc['promoter_pct']:.1f}% (Screener)")

        if sc["de_ratio"] is not None and sc["de_ratio"] < 0.1:
            flags.append(f"🛡️ Debt-Free D/E {sc['de_ratio']:.2f} (Screener)")

        if (sc["rev_growth_yoy_pct"] or 0) > 25:
            flags.append(f"💰 QoQ Rev Growth {sc['rev_growth_yoy_pct']:.1f}% (Screener)")

        if (sc["pat_growth_yoy_pct"] or 0) > 30:
            flags.append(f"🏆 QoQ PAT Growth {sc['pat_growth_yoy_pct']:.1f}% (Screener)")

        # Check for consecutive revenue decline (risk flag)
        qs = sc.get("_quarterly_sales", [])
        if len(qs) >= 4 and all(v is not None for v in qs[:3]):
            if qs[0] < qs[1] < qs[2]:
                flags.append("⚠️ Revenue declining 3 consecutive quarters (Screener)")

        cand["screener"]       = sc
        cand["screener_flags"] = flags

        # Prepend screener flags to all_flags (they're more reliable)
        cand["all_flags"] = flags + [
            f for f in cand.get("all_flags", [])
            if "(Screener)" not in f
        ]

        log.info(
            f"    ✓ {sym_clean}: ROCE={sc['roce_pct']}%  "
            f"Promoter={sc['promoter_pct']}%  "
            f"D/E={sc['de_ratio']}  "
            f"RevGrowth={sc['rev_growth_yoy_pct']}%"
        )
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(f"Screener enrichment complete. "
             f"{sum(1 for c in enriched if c.get('screener'))} / {total} fetched successfully.")
    return enriched


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# Run: python screener_fetcher.py INFY
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    sym = sys.argv[1] if len(sys.argv) > 1 else "INFY"
    print(f"Fetching Screener.in data for {sym}...")
    result = fetch_screener_data(sym + ".NS")
    if result:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Failed to fetch data for {sym}")
