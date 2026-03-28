"""
screener_fetcher.py — Screener.in Data Enrichment

Fetches accurate Indian fundamental data directly from Screener.in
to replace unreliable yfinance data for the final Layer 2 candidates.

FIX v2 (2026-03):
- Fuzzy ratio name matching — Screener returns "Return on Capital Employed"
  not "ROCE"; exact string match was silently returning None for every stock.
  All ratio lookups now use fuzzy contains-based matching.
- Added fallback table key formats (Screener occasionally restructures JSON).
- Added debug logging showing exactly which ratio names were found so future
  breakages are instantly visible in the GitHub Actions log.
- Promoter holding extraction made more robust (multiple schedule name patterns).
"""

import logging
import time
import re
from typing import Optional

import requests

log = logging.getLogger(__name__)

SCREENER_BASE = "https://www.screener.in"
SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.screener.in/",
}

SLEEP_BETWEEN_REQUESTS = 1.5  # seconds — be polite to Screener.in


# ──────────────────────────────────────────────────────────────
# SYMBOL NORMALISATION
# ──────────────────────────────────────────────────────────────

def nse_to_screener_symbol(symbol: str) -> str:
    """Strip .NS / .BO suffix for Screener.in lookups."""
    return symbol.replace(".NS", "").replace(".BO", "").strip().upper()


# ──────────────────────────────────────────────────────────────
# FUZZY RATIO LOOKUP — THE CORE FIX
# Screener returns ratio names like "Return on Capital Employed",
# "Return on Equity", "Stock P/E", which don't exact-match "ROCE" etc.
# ──────────────────────────────────────────────────────────────

# Maps our internal name → list of substrings to match (case-insensitive)
# First match wins.
RATIO_ALIASES = {
    "market_cap":    ["market cap"],
    "current_price": ["current price"],
    "pe_ratio":      ["stock p/e", "p/e ratio", "price to earnings", "pe ratio"],
    "book_value":    ["book value"],
    "dividend_yield":["dividend yield"],
    "roce":          ["return on capital employed", "roce"],
    "roe":           ["return on equity", "roe"],
    "face_value":    ["face value"],
    "high_52w":      ["high / low", "52 week high", "52w high"],
    "debt_equity":   ["debt / equity", "debt/equity", "d/e ratio"],
    "sales_growth":  ["sales growth", "revenue growth"],
}


def _fuzzy_find_ratio(ratios_list: list, internal_key: str) -> Optional[str]:
    """
    Search ratios list for a ratio matching any alias for internal_key.
    Returns the raw value string, or None.
    """
    aliases = RATIO_ALIASES.get(internal_key, [internal_key])
    for entry in ratios_list:
        name = str(entry.get("name", "")).strip().lower()
        for alias in aliases:
            if alias.lower() in name:
                return entry.get("value")
    return None


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        cleaned = str(val).replace(",", "").replace("%", "").strip()
        if cleaned in ("", "-", "N/A", "NA", "—"):
            return None
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────
# SCREENER.IN JSON FETCHER
# ──────────────────────────────────────────────────────────────

def fetch_screener_data(symbol: str) -> Optional[dict]:
    """
    Fetch company data from Screener.in JSON endpoint.
    Returns a parsed dict with clean financials, or None on failure.
    """
    sc_symbol = nse_to_screener_symbol(symbol)
    url = f"{SCREENER_BASE}/api/company/{sc_symbol}/?format=json"

    try:
        r = requests.get(url, headers=SCREENER_HEADERS, timeout=15)
        if r.status_code == 404:
            url_cons = f"{SCREENER_BASE}/api/company/{sc_symbol}/consolidated/?format=json"
            r = requests.get(url_cons, headers=SCREENER_HEADERS, timeout=15)
        if r.status_code != 200:
            log.debug(f"Screener {sc_symbol}: HTTP {r.status_code}")
            return None

        data = r.json()

        # Debug: log all ratio names so we can catch future breakages
        ratio_names = [e.get("name") for e in data.get("ratios", [])]
        log.debug(f"Screener {sc_symbol}: ratio keys found = {ratio_names}")

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
    Uses fuzzy ratio matching — robust to Screener renaming fields.
    """
    ratios = data.get("ratios", [])

    # ── Key ratios via fuzzy lookup ──
    market_cap_cr  = _safe_float(_fuzzy_find_ratio(ratios, "market_cap"))
    current_price  = _safe_float(_fuzzy_find_ratio(ratios, "current_price"))
    pe_ratio       = _safe_float(_fuzzy_find_ratio(ratios, "pe_ratio"))
    book_value     = _safe_float(_fuzzy_find_ratio(ratios, "book_value"))
    dividend_yield = _safe_float(_fuzzy_find_ratio(ratios, "dividend_yield"))
    roce           = _safe_float(_fuzzy_find_ratio(ratios, "roce"))
    roe            = _safe_float(_fuzzy_find_ratio(ratios, "roe"))
    face_value     = _safe_float(_fuzzy_find_ratio(ratios, "face_value"))

    # 52W high (sometimes "High / Low" = "XXX / YYY")
    high_low_raw = _fuzzy_find_ratio(ratios, "high_52w")
    high_52w = None
    if high_low_raw:
        parts = str(high_low_raw).split("/")
        high_52w = _safe_float(parts[0].strip()) if parts else None

    log.debug(
        f"Screener {symbol}: ROCE={roce} ROE={roe} PE={pe_ratio} "
        f"MCap={market_cap_cr} BookVal={book_value}"
    )

    # ── Promoter holding — try multiple schedule name patterns ──
    promoter_pct = None
    for schedule in data.get("schedules", []):
        sched_name = str(schedule.get("name", "")).lower()
        if any(kw in sched_name for kw in ["promoter", "shareholding", "insider"]):
            rows = schedule.get("rows", [])
            for row in rows:
                cells = row.get("cells", [])
                if cells:
                    val = _safe_float(cells[0])
                    if val is not None and 0 < val <= 100:
                        promoter_pct = val
                        break
            if promoter_pct is not None:
                break

    # ── Quarterly financials ──
    quarterly_sales   = _extract_table(data, "quarterly", ["sales", "revenue", "net sales"])
    quarterly_profit  = _extract_table(data, "quarterly", ["net profit", "profit after tax", "pat"])
    annual_sales      = _extract_table(data, "annual",    ["sales", "revenue", "net sales"])
    annual_profit     = _extract_table(data, "annual",    ["net profit", "profit after tax", "pat"])

    # Revenue growth — latest quarter vs same quarter last year (YoY)
    rev_growth_yoy = None
    if quarterly_sales and len(quarterly_sales) >= 5:
        latest   = quarterly_sales[0]
        year_ago = quarterly_sales[4]
        if latest is not None and year_ago and year_ago != 0:
            rev_growth_yoy = round((latest - year_ago) / abs(year_ago) * 100, 1)

    # 3-year revenue CAGR
    rev_cagr_3yr = None
    if annual_sales and len(annual_sales) >= 4:
        latest_yr = annual_sales[0]
        three_yr  = annual_sales[3]
        if latest_yr and three_yr and three_yr > 0:
            rev_cagr_3yr = round(((latest_yr / three_yr) ** (1/3) - 1) * 100, 1)

    # PAT growth YoY
    pat_growth_yoy = None
    if quarterly_profit and len(quarterly_profit) >= 5:
        latest   = quarterly_profit[0]
        year_ago = quarterly_profit[4]
        if latest is not None and year_ago and year_ago != 0:
            pat_growth_yoy = round((latest - year_ago) / abs(year_ago) * 100, 1)

    # D/E ratio
    de_ratio = _get_de_ratio(data, ratios)

    # Operating profit margin
    opm = None
    opm_data = _extract_table(data, "quarterly", ["opm", "operating profit margin", "ebitda margin"])
    if opm_data:
        opm = _safe_float(opm_data[0])

    return {
        "source":              "screener.in",
        "symbol":              symbol,
        "market_cap_cr":       market_cap_cr,
        "current_price":       current_price,
        "pe_ratio":            pe_ratio,
        "book_value":          book_value,
        "roce_pct":            roce,
        "roe_pct":             roe,
        "promoter_pct":        promoter_pct,
        "de_ratio":            de_ratio,
        "rev_growth_yoy_pct":  rev_growth_yoy,
        "rev_cagr_3yr_pct":    rev_cagr_3yr,
        "pat_growth_yoy_pct":  pat_growth_yoy,
        "opm_pct":             opm,
        "dividend_yield_pct":  dividend_yield,
        "_quarterly_sales":    quarterly_sales[:8] if quarterly_sales else [],
        "_quarterly_profit":   quarterly_profit[:8] if quarterly_profit else [],
    }


def _extract_table(data: dict, table_type: str, row_name_aliases: list) -> Optional[list]:
    """
    Extract a specific row from Screener's financial tables using fuzzy matching.

    table_type: "quarterly" or "annual"
    row_name_aliases: list of lowercase substrings to match (first match wins)

    Screener returns tables as lists of dicts with a "rows" key.
    The first cell in each row is the label; the rest are values newest-first.
    """
    aliases = [a.lower() for a in row_name_aliases]
    tables = data.get(table_type, []) or []

    for table in tables:
        rows = table.get("rows", []) if isinstance(table, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells", [])
            label = str(cells[0]).strip().lower() if cells else ""
            if any(alias in label for alias in aliases):
                values = []
                for cell in cells[1:]:
                    try:
                        cleaned = str(cell).replace(",", "").replace("%", "").strip()
                        values.append(float(cleaned) if cleaned not in ("", "-", "N/A", "—") else None)
                    except (ValueError, TypeError):
                        values.append(None)
                return values if values else None

    # Fallback: top-level keys that Screener sometimes uses
    for key in [table_type + "_results", table_type]:
        block = data.get(key)
        if isinstance(block, dict):
            for row_key, row_vals in block.items():
                if any(alias in row_key.lower() for alias in aliases) and isinstance(row_vals, list):
                    return [float(v) if v is not None else None for v in row_vals]

    return None


def _get_de_ratio(data: dict, ratios: list) -> Optional[float]:
    """Extract Debt/Equity from Screener — ratios first, then balance sheet."""
    # Try fuzzy ratio lookup
    raw = _fuzzy_find_ratio(ratios, "debt_equity")
    val = _safe_float(raw)
    if val is not None:
        return val

    # Try balance sheet tables
    debt   = _extract_table(data, "annual", ["borrowings", "total debt", "long term debt"])
    equity = _extract_table(data, "annual", ["equity capital", "share capital"])
    reserves = _extract_table(data, "annual", ["reserves", "reserves and surplus"])

    if debt and equity and reserves:
        d   = debt[0]
        e   = equity[0]
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
    Called AFTER the main scan on the final shortlist only.
    """
    enriched = candidates[:max_candidates]
    total = len(enriched)
    log.info(f"Enriching {total} candidates with Screener.in data...")

    for i, cand in enumerate(enriched):
        symbol = cand["symbol"]
        sym_clean = nse_to_screener_symbol(symbol)
        log.info(f"  [{i+1}/{total}] Screener fetch: {sym_clean}")

        sc = fetch_screener_data(symbol)

        if sc is None:
            log.debug(f"  {sym_clean}: Screener fetch failed — keeping yfinance data")
            cand["screener"] = None
            cand["screener_flags"] = []
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        # ── Merge: override yfinance with Screener where available ──
        if sc["promoter_pct"] is not None:
            cand["shareholding"]["insider_pct"] = sc["promoter_pct"]
            cand["shareholding"]["promoter_score"] = round(min(sc["promoter_pct"] / 75, 1.0), 3)

        if sc["rev_growth_yoy_pct"] is not None:
            cand["growth"]["revenue_growth_pct"] = sc["rev_growth_yoy_pct"]

        if sc["pat_growth_yoy_pct"] is not None:
            cand["growth"]["earnings_growth_pct"] = sc["pat_growth_yoy_pct"]

        if sc["opm_pct"] is not None:
            cand["growth"]["profit_margin_pct"] = sc["opm_pct"]

        if sc["de_ratio"] is not None:
            cand["debt"]["current_de"] = sc["de_ratio"]
            if sc["de_ratio"] <= 0.15:
                cand["debt"]["debt_score"] = 0.8
                cand["debt"]["is_near_zero_debt"] = True
            else:
                cand["debt"]["debt_score"] = round(max(1.0 - sc["de_ratio"] / 2.0, 0.0), 3)

        if sc["roce_pct"] is not None:
            cand["screener_roce"] = sc["roce_pct"]

        # ── Generate Screener-specific flags ──
        flags = []
        if sc["roce_pct"] and sc["roce_pct"] > 20:
            flags.append(f"⭐ ROCE {sc['roce_pct']:.1f}% (Screener)")
        elif sc["roce_pct"] and sc["roce_pct"] > 15:
            flags.append(f"✅ ROCE {sc['roce_pct']:.1f}% (Screener)")
        elif sc["roce_pct"] is not None and sc["roce_pct"] < 10:
            flags.append(f"⚠️ Low ROCE {sc['roce_pct']:.1f}% — possible value trap")

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

        # Revenue declining 3 consecutive quarters (risk flag)
        qs = sc.get("_quarterly_sales", [])
        if len(qs) >= 4 and all(v is not None for v in qs[:3]):
            if qs[0] < qs[1] < qs[2]:
                flags.append("⚠️ Revenue declining 3 consecutive quarters (Screener)")

        cand["screener"] = sc
        cand["screener_flags"] = flags
        cand["all_flags"] = flags + [
            f for f in cand.get("all_flags", [])
            if "(Screener)" not in f
        ]

        log.info(
            f"  ✓ {sym_clean}: ROCE={sc['roce_pct']}% "
            f"Promoter={sc['promoter_pct']}% "
            f"D/E={sc['de_ratio']} "
            f"RevGrowth={sc['rev_growth_yoy_pct']}%"
        )

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(
        f"Screener enrichment complete. "
        f"{sum(1 for c in enriched if c.get('screener'))} / {total} fetched successfully."
    )
    return enriched


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# Run: python screener_fetcher.py INFY
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.DEBUG)
    sym = sys.argv[1] if len(sys.argv) > 1 else "INFY"
    print(f"Fetching Screener.in data for {sym}...")
    result = fetch_screener_data(sym + ".NS")
    if result:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Failed to fetch data for {sym}")
