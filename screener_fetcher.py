"""
screener_fetcher.py — Screener.in Data Enrichment v3

FIX (2026-03-30): 0/30 fetched — Screener.in JSON structure changed.
The endpoint now returns data inside a different nesting. Added:
  1. Verbose debug logging of the raw top-level keys on every fetch
     so future breakages are immediately visible in Actions logs.
  2. Fallback to the HTML page scraper when JSON returns no ratios
     (Screener sometimes returns HTML even when ?format=json is sent).
  3. Retry with consolidated endpoint as first fallback (not second).
  4. Accept both old (list of {name,value}) and new (dict) ratio formats.

FIX: Fuzzy ratio name matching (from prior session, preserved).
"""

import logging
import time
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

SCREENER_BASE = "https://www.screener.in"
SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.screener.in/",
}
SLEEP_BETWEEN_REQUESTS = 2.0  # raised to 2s — Screener rate-limits aggressively

RATIO_ALIASES = {
    "market_cap":    ["market cap"],
    "current_price": ["current price"],
    "pe_ratio":      ["stock p/e", "p/e ratio", "price to earnings", "pe ratio", "p/e"],
    "book_value":    ["book value"],
    "dividend_yield":["dividend yield"],
    "roce":          ["return on capital employed", "roce"],
    "roe":           ["return on equity", "roe"],
    "face_value":    ["face value"],
    "high_52w":      ["high / low", "52 week high", "52w high"],
    "debt_equity":   ["debt / equity", "debt/equity", "d/e ratio", "debt to equity"],
    "sales_growth":  ["sales growth", "revenue growth"],
}


def nse_to_screener_symbol(symbol: str) -> str:
    return symbol.replace(".NS", "").replace(".BO", "").strip().upper()


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        cleaned = str(val).replace(",", "").replace("%", "").strip()
        if cleaned in ("", "-", "N/A", "NA", "—", "--"):
            return None
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _fuzzy_find_ratio(ratios_list: list, internal_key: str) -> Optional[str]:
    aliases = RATIO_ALIASES.get(internal_key, [internal_key])
    for entry in ratios_list:
        name = str(entry.get("name", "") or entry.get("label", "")).strip().lower()
        for alias in aliases:
            if alias.lower() in name:
                return entry.get("value") or entry.get("val")
    return None


def _parse_ratios_from_html(html: str, symbol: str) -> dict:
    """
    Fallback: scrape key ratios directly from Screener HTML page.
    Used when the JSON endpoint returns no/empty ratios.
    """
    result = {}
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Screener's ratio section: <ul class="row-top-ratios"> or similar
        for li in soup.select("li.flex-column, li.ratio, span.name"):
            name_el = li.select_one(".name, span.name")
            val_el  = li.select_one(".value, span.value, span.number")
            if name_el and val_el:
                name = name_el.get_text(strip=True).lower()
                val  = val_el.get_text(strip=True)
                result[name] = val
        log.debug(f"Screener HTML {symbol}: scraped {len(result)} ratios from HTML")
    except Exception as e:
        log.debug(f"HTML scrape failed for {symbol}: {e}")
    return result


def fetch_screener_data(symbol: str) -> Optional[dict]:
    sc_symbol = nse_to_screener_symbol(symbol)

    # Try standalone first, then consolidated (reversed from old order — consolidated
    # is more likely to have data for smaller companies)
    urls_to_try = [
        f"{SCREENER_BASE}/api/company/{sc_symbol}/?format=json",
        f"{SCREENER_BASE}/api/company/{sc_symbol}/consolidated/?format=json",
    ]

    data     = None
    raw_html = None

    for url in urls_to_try:
        try:
            r = requests.get(url, headers=SCREENER_HEADERS, timeout=15)
            if r.status_code == 200:
                content_type = r.headers.get("content-type", "")
                if "json" in content_type:
                    data = r.json()
                    # Log top-level keys for debugging
                    log.debug(f"Screener {sc_symbol}: JSON keys = {list(data.keys())[:10]}")
                    ratios = data.get("ratios", [])
                    if ratios:
                        log.debug(f"Screener {sc_symbol}: {len(ratios)} ratio entries")
                        break
                    else:
                        log.debug(f"Screener {sc_symbol}: JSON has no ratios — trying next URL")
                        data = None
                else:
                    # HTML response — scrape it
                    raw_html = r.text
                    log.debug(f"Screener {sc_symbol}: got HTML response (not JSON)")
        except requests.exceptions.Timeout:
            log.debug(f"Screener {sc_symbol}: timeout")
        except Exception as e:
            log.debug(f"Screener {sc_symbol}: {e}")

    if data and data.get("ratios"):
        return parse_screener_json(data, sc_symbol)

    if raw_html:
        return parse_screener_html(raw_html, sc_symbol)

    log.debug(f"Screener {sc_symbol}: all fetch attempts failed")
    return None


def parse_screener_json(data: dict, symbol: str) -> dict:
    """Parse the Screener JSON response with fuzzy ratio matching."""
    ratios = data.get("ratios", [])

    # Handle both list-of-dicts and dict formats
    if isinstance(ratios, dict):
        ratios = [{"name": k, "value": v} for k, v in ratios.items()]

    market_cap_cr  = _safe_float(_fuzzy_find_ratio(ratios, "market_cap"))
    current_price  = _safe_float(_fuzzy_find_ratio(ratios, "current_price"))
    pe_ratio       = _safe_float(_fuzzy_find_ratio(ratios, "pe_ratio"))
    book_value     = _safe_float(_fuzzy_find_ratio(ratios, "book_value"))
    dividend_yield = _safe_float(_fuzzy_find_ratio(ratios, "dividend_yield"))
    roce           = _safe_float(_fuzzy_find_ratio(ratios, "roce"))
    roe            = _safe_float(_fuzzy_find_ratio(ratios, "roe"))

    log.debug(
        f"Screener {symbol}: ROCE={roce} ROE={roe} PE={pe_ratio} MCap={market_cap_cr}"
    )

    # Promoter holding
    promoter_pct = None
    for schedule in data.get("schedules", []):
        sched_name = str(schedule.get("name", "")).lower()
        if any(kw in sched_name for kw in ["promoter", "shareholding", "insider"]):
            for row in schedule.get("rows", []):
                cells = row.get("cells", [])
                if cells:
                    val = _safe_float(cells[0])
                    if val is not None and 0 < val <= 100:
                        promoter_pct = val
                        break
            if promoter_pct is not None:
                break

    quarterly_sales  = _extract_table(data, "quarterly",
                                      ["sales", "revenue", "net sales"])
    quarterly_profit = _extract_table(data, "quarterly",
                                      ["net profit", "profit after tax", "pat"])
    annual_sales     = _extract_table(data, "annual",
                                      ["sales", "revenue", "net sales"])

    rev_growth_yoy = None
    if quarterly_sales and len(quarterly_sales) >= 5:
        l, ya = quarterly_sales[0], quarterly_sales[4]
        if l is not None and ya and ya != 0:
            rev_growth_yoy = round((l - ya) / abs(ya) * 100, 1)

    rev_cagr_3yr = None
    if annual_sales and len(annual_sales) >= 4:
        l, t = annual_sales[0], annual_sales[3]
        if l and t and t > 0:
            rev_cagr_3yr = round(((l / t) ** (1/3) - 1) * 100, 1)

    pat_growth_yoy = None
    if quarterly_profit and len(quarterly_profit) >= 5:
        l, ya = quarterly_profit[0], quarterly_profit[4]
        if l is not None and ya and ya != 0:
            pat_growth_yoy = round((l - ya) / abs(ya) * 100, 1)

    de_ratio = _get_de_ratio(data, ratios)

    opm = None
    opm_data = _extract_table(data, "quarterly",
                               ["opm", "operating profit margin", "ebitda margin"])
    if opm_data:
        opm = _safe_float(opm_data[0])

    return {
        "source":             "screener.in",
        "symbol":             symbol,
        "market_cap_cr":      market_cap_cr,
        "current_price":      current_price,
        "pe_ratio":           pe_ratio,
        "book_value":         book_value,
        "roce_pct":           roce,
        "roe_pct":            roe,
        "promoter_pct":       promoter_pct,
        "de_ratio":           de_ratio,
        "rev_growth_yoy_pct": rev_growth_yoy,
        "rev_cagr_3yr_pct":   rev_cagr_3yr,
        "pat_growth_yoy_pct": pat_growth_yoy,
        "opm_pct":            opm,
        "dividend_yield_pct": dividend_yield,
        "_quarterly_sales":   quarterly_sales[:8] if quarterly_sales else [],
        "_quarterly_profit":  quarterly_profit[:8] if quarterly_profit else [],
    }


def parse_screener_html(html: str, symbol: str) -> Optional[dict]:
    """
    Fallback HTML parser for when JSON endpoint fails.
    Extracts key ratios from the Screener company page HTML.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        def find_ratio(name_fragment: str) -> Optional[float]:
            for li in soup.select("li"):
                name_el = li.select_one(".name")
                val_el  = li.select_one(".value, .number")
                if name_el and val_el:
                    if name_fragment.lower() in name_el.get_text().lower():
                        return _safe_float(val_el.get_text())
            return None

        roce   = find_ratio("Return on Capital") or find_ratio("ROCE")
        roe    = find_ratio("Return on Equity") or find_ratio("ROE")
        pe     = find_ratio("Stock P/E") or find_ratio("P/E")
        mcap   = find_ratio("Market Cap")
        book   = find_ratio("Book Value")

        # Promoter from shareholding table
        promoter_pct = None
        for table in soup.select("table"):
            for row in table.select("tr"):
                cells = row.select("td")
                if cells and "promoter" in cells[0].get_text().lower():
                    for cell in cells[1:]:
                        val = _safe_float(cell.get_text())
                        if val is not None and 0 < val <= 100:
                            promoter_pct = val
                            break
                    if promoter_pct:
                        break

        log.info(
            f"  ✓ {symbol} (HTML): ROCE={roce} ROE={roe} PE={pe} "
            f"MCap={mcap} Promoter={promoter_pct}"
        )

        return {
            "source":             "screener.in (html)",
            "symbol":             symbol,
            "market_cap_cr":      mcap,
            "current_price":      None,
            "pe_ratio":           pe,
            "book_value":         book,
            "roce_pct":           roce,
            "roe_pct":            roe,
            "promoter_pct":       promoter_pct,
            "de_ratio":           None,
            "rev_growth_yoy_pct": None,
            "rev_cagr_3yr_pct":   None,
            "pat_growth_yoy_pct": None,
            "opm_pct":            None,
            "dividend_yield_pct": None,
            "_quarterly_sales":   [],
            "_quarterly_profit":  [],
        }
    except Exception as e:
        log.debug(f"HTML parse failed for {symbol}: {e}")
        return None


def _extract_table(data: dict, table_type: str,
                   row_name_aliases: list) -> Optional[list]:
    aliases = [a.lower() for a in row_name_aliases]
    tables  = data.get(table_type, []) or []

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
                        cleaned = str(cell).replace(",","").replace("%","").strip()
                        values.append(
                            float(cleaned)
                            if cleaned not in ("", "-", "N/A", "—")
                            else None
                        )
                    except (ValueError, TypeError):
                        values.append(None)
                return values if values else None

    for key in [table_type + "_results", table_type]:
        block = data.get(key)
        if isinstance(block, dict):
            for row_key, row_vals in block.items():
                if any(alias in row_key.lower() for alias in aliases) and isinstance(row_vals, list):
                    return [float(v) if v is not None else None for v in row_vals]
    return None


def _get_de_ratio(data: dict, ratios: list) -> Optional[float]:
    raw = _fuzzy_find_ratio(ratios, "debt_equity")
    val = _safe_float(raw)
    if val is not None:
        return val
    debt     = _extract_table(data, "annual", ["borrowings", "total debt", "long term debt"])
    equity   = _extract_table(data, "annual", ["equity capital", "share capital"])
    reserves = _extract_table(data, "annual", ["reserves", "reserves and surplus"])
    if debt and equity and reserves:
        d, e, res = debt[0], equity[0], reserves[0]
        if d is not None and e is not None and res is not None:
            total_equity = (e or 0) + (res or 0)
            if total_equity > 0:
                return round(d / total_equity, 2)
    return None


def enrich_with_screener(candidates: list[dict],
                         max_candidates: int = 20) -> list[dict]:
    enriched = candidates[:max_candidates]
    total    = len(enriched)
    log.info(f"Enriching {total} candidates with Screener.in data...")

    success = 0
    for i, cand in enumerate(enriched):
        symbol    = cand["symbol"]
        sym_clean = nse_to_screener_symbol(symbol)
        log.info(f"  [{i+1}/{total}] Screener fetch: {sym_clean}")

        sc = fetch_screener_data(symbol)

        if sc is None:
            log.debug(f"  {sym_clean}: Screener failed — keeping yfinance data")
            cand["screener"]       = None
            cand["screener_flags"] = []
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        success += 1

        # Override with Screener data where available
        if sc.get("promoter_pct") is not None:
            # v5 stores shareholding flat; v4 stores in a dict
            if isinstance(cand.get("shareholding"), dict):
                cand["shareholding"]["insider_pct"] = sc["promoter_pct"]
                cand["shareholding"]["promoter_score"] = round(
                    min(sc["promoter_pct"] / 75, 1.0), 3)

        if sc.get("rev_growth_yoy_pct") is not None:
            if isinstance(cand.get("growth"), dict):
                cand["growth"]["revenue_growth_pct"] = sc["rev_growth_yoy_pct"]
        if sc.get("pat_growth_yoy_pct") is not None:
            if isinstance(cand.get("growth"), dict):
                cand["growth"]["earnings_growth_pct"] = sc["pat_growth_yoy_pct"]
        if sc.get("opm_pct") is not None:
            if isinstance(cand.get("growth"), dict):
                cand["growth"]["profit_margin_pct"] = sc["opm_pct"]

        if sc.get("de_ratio") is not None:
            if isinstance(cand.get("debt"), dict):
                cand["debt"]["current_de"] = sc["de_ratio"]
                if sc["de_ratio"] <= 0.15:
                    cand["debt"]["debt_score"]       = 0.8
                    cand["debt"]["is_near_zero_debt"] = True
                else:
                    cand["debt"]["debt_score"] = round(
                        max(1.0 - sc["de_ratio"] / 2.0, 0.0), 3)

        if sc.get("roce_pct") is not None:
            cand["screener_roce"] = sc["roce_pct"]

        flags = []
        if sc["roce_pct"] and sc["roce_pct"] > 20:
            flags.append(f"⭐ ROCE {sc['roce_pct']:.1f}% (Screener)")
        elif sc["roce_pct"] and sc["roce_pct"] > 15:
            flags.append(f"✅ ROCE {sc['roce_pct']:.1f}% (Screener)")
        elif sc["roce_pct"] is not None and sc["roce_pct"] < 10:
            flags.append(f"⚠️ Low ROCE {sc['roce_pct']:.1f}%")

        if sc.get("rev_cagr_3yr_pct") and sc["rev_cagr_3yr_pct"] > 20:
            flags.append(f"📈 3Yr CAGR {sc['rev_cagr_3yr_pct']:.1f}% (Screener)")
        if sc.get("promoter_pct") and sc["promoter_pct"] > 70:
            flags.append(f"👥 Promoter {sc['promoter_pct']:.1f}% (Screener)")
        if sc.get("de_ratio") is not None and sc["de_ratio"] < 0.1:
            flags.append(f"🛡️ Debt-Free D/E {sc['de_ratio']:.2f} (Screener)")
        if (sc.get("rev_growth_yoy_pct") or 0) > 25:
            flags.append(f"💰 Rev Growth {sc['rev_growth_yoy_pct']:.1f}% YoY (Screener)")
        if (sc.get("pat_growth_yoy_pct") or 0) > 30:
            flags.append(f"🏆 PAT Growth {sc['pat_growth_yoy_pct']:.1f}% YoY (Screener)")

        qs = sc.get("_quarterly_sales", [])
        if len(qs) >= 3 and all(v is not None for v in qs[:3]):
            if qs[0] < qs[1] < qs[2]:
                flags.append("⚠️ Revenue declining 3 qtrs (Screener)")

        cand["screener"]       = sc
        cand["screener_flags"] = flags
        cand["all_flags"]      = flags + [
            f for f in cand.get("all_flags", []) if "(Screener)" not in f
        ]

        log.info(
            f"  ✓ {sym_clean}: ROCE={sc['roce_pct']}% "
            f"Promoter={sc['promoter_pct']}% "
            f"D/E={sc['de_ratio']} Rev={sc['rev_growth_yoy_pct']}%"
        )
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(f"Screener enrichment: {success}/{total} fetched successfully.")
    return enriched


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
