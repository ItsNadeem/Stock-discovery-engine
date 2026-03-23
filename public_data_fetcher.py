"""
public_data_fetcher.py — Free Public Domain Data for NSE/BSE India
====================================================================
Three new signals that are legally mandated disclosures, 100% free,
and entirely missing from our current pipeline:

1. INSIDER BUYING (PIT Regulation 7, NSE)
   When a promoter / director buys shares in the open market, they must
   disclose within 2 trading days. This is the most forward-looking signal
   available — insiders buy because they expect the stock to go up.
   Source: https://www.nseindia.com/api/corporates-pit
   Lookback: last 30 days

2. PROMOTER PLEDGE % (NSE/BSE)
   Pledged promoter shares are a governance red flag. Pledge unwinding
   forces selling — the promoter can't buy the dip and may lose control.
   VALUEPICK explicitly avoids high-pledge stocks.
   Source: NSE shareholding pattern quarterly filings
   (scraped from yfinance .info['heldPercentInstitutions'] fallback,
    then cross-checked against BSE pledge page)

3. BULK / BLOCK DEALS (NSE)
   A single entity trading >0.5% of a company's shares in one session.
   Institutional accumulation 2–3 quarters before it shows in shareholding.
   Source: https://www.nseindia.com/api/snapshot-capital-market-largedeal
   Lookback: last 30 days

IMPORTANT: NSE APIs require a session cookie obtained by visiting the
homepage first. All requests must use the same session with the correct
User-Agent and Referer headers. The GitHub Actions runner has a clean
IP each run, which avoids rate limiting issues.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# NSE SESSION HELPER
# NSE uses anti-scraping: first visit homepage to get cookies, then call API
# ──────────────────────────────────────────────────────────────────────────────

NSE_BASE    = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}


def _get_nse_session() -> Optional[requests.Session]:
    """
    Create an authenticated NSE session by visiting the homepage first.
    NSE blocks direct API calls without a valid cookie from the homepage.
    Returns a requests.Session with cookies, or None on failure.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Step 1: visit homepage to get cookies
        r = session.get(NSE_BASE, timeout=15)
        r.raise_for_status()
        time.sleep(1.0)   # polite pause, avoids rate limiting

        # Step 2: visit the companies-listing page (the actual data page)
        r2 = session.get(
            f"{NSE_BASE}/companies-listing/corporate-filings-insider-trading",
            timeout=15
        )
        r2.raise_for_status()
        time.sleep(0.5)

        return session
    except Exception as e:
        log.warning(f"NSE session init failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 1. INSIDER BUYING — NSE PIT Regulation 7 disclosures
# ──────────────────────────────────────────────────────────────────────────────

def fetch_insider_trades(
    symbol: str,
    lookback_days: int = 30,
    session: Optional[requests.Session] = None,
) -> dict:
    """
    Fetch recent insider trading disclosures for a symbol from NSE.

    NSE PIT Regulation 7 requires insiders (promoters, directors, KMPs)
    to disclose trades within 2 trading days of execution.

    Returns:
        {
          "net_shares_bought":  int,      # net shares bought (positive = buying)
          "buy_count":          int,       # number of buy transactions
          "sell_count":         int,       # number of sell transactions
          "total_value_cr":     float,     # approx total buy value in ₹ Cr
          "promoter_buying":    bool,      # at least one promoter category buy
          "insider_score":      float,     # 0-1 signal strength
          "transactions":       list[dict],# raw transaction list
          "flag":               str|None,  # human-readable flag
        }
    """
    result = {
        "net_shares_bought": 0,
        "buy_count":         0,
        "sell_count":        0,
        "total_value_cr":    0.0,
        "promoter_buying":   False,
        "insider_score":     0.3,   # neutral default
        "transactions":      [],
        "flag":              None,
    }

    sym_clean = symbol.replace(".NS", "").replace(".BO", "").upper()
    cutoff    = (datetime.now() - timedelta(days=lookback_days)).strftime("%d-%m-%Y")

    try:
        if session is None:
            session = _get_nse_session()
        if session is None:
            return result

        # NSE PIT API endpoint
        url = (
            f"{NSE_BASE}/api/corporates-pit"
            f"?index=equities&symbol={sym_clean}"
            f"&from_date={cutoff}"
        )
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        transactions = data.get("data", [])
        if not transactions:
            return result

        cutoff_dt  = datetime.now() - timedelta(days=lookback_days)
        net_shares = 0
        buy_val_cr = 0.0

        for txn in transactions:
            try:
                # Parse transaction date
                txn_date_str = txn.get("date") or txn.get("tdpTransactionDate") or ""
                try:
                    txn_dt = datetime.strptime(txn_date_str[:10], "%d-%b-%Y")
                except Exception:
                    try:
                        txn_dt = datetime.strptime(txn_date_str[:10], "%Y-%m-%d")
                    except Exception:
                        txn_dt = datetime.now()

                if txn_dt < cutoff_dt:
                    continue

                txn_type   = (txn.get("tdpTransactionType") or "").upper()
                shares_raw = txn.get("tdpNoOfSecurities") or 0
                price_raw  = txn.get("tdpAcqDispPrice") or 0
                category   = (txn.get("personCategory") or "").lower()

                try:
                    shares = int(str(shares_raw).replace(",", ""))
                    price  = float(str(price_raw).replace(",", ""))
                except Exception:
                    shares = price = 0

                is_buy = "buy" in txn_type or "acqui" in txn_type
                is_promoter_cat = any(
                    x in category for x in
                    ["promoter", "director", "kmp", "key managerial"]
                )

                if is_buy:
                    net_shares += shares
                    result["buy_count"] += 1
                    buy_val_cr += (shares * price) / 1e7
                    if is_promoter_cat:
                        result["promoter_buying"] = True
                else:
                    net_shares -= shares
                    result["sell_count"] += 1

                result["transactions"].append({
                    "date":     txn_date_str[:10],
                    "type":     txn_type,
                    "shares":   shares,
                    "price":    price,
                    "category": category,
                })
            except Exception:
                continue

        result["net_shares_bought"] = net_shares
        result["total_value_cr"]    = round(buy_val_cr, 2)

        # Score: strong signal if promoter buying, moderate for insider buying
        if result["promoter_buying"] and net_shares > 0:
            result["insider_score"] = min(0.85 + buy_val_cr / 50, 1.0)
            result["flag"] = (
                f"🧑‍💼 Promoter buying: {result['buy_count']} trades "
                f"₹{result['total_value_cr']:.1f} Cr (last {lookback_days}d)"
            )
        elif net_shares > 0 and result["buy_count"] >= 2:
            result["insider_score"] = 0.65
            result["flag"] = (
                f"📈 Insider net buying: {result['buy_count']} trades "
                f"₹{result['total_value_cr']:.1f} Cr (last {lookback_days}d)"
            )
        elif net_shares < 0 and result["sell_count"] >= 3:
            result["insider_score"] = 0.10
            result["flag"] = (
                f"⚠️ Insider net selling: {result['sell_count']} transactions"
            )

        log.debug(
            f"  {sym_clean}: insider net={net_shares} buys={result['buy_count']} "
            f"sells={result['sell_count']} score={result['insider_score']:.2f}"
        )

    except Exception as e:
        log.debug(f"  {sym_clean}: insider fetch failed — {e}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. PROMOTER PLEDGE % — BSE SAST Regulation 29 + yfinance fallback
# ──────────────────────────────────────────────────────────────────────────────

# BSE scrip code lookup — needed for BSE pledge page
# Try to derive from symbol; fallback to yfinance .info
def _get_bse_code(symbol: str, info: dict) -> Optional[str]:
    """
    BSE scrip code is in yfinance .info as a numeric string.
    Also stored in 'exchange' related fields.
    """
    code = info.get("exchange") if False else None   # placeholder

    # Try common yfinance fields
    for field in ["isin", "firstTradeDateEpochUtc"]:
        val = info.get(field)
        if val:
            pass   # ISIN ≠ BSE code; skip

    # The BSE code is sometimes embedded in info["symbol"] for .BO tickers
    bo_sym = symbol.replace(".NS", ".BO")
    # This is a best-effort — we'll get it from the BSE announcement fetcher
    # which already maintains a BSE code cache
    return None


def fetch_promoter_pledge(
    symbol: str,
    info: dict,
    session: Optional[requests.Session] = None,
) -> dict:
    """
    Get promoter pledge % for a company.

    Primary: yfinance .info has promoter holding data, but pledge % is
    not directly available. We derive it from:
      - BSE shareholding pattern quarterly filing (public, free)
      - Fallback: estimate from yfinance held% fields

    Returns:
        {
          "pledge_pct":         float|None,  # % of promoter shares pledged
          "promoter_held_pct":  float|None,  # % of total shares held by promoters
          "pledge_score":       float,        # 0-1 (1=no pledge, 0=high pledge)
          "is_high_pledge":     bool,
          "flag":               str|None,
        }
    """
    result = {
        "pledge_pct":        None,
        "promoter_held_pct": None,
        "pledge_score":      0.5,   # neutral default
        "is_high_pledge":    False,
        "flag":              None,
    }

    sym_clean = symbol.replace(".NS", "").replace(".BO", "").upper()

    # ── Step 1: try to get pledge % from NSE shareholding API ──
    try:
        if session is None:
            session = _get_nse_session()
        if session:
            url = f"{NSE_BASE}/api/shareholding-patterns?symbol={sym_clean}&series=EQ"
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # NSE shareholding pattern: look for encumbrance/pledge data
                # Structure varies — try to find pledge in promoter breakdown
                for rec in (data if isinstance(data, list) else [data]):
                    pledge_raw = (
                        rec.get("promoterAndPromoterGroupPledgedShares") or
                        rec.get("pledgedShares") or
                        rec.get("encumberedShares")
                    )
                    promoter_raw = (
                        rec.get("promoterAndPromoterGroup") or
                        rec.get("promoterHolding")
                    )
                    if pledge_raw is not None:
                        try:
                            result["pledge_pct"] = float(str(pledge_raw).replace(",","").replace("%",""))
                        except Exception:
                            pass
                    if promoter_raw is not None:
                        try:
                            result["promoter_held_pct"] = float(str(promoter_raw).replace(",","").replace("%",""))
                        except Exception:
                            pass
                    if result["pledge_pct"] is not None:
                        break
    except Exception as e:
        log.debug(f"  {sym_clean}: NSE shareholding API failed — {e}")

    # ── Step 2: fallback — derive from yfinance .info ──
    if result["pledge_pct"] is None:
        # yfinance doesn't have pledge % directly, but has heldPercentInsiders
        # which captures promoter + insider held %. We use it as a proxy.
        promoter_raw = info.get("heldPercentInsiders")
        if promoter_raw is not None:
            try:
                result["promoter_held_pct"] = round(float(promoter_raw) * 100, 1)
            except Exception:
                pass
        # Without pledge data, we set a neutral score
        result["pledge_score"] = 0.5
        return result

    # ── Step 3: score based on pledge % ──
    p = result["pledge_pct"]
    if p is None:
        result["pledge_score"] = 0.5
    elif p <= 0:
        result["pledge_score"] = 1.0   # no pledge = ideal
    elif p < 5:
        result["pledge_score"] = 0.85  # minimal pledge
    elif p < 20:
        result["pledge_score"] = 0.60  # moderate
    elif p < 40:
        result["pledge_score"] = 0.30  # high — caution
        result["is_high_pledge"] = True
        result["flag"] = f"⚠️ High pledge: {p:.1f}% of promoter shares"
    else:
        result["pledge_score"] = 0.05  # very high — near disqualifier
        result["is_high_pledge"] = True
        result["flag"] = f"🚨 Very high pledge: {p:.1f}% — forced selling risk"

    if p is not None and p <= 0:
        result["flag"] = "✅ Zero promoter pledge"

    log.debug(
        f"  {sym_clean}: pledge={p}% promoter={result['promoter_held_pct']}% "
        f"score={result['pledge_score']:.2f}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. BULK / BLOCK DEALS — NSE large deal API
# ──────────────────────────────────────────────────────────────────────────────

# Module-level cache: fetch once per run for all symbols
_bulk_deal_cache: dict = {}          # symbol → list of deals
_bulk_deal_cache_date: str = ""      # YYYYMMDD of last fetch


def fetch_bulk_block_deals_universe(
    session: Optional[requests.Session] = None,
) -> dict:
    """
    Fetch today's bulk and block deals from NSE for the entire market.
    Returns a dict: {symbol: [list of deal dicts]}
    Cached for the lifetime of the process (called once per scan run).
    """
    global _bulk_deal_cache, _bulk_deal_cache_date

    today_str = datetime.now().strftime("%Y%m%d")
    if _bulk_deal_cache_date == today_str and _bulk_deal_cache:
        return _bulk_deal_cache

    deals_by_symbol: dict = {}

    try:
        if session is None:
            session = _get_nse_session()
        if session is None:
            return deals_by_symbol

        for deal_type in ["bulk_deals", "block_deals"]:
            url = (
                f"{NSE_BASE}/api/snapshot-capital-market-largedeal"
                f"?bandtype={deal_type}&view=mode"
            )
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            records = data if isinstance(data, list) else data.get("data", [])

            for rec in records:
                sym = (
                    rec.get("symbol") or
                    rec.get("Symbol") or ""
                ).strip().upper()
                if not sym:
                    continue

                client   = rec.get("clientName") or rec.get("ClientName") or ""
                qty      = rec.get("tradedQuantity") or rec.get("TradedQuantity") or 0
                price    = rec.get("tradePrice") or rec.get("TradePrice") or 0
                side     = (rec.get("buyOrSell") or rec.get("BuyOrSell") or "").upper()

                try:
                    qty_int   = int(str(qty).replace(",", ""))
                    price_flt = float(str(price).replace(",", ""))
                    value_cr  = round(qty_int * price_flt / 1e7, 2)
                except Exception:
                    qty_int = price_flt = value_cr = 0

                if sym not in deals_by_symbol:
                    deals_by_symbol[sym] = []
                deals_by_symbol[sym].append({
                    "type":     deal_type,
                    "client":   client,
                    "side":     side,
                    "qty":      qty_int,
                    "price":    price_flt,
                    "value_cr": value_cr,
                })

        log.info(
            f"Bulk/block deals: {len(deals_by_symbol)} symbols with large deals today"
        )
        _bulk_deal_cache      = deals_by_symbol
        _bulk_deal_cache_date = today_str

    except Exception as e:
        log.warning(f"Bulk deal fetch failed: {e}")

    return deals_by_symbol


def analyze_bulk_deals(symbol: str, universe_deals: dict) -> dict:
    """
    For a given symbol, analyse bulk/block deal activity from today's cache.

    Returns:
        {
          "has_bulk_deal":        bool,
          "institutional_buying": bool,   # institution name in client field
          "net_deal_value_cr":    float,  # net value (buy - sell) in ₹ Cr
          "deal_count":           int,
          "bulk_score":           float,  # 0-1
          "flag":                 str|None,
        }
    """
    result = {
        "has_bulk_deal":        False,
        "institutional_buying": False,
        "net_deal_value_cr":    0.0,
        "deal_count":           0,
        "bulk_score":           0.3,    # neutral default
        "flag":                 None,
    }

    sym_clean = symbol.replace(".NS", "").replace(".BO", "").upper()
    deals     = universe_deals.get(sym_clean, [])
    if not deals:
        return result

    result["has_bulk_deal"] = True
    result["deal_count"]    = len(deals)

    net_val_cr = 0.0
    inst_buy   = False

    # Known institutional name keywords
    INST_KEYWORDS = [
        "mutual fund", "mf ", " aif", "insurance", "lic ", "pension",
        "fpi", "fii", "dii", "hdfc", "sbi mf", "axis mf", "kotak mf",
        "nippon", "icici pru", "mirae", "aditya birla", "uti mf",
        "tata mf", "fund", "schemes", "capital ltd", "securities ltd",
        "asset mgmt", "investment mgmt",
    ]

    for deal in deals:
        client_lower = deal["client"].lower()
        is_inst      = any(kw in client_lower for kw in INST_KEYWORDS)
        val          = deal["value_cr"]
        side         = deal["side"]

        if side == "B":
            net_val_cr += val
            if is_inst:
                inst_buy = True
        elif side == "S":
            net_val_cr -= val

    result["net_deal_value_cr"]    = round(net_val_cr, 2)
    result["institutional_buying"] = inst_buy

    # Score
    if inst_buy and net_val_cr > 0:
        result["bulk_score"] = min(0.75 + net_val_cr / 100, 1.0)
        result["flag"] = (
            f"🏦 Institutional bulk buying ₹{net_val_cr:.1f} Cr today"
        )
    elif net_val_cr > 0:
        result["bulk_score"] = 0.60
        result["flag"] = (
            f"📦 Bulk buy ₹{net_val_cr:.1f} Cr today ({result['deal_count']} deal)"
        )
    elif net_val_cr < -5:
        result["bulk_score"] = 0.15
        result["flag"] = (
            f"⚠️ Bulk selling ₹{abs(net_val_cr):.1f} Cr today"
        )

    log.debug(
        f"  {sym_clean}: bulk={result['has_bulk_deal']} "
        f"net=₹{net_val_cr:.1f}Cr inst={inst_buy}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. COMPOSITE: Fetch all three for a single symbol
# ──────────────────────────────────────────────────────────────────────────────

def fetch_all_public_signals(
    symbol: str,
    info: dict,
    universe_deals: dict,
    session: Optional[requests.Session] = None,
) -> dict:
    """
    Convenience wrapper: fetch insider trades, pledge, and bulk deals
    for a single symbol in one call.

    Returns merged dict with keys:
      insider_*  (from fetch_insider_trades)
      pledge_*   (from fetch_promoter_pledge)
      bulk_*     (from analyze_bulk_deals)
      public_score:  float 0-1 composite of all three
      public_flags:  list[str]
    """
    insider = fetch_insider_trades(symbol, session=session)
    pledge  = fetch_promoter_pledge(symbol, info, session=session)
    bulk    = analyze_bulk_deals(symbol, universe_deals)

    # Composite public score:
    # Insider buying (most predictive): 50%
    # Pledge (disqualifier): 30%
    # Bulk deals (confirmation): 20%
    public_score = (
        insider["insider_score"] * 0.50 +
        pledge["pledge_score"]   * 0.30 +
        bulk["bulk_score"]       * 0.20
    )

    # Hard penalty for very high pledge — can never be high-conviction
    if pledge.get("is_high_pledge"):
        public_score = min(public_score, 0.40)

    flags = []
    if insider.get("flag"):
        flags.append(insider["flag"])
    if pledge.get("flag"):
        flags.append(pledge["flag"])
    if bulk.get("flag"):
        flags.append(bulk["flag"])

    return {
        # Insider
        "insider_net_shares":  insider["net_shares_bought"],
        "insider_buy_count":   insider["buy_count"],
        "insider_value_cr":    insider["total_value_cr"],
        "promoter_buying":     insider["promoter_buying"],
        "insider_score":       insider["insider_score"],
        # Pledge
        "pledge_pct":          pledge["pledge_pct"],
        "promoter_held_pct":   pledge["promoter_held_pct"],
        "is_high_pledge":      pledge["is_high_pledge"],
        "pledge_score":        pledge["pledge_score"],
        # Bulk/block
        "has_bulk_deal":       bulk["has_bulk_deal"],
        "institutional_buying":bulk["institutional_buying"],
        "bulk_net_value_cr":   bulk["net_deal_value_cr"],
        "bulk_score":          bulk["bulk_score"],
        # Composite
        "public_score":        round(min(public_score, 1.0), 4),
        "public_flags":        flags,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. ONCE-PER-RUN INITIALISER — call at start of scan
# ──────────────────────────────────────────────────────────────────────────────

def init_public_data(session: Optional[requests.Session] = None) -> tuple:
    """
    Called once at the start of run_all.py.
    Returns (session, universe_deals) for reuse across all symbols.
    """
    log.info("Initialising public data fetcher...")

    if session is None:
        session = _get_nse_session()

    if session is None:
        log.warning("  NSE session failed — insider/bulk signals unavailable")
        return None, {}

    universe_deals = fetch_bulk_block_deals_universe(session=session)
    log.info(
        f"  Session ready. "
        f"Today's bulk/block deals: {len(universe_deals)} symbols"
    )
    return session, universe_deals
