"""
public_data_fetcher.py — Free Public Domain Data for NSE/BSE India

FIX (2026-03-30): NSE blocks GitHub Actions runner IPs with 403.
The session init was retrying every symbol (2+ times each = 4000+ failed
requests per run, flooding the log with warnings).

Fix: init_public_data() now tries once with a longer timeout and retries
with a different User-Agent. If it still fails, it returns (None, {}) and
all downstream calls gracefully return neutral defaults — no more per-symbol
session attempts.

The NSE block is a known issue with GitHub-hosted runners. Options:
  1. Use a self-hosted runner (best fix, out of scope here)
  2. Accept that insider/bulk signals are unavailable on hosted runners
     and treat them as neutral (0.3 score) — implemented here

Unchanged: all signal logic, scoring, pledge fetch.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"

# Multiple UA strings to try — NSE blocks some but not all
NSE_UA_OPTIONS = [
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"),
    "python-requests/2.31.0",
]


def _get_nse_session() -> Optional[requests.Session]:
    """
    Try to create an NSE session. Returns None if all attempts fail.
    Only called ONCE per run from init_public_data().
    """
    for ua in NSE_UA_OPTIONS:
        session = requests.Session()
        session.headers.update({
            "User-Agent": ua,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/",
            "Connection": "keep-alive",
        })
        try:
            r = session.get(NSE_BASE, timeout=20)
            if r.status_code == 200:
                time.sleep(1.0)
                r2 = session.get(
                    f"{NSE_BASE}/companies-listing/corporate-filings-insider-trading",
                    timeout=20
                )
                if r2.status_code == 200:
                    time.sleep(0.5)
                    log.info("  NSE session established")
                    return session
        except Exception as e:
            log.debug(f"NSE session attempt failed ({ua[:30]}…): {e}")

    log.warning(
        "NSE session failed — insider/bulk signals unavailable. "
        "This is expected on GitHub-hosted runners (NSE blocks their IPs). "
        "Use a self-hosted runner for live insider data."
    )
    return None


# ─── Insider trades ──────────────────────────────────────────────────────────

def fetch_insider_trades(symbol: str, lookback_days: int = 30,
                         session: Optional[requests.Session] = None) -> dict:
    result = {
        "net_shares_bought": 0, "buy_count": 0, "sell_count": 0,
        "total_value_cr": 0.0, "promoter_buying": False,
        "insider_score": 0.3, "transactions": [], "flag": None,
    }
    if session is None:
        return result  # no session = neutral, no per-symbol retry

    sym_clean = symbol.replace(".NS","").replace(".BO","").upper()
    cutoff    = (datetime.now() - timedelta(days=lookback_days)).strftime("%d-%m-%Y")

    try:
        url  = (f"{NSE_BASE}/api/corporates-pit"
                f"?index=equities&symbol={sym_clean}&from_date={cutoff}")
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        transactions = resp.json().get("data", [])

        cutoff_dt = datetime.now() - timedelta(days=lookback_days)
        net_shares  = 0
        buy_val_cr  = 0.0

        for txn in transactions:
            try:
                dt_str = txn.get("date") or txn.get("tdpTransactionDate") or ""
                try:    txn_dt = datetime.strptime(dt_str[:10], "%d-%b-%Y")
                except: txn_dt = datetime.strptime(dt_str[:10], "%Y-%m-%d")
                if txn_dt < cutoff_dt:
                    continue

                txn_type = (txn.get("tdpTransactionType") or "").upper()
                shares_raw = txn.get("tdpNoOfSecurities") or 0
                price_raw  = txn.get("tdpAcqDispPrice") or 0
                category   = (txn.get("personCategory") or "").lower()
                shares = int(str(shares_raw).replace(",",""))
                price  = float(str(price_raw).replace(",",""))
                is_buy = "buy" in txn_type or "acqui" in txn_type
                is_promoter = any(x in category for x in
                                  ["promoter","director","kmp","key managerial"])
                if is_buy:
                    net_shares += shares; result["buy_count"] += 1
                    buy_val_cr += (shares * price) / 1e7
                    if is_promoter: result["promoter_buying"] = True
                else:
                    net_shares -= shares; result["sell_count"] += 1
                result["transactions"].append({
                    "date": dt_str[:10], "type": txn_type,
                    "shares": shares, "price": price, "category": category,
                })
            except Exception:
                continue

        result["net_shares_bought"] = net_shares
        result["total_value_cr"]    = round(buy_val_cr, 2)

        if result["promoter_buying"] and net_shares > 0:
            result["insider_score"] = min(0.85 + buy_val_cr / 50, 1.0)
            result["flag"] = (f"🧑‍💼 Promoter buying: {result['buy_count']} trades "
                              f"₹{result['total_value_cr']:.1f}Cr")
        elif net_shares > 0 and result["buy_count"] >= 2:
            result["insider_score"] = 0.65
            result["flag"] = (f"📈 Insider net buying: {result['buy_count']} trades "
                              f"₹{result['total_value_cr']:.1f}Cr")
        elif net_shares < 0 and result["sell_count"] >= 3:
            result["insider_score"] = 0.10
            result["flag"] = f"⚠️ Insider net selling: {result['sell_count']} txns"

    except Exception as e:
        log.debug(f"  {sym_clean}: insider fetch failed — {e}")

    return result


# ─── Promoter pledge ─────────────────────────────────────────────────────────

def fetch_promoter_pledge(symbol: str, info: dict,
                          session: Optional[requests.Session] = None) -> dict:
    result = {
        "pledge_pct": None, "promoter_held_pct": None,
        "pledge_score": 0.5, "is_high_pledge": False, "flag": None,
    }
    sym_clean = symbol.replace(".NS","").replace(".BO","").upper()

    if session:
        try:
            url  = f"{NSE_BASE}/api/shareholding-patterns?symbol={sym_clean}&series=EQ"
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                for rec in (data if isinstance(data, list) else [data]):
                    pledge_raw = (rec.get("promoterAndPromoterGroupPledgedShares") or
                                  rec.get("pledgedShares") or rec.get("encumberedShares"))
                    if pledge_raw is not None:
                        result["pledge_pct"] = float(
                            str(pledge_raw).replace(",","").replace("%",""))
                        break
        except Exception as e:
            log.debug(f"  {sym_clean}: NSE shareholding failed — {e}")

    if result["pledge_pct"] is None:
        raw = info.get("heldPercentInsiders")
        if raw is not None:
            result["promoter_held_pct"] = round(float(raw) * 100, 1)
        return result

    p = result["pledge_pct"]
    if p <= 0:
        result["pledge_score"] = 1.0; result["flag"] = "✅ Zero promoter pledge"
    elif p < 5:
        result["pledge_score"] = 0.85
    elif p < 20:
        result["pledge_score"] = 0.60
    elif p < 40:
        result["pledge_score"] = 0.30; result["is_high_pledge"] = True
        result["flag"] = f"⚠️ High pledge: {p:.1f}%"
    else:
        result["pledge_score"] = 0.05; result["is_high_pledge"] = True
        result["flag"] = f"🚨 Very high pledge: {p:.1f}%"

    return result


# ─── Bulk / block deals ──────────────────────────────────────────────────────

_bulk_deal_cache:      dict = {}
_bulk_deal_cache_date: str  = ""


def fetch_bulk_block_deals_universe(session: Optional[requests.Session] = None) -> dict:
    global _bulk_deal_cache, _bulk_deal_cache_date
    today_str = datetime.now().strftime("%Y%m%d")
    if _bulk_deal_cache_date == today_str and _bulk_deal_cache:
        return _bulk_deal_cache

    deals: dict = {}
    if session is None:
        return deals

    try:
        for deal_type in ["bulk_deals", "block_deals"]:
            url  = (f"{NSE_BASE}/api/snapshot-capital-market-largedeal"
                    f"?bandtype={deal_type}&view=mode")
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            data    = resp.json()
            records = data if isinstance(data, list) else data.get("data", [])
            for rec in records:
                sym = (rec.get("symbol") or rec.get("Symbol") or "").strip().upper()
                if not sym: continue
                client = rec.get("clientName") or rec.get("ClientName") or ""
                qty    = rec.get("tradedQuantity") or rec.get("TradedQuantity") or 0
                price  = rec.get("tradePrice") or rec.get("TradePrice") or 0
                side   = (rec.get("buyOrSell") or rec.get("BuyOrSell") or "").upper()
                try:
                    qi = int(str(qty).replace(",",""))
                    pf = float(str(price).replace(",",""))
                    vc = round(qi * pf / 1e7, 2)
                except Exception:
                    qi = pf = vc = 0
                if sym not in deals: deals[sym] = []
                deals[sym].append({
                    "type": deal_type, "client": client,
                    "side": side, "qty": qi, "price": pf, "value_cr": vc,
                })
        log.info(f"Bulk/block deals: {len(deals)} symbols with large deals today")
        _bulk_deal_cache      = deals
        _bulk_deal_cache_date = today_str
    except Exception as e:
        log.warning(f"Bulk deal fetch failed: {e}")

    return deals


def analyze_bulk_deals(symbol: str, universe_deals: dict) -> dict:
    result = {
        "has_bulk_deal": False, "institutional_buying": False,
        "net_deal_value_cr": 0.0, "deal_count": 0,
        "bulk_score": 0.3, "flag": None,
    }
    sym_clean = symbol.replace(".NS","").replace(".BO","").upper()
    deals = universe_deals.get(sym_clean, [])
    if not deals:
        return result

    result["has_bulk_deal"] = True
    result["deal_count"]    = len(deals)

    INST_KW = ["mutual fund","mf ","aif","insurance","lic ","pension","fpi","fii",
               "dii","hdfc","sbi mf","axis mf","kotak mf","nippon","icici pru",
               "mirae","aditya birla","uti mf","tata mf","fund","schemes",
               "capital ltd","securities ltd","asset mgmt","investment mgmt"]

    net_val_cr = 0.0; inst_buy = False
    for deal in deals:
        cl = deal["client"].lower()
        is_inst = any(kw in cl for kw in INST_KW)
        val = deal["value_cr"]
        if deal["side"] == "B":
            net_val_cr += val
            if is_inst: inst_buy = True
        elif deal["side"] == "S":
            net_val_cr -= val

    result["net_deal_value_cr"]    = round(net_val_cr, 2)
    result["institutional_buying"] = inst_buy

    if inst_buy and net_val_cr > 0:
        result["bulk_score"] = min(0.75 + net_val_cr / 100, 1.0)
        result["flag"] = f"🏦 Institutional bulk buying ₹{net_val_cr:.1f}Cr"
    elif net_val_cr > 0:
        result["bulk_score"] = 0.60
        result["flag"] = f"📦 Bulk buy ₹{net_val_cr:.1f}Cr"
    elif net_val_cr < -5:
        result["bulk_score"] = 0.15
        result["flag"] = f"⚠️ Bulk selling ₹{abs(net_val_cr):.1f}Cr"

    return result


# ─── Composite ───────────────────────────────────────────────────────────────

def fetch_all_public_signals(symbol: str, info: dict, universe_deals: dict,
                             session: Optional[requests.Session] = None) -> dict:
    insider = fetch_insider_trades(symbol, session=session)
    pledge  = fetch_promoter_pledge(symbol, info, session=session)
    bulk    = analyze_bulk_deals(symbol, universe_deals)

    public_score = (
        insider["insider_score"] * 0.50 +
        pledge["pledge_score"]   * 0.30 +
        bulk["bulk_score"]       * 0.20
    )
    if pledge.get("is_high_pledge"):
        public_score = min(public_score, 0.40)

    flags = [f for f in [insider.get("flag"), pledge.get("flag"), bulk.get("flag")]
             if f]

    return {
        "insider_net_shares":    insider["net_shares_bought"],
        "insider_buy_count":     insider["buy_count"],
        "insider_value_cr":      insider["total_value_cr"],
        "promoter_buying":       insider["promoter_buying"],
        "insider_score":         insider["insider_score"],
        "pledge_pct":            pledge["pledge_pct"],
        "promoter_held_pct":     pledge["promoter_held_pct"],
        "is_high_pledge":        pledge["is_high_pledge"],
        "pledge_score":          pledge["pledge_score"],
        "has_bulk_deal":         bulk["has_bulk_deal"],
        "institutional_buying":  bulk["institutional_buying"],
        "bulk_net_value_cr":     bulk["net_deal_value_cr"],
        "bulk_score":            bulk["bulk_score"],
        "public_score":          round(min(public_score, 1.0), 4),
        "public_flags":          flags,
    }


# ─── Once-per-run initialiser ─────────────────────────────────────────────────

def init_public_data(session: Optional[requests.Session] = None) -> tuple:
    """
    Called once at run start. Returns (session, universe_deals).
    If NSE is blocked, returns (None, {}) — all signals return neutral.
    """
    log.info("Initialising public data fetcher...")
    if session is None:
        session = _get_nse_session()

    if session is None:
        return None, {}

    universe_deals = fetch_bulk_block_deals_universe(session=session)
    log.info(f"  Session ready. Bulk/block deals: {len(universe_deals)} symbols today")
    return session, universe_deals
