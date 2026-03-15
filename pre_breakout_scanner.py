"""
pre_breakout_scanner.py — Layer 2: Early-Stage Multibagger Detection

Inspired by valuepicks methodology.
30-year Indian market veteran. Verified calls: Paushak ₹74→₹10,000 (140x),
Tasty Bite ₹165→₹9,420, EKI ₹150→₹10,000 (66x in 9 months).

Signals we scan for (blog + Twitter analysis):

  ORIGINAL SIGNALS (from blog posts):
  1. Promoter stake hike via preferential allotment (Jay Kay: 32%→52%)
  2. Market cap trading below liquid asset value (Max India: mcap < cash+assets)
  3. New capacity expansion announced (Cosmo Ferrites: +50% capacity)
  4. Export revenue acceleration (Cosmo: 48% export growth in one quarter)
  5. New business pivot by credible promoter group
  6. Low market cap + high promoter confidence (buybacks, promoter doesn't sell)

  NEW SIGNALS (from @valuepick Twitter / "Guess The Gem" posts):
  7. DEBT ELIMINATION — company actively paying down loans (re-rating trigger)
     Source: Twitter comment threads — "closed debt with SBI" as specific signal
  8. TRUSTED PROMOTER GROUP — Murugappa, JK Group, Alembic, Cosmo Films,
     Tata, TVS, Bajaj, Sundaram. Group pedigree = minority-friendly management.
     Source: Shanthi Gears (Murugappa), Jay Kay (JK Group), Cosmo Ferrites (Cosmo Films)
  9. DEEP VALUE vs SECTOR PEERS — P/E at fraction of industry average
     Source: "Guess The Gem" clues always included "P/E at 1/4 of industry average"

  FIXES (v2 — after first real run on 2026-03-15):
  FIX 1: Exclude pure holding/investment companies.
          BFINVEST, NAHARCAP, CORALFINAC, AFSL etc dominated first run.
          They score perfectly (zero debt, high cash, low P/B, high promoter)
          because cash IS their business — they are NOT real operating companies.
  FIX 2: Exclude passive income businesses.
          Operating margin >85% = investment/dividend income, not operations.
          Revenue growth >300% TTM = one-off gain or data artefact, not real growth.
          INDSWFTLAB showed 1641% revenue growth — base-effect distortion.

Data sources (all free, no auth):
  - BSE India corporate announcements
  - yfinance .info + quarterly financials
"""

import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────
@dataclass
class PreBreakoutConfig:
    # Universe
    max_market_cap_cr: float = 2_000.0
    min_market_cap_cr: float = 10.0
    max_price: float         = 500.0
    min_price: float         = 5.0

    # ── FIX 1: Excluded industries ──
    # Investment holding companies score perfectly on every metric but are
    # NOT real operating businesses. Excluded after first run showed they
    # dominated the entire watchlist (BFINVEST, NAHARCAP, CORALFINAC etc).
    excluded_industries: list = field(default_factory=lambda: [
        "asset management", "investment holding", "core investment company",
        "holding company", "investment company", "closed end fund",
        "venture capital", "private equity", "financial holding",
        "capital markets", "diversified financials",
    ])

    excluded_description_keywords: list = field(default_factory=lambda: [
        "core investment company", "holding company", "investment in shares",
        "investment in securities", "investment holding", "holds equity",
        "invests in shares", "holding of shares", "investment activities",
        "systematically important non-deposit", "non-deposit taking core",
    ])

    excluded_name_patterns: list = field(default_factory=lambda: [
        "investments ltd", "investment ltd", "holdings ltd",
        "capital & finance", "finance & investment",
    ])

    # ── FIX 2: Passive income thresholds ──
    # Operating margin >85% = income from investments, not operations.
    # Revenue growth >300% = almost certainly a data artefact or one-off gain.
    max_operating_margin_pct: float          = 85.0
    max_believable_revenue_growth_pct: float = 300.0

    # Promoter signals
    min_promoter_holding_pct: float    = 35.0
    promoter_hike_threshold_pct: float = 2.0

    # Asset value
    mcap_to_book_max: float = 1.5

    # Capacity expansion keywords
    capex_keywords: list = field(default_factory=lambda: [
        "capacity expansion", "new plant", "greenfield", "brownfield",
        "capacity enhancement", "capex", "new facility", "production capacity",
        "setting up", "commissioning", "commercial production",
    ])

    # Business pivot keywords
    pivot_keywords: list = field(default_factory=lambda: [
        "new business", "diversification", "joint venture", "jv",
        "technology transfer", "new segment", "new vertical",
        "strategic partnership", "new product line", "foray into",
    ])

    # Trusted promoter groups (@valuepick picks always from clean groups)
    trusted_groups: list = field(default_factory=lambda: [
        "murugappa", "jk group", "jk cement", "singhania",
        "alembic", "cosmo films", "max group",
        "tata", "tvs", "bajaj", "sundaram", "godrej",
        "mahindra", "birla", "wipro", "infosys", "hdfc",
        "kotak", "pi industries", "astral", "aarti",
        "deepak nitrite", "navin fluorine", "fine organics",
        "galaxy surfactants", "alkyl amines", "vinati organics",
    ])

    # Debt elimination thresholds
    debt_reduction_threshold: float = 0.3
    near_zero_debt_de: float        = 0.15

    # Peer P/E discount
    peer_pe_discount_threshold: float = 0.50

    sector_pe_map: dict = field(default_factory=lambda: {
        "Chemicals":           28.0,
        "Pharmaceuticals":     32.0,
        "Consumer Goods":      45.0,
        "Industrial":          22.0,
        "Technology":          30.0,
        "Auto Components":     20.0,
        "Metals":              14.0,
        "Textiles":            18.0,
        "Packaging":           25.0,
        "Specialty Chemicals": 35.0,
        "Defence":             40.0,
        "Logistics":           28.0,
        "Healthcare":          35.0,
        "Engineering":         24.0,
        "Agro Chemicals":      22.0,
        "Default":             25.0,
    })

    announcement_lookback_days: int = 90

    # Scoring weights (w_ fields sum to 1.0; undiscovered bonus is additive)
    w_promoter:  float = 0.25
    w_valuation: float = 0.20
    w_growth:    float = 0.20
    w_catalyst:  float = 0.15
    w_group:     float = 0.10
    w_debt:      float = 0.05
    w_pe_value:  float = 0.05

    top_n: int = 15


PCFG = PreBreakoutConfig()

BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
}


# ──────────────────────────────────────────────────────
# FIX 1 + FIX 2: UNIVERSE QUALITY GATE
# ──────────────────────────────────────────────────────

def should_exclude(info: dict) -> tuple[bool, str]:
    """
    Returns (True, reason) if this stock must be skipped.

    FIX 1 — Holding/investment company filter:
      These companies score perfectly on debt, cash, P/B and promoter holding
      because cash IS their business. They have zero real operating activity
      and will never deliver Cosmo Ferrites / EKI type returns.

    FIX 2 — Passive income filter:
      Operating margin >85% means income is from dividends/investments.
      Revenue growth >300% TTM is a one-off gain or data artefact.
    """
    industry     = (info.get("industry") or "").lower()
    sector       = (info.get("sector") or "").lower()
    summary      = (info.get("longBusinessSummary") or "").lower()
    company_name = (info.get("longName") or "").lower()

    # FIX 1a: Industry/sector name
    for excl in PCFG.excluded_industries:
        if excl in industry or excl in sector:
            return True, f"Excluded industry: '{excl}'"

    # FIX 1b: Business description keywords
    for kw in PCFG.excluded_description_keywords:
        if kw in summary:
            return True, f"Holding company description keyword: '{kw}'"

    # FIX 1c: Company name patterns
    for pattern in PCFG.excluded_name_patterns:
        if pattern in company_name:
            return True, f"Holding company name pattern: '{pattern}'"

    # FIX 2a: Passive income via operating margin
    oper_margin = (info.get("operatingMargins") or 0) * 100
    if oper_margin > PCFG.max_operating_margin_pct:
        return True, f"Passive income: op margin {oper_margin:.0f}% > {PCFG.max_operating_margin_pct}%"

    # FIX 2b: Unbelievable revenue growth
    rev_growth = (info.get("revenueGrowth") or 0) * 100
    if rev_growth > PCFG.max_believable_revenue_growth_pct:
        return True, f"Likely data artefact: revenue growth {rev_growth:.0f}%"

    return False, ""


# ──────────────────────────────────────────────────────
# BSE DATA FETCHERS
# ──────────────────────────────────────────────────────

def fetch_bse_announcements(symbol_code: str, days_back: int = 90) -> list[dict]:
    try:
        url = (
            f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
            f"?strCat=-1&strPrevDate=&strScrip={symbol_code}"
            f"&strSearch=P&strToDate=&strType=C&subcategory=-1"
        )
        r = requests.get(url, headers=BSE_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        cutoff = datetime.now() - timedelta(days=days_back)
        result = []
        for ann in data.get("Table", []):
            try:
                dt_str = ann.get("News_submission_dt", "")
                dt = datetime.strptime(dt_str[:10], "%Y-%m-%d") if dt_str else None
                if dt and dt >= cutoff:
                    result.append({
                        "date":     dt.strftime("%Y-%m-%d"),
                        "title":    ann.get("NEWSSUB", "").strip(),
                        "category": ann.get("CATEGORYNAME", "").strip(),
                    })
            except Exception:
                continue
        return result
    except Exception as e:
        log.debug(f"BSE announcements failed for {symbol_code}: {e}")
        return []


def get_bse_code_from_symbol(symbol: str) -> Optional[str]:
    sym_clean = symbol.replace(".NS", "")
    try:
        url = (
            f"https://api.bseindia.com/BseIndiaAPI/api/fetchCompanyList/w"
            f"?marketcap=&industry=&status=Active&scripcode="
            f"&companyname={sym_clean}&segment=Equity"
        )
        r = requests.get(url, headers=BSE_HEADERS, timeout=8)
        if r.status_code == 200:
            data  = r.json()
            items = data if isinstance(data, list) else data.get("Table", [])
            for item in items:
                if (sym_clean.lower() in str(item.get("SCRIP_CD", "")).lower() or
                        sym_clean.lower() in str(item.get("Issuer_Name", "")).lower()):
                    return str(item.get("SCRIP_CD", ""))
    except Exception as e:
        log.debug(f"BSE code lookup failed for {sym_clean}: {e}")
    return None


# ──────────────────────────────────────────────────────
# SIGNAL ANALYZERS
# ──────────────────────────────────────────────────────

def analyze_shareholding(info: dict) -> dict:
    insider_pct       = (info.get("heldPercentInsiders") or 0) * 100
    inst_pct          = (info.get("heldPercentInstitutions") or 0) * 100
    promoter_score    = min(insider_pct / 75, 1.0)
    undiscovered_score = max(1.0 - inst_pct / 30, 0.0)
    return {
        "insider_pct":         round(insider_pct, 1),
        "institutional_pct":   round(inst_pct, 1),
        "promoter_score":      round(promoter_score, 3),
        "undiscovered_score":  round(undiscovered_score, 3),
    }


def analyze_asset_value(info: dict) -> dict:
    mktcap    = info.get("marketCap") or 0
    book_val  = info.get("bookValue") or 0
    cash      = info.get("totalCash") or 0
    debt      = info.get("totalDebt") or 0
    price     = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    mktcap_cr    = mktcap / 1e7
    net_cash_cr  = (cash - debt) / 1e7
    cash_to_mcap = net_cash_cr / mktcap_cr if mktcap_cr > 0 else 0
    pb           = price / book_val if book_val > 0 else 99

    pb_score        = max(1.0 - (pb - 0.5) / 2.5, 0.0)
    cash_score      = min(max(cash_to_mcap, 0), 1.0)
    valuation_score = pb_score * 0.6 + cash_score * 0.4

    return {
        "market_cap_cr":     round(mktcap_cr, 0),
        "pb_ratio":          round(pb, 2),
        "net_cash_cr":       round(net_cash_cr, 0),
        "cash_to_mcap_pct":  round(cash_to_mcap * 100, 1),
        "valuation_score":   round(valuation_score, 3),
        "trading_near_book": pb <= PCFG.mcap_to_book_max,
    }


def analyze_growth(ticker: yf.Ticker, info: dict) -> dict:
    # Clamp revenue growth to believable range (already filtered >300% via should_exclude)
    rev_growth   = min((info.get("revenueGrowth") or 0) * 100, PCFG.max_believable_revenue_growth_pct)
    earn_growth  = (info.get("earningsGrowth") or 0) * 100
    profit_mgn   = (info.get("profitMargins") or 0) * 100

    accel_signal = False
    try:
        qf = ticker.quarterly_financials
        if qf is not None and not qf.empty:
            for idx_name in qf.index:
                if "revenue" in str(idx_name).lower():
                    rev_row = qf.loc[idx_name]
                    if len(rev_row) >= 4:
                        recent = rev_row.iloc[:2].mean()
                        prior  = rev_row.iloc[2:4].mean()
                        if prior > 0 and recent > prior * 1.15:
                            accel_signal = True
                    break
    except Exception:
        pass

    rev_score    = min(max(rev_growth, 0) / 60, 1.0)
    earn_score   = min(max(earn_growth, 0) / 80, 1.0)
    margin_score = min(max(profit_mgn, 0) / 20, 1.0)
    growth_score = min(rev_score * 0.35 + earn_score * 0.35 + margin_score * 0.30
                       + (0.2 if accel_signal else 0), 1.0)

    return {
        "revenue_growth_pct":  round(rev_growth, 1),
        "earnings_growth_pct": round(earn_growth, 1),
        "profit_margin_pct":   round(profit_mgn, 1),
        "growth_accelerating": accel_signal,
        "growth_score":        round(growth_score, 3),
    }


def analyze_promoter_group(info: dict) -> dict:
    full_text = " ".join([
        (info.get("longName") or ""),
        (info.get("sector") or ""),
        (info.get("industry") or ""),
        (info.get("longBusinessSummary") or ""),
    ]).lower()

    matched_group = None
    for group in PCFG.trusted_groups:
        if group.lower() in full_text:
            matched_group = group.title()
            break

    group_score = 0.8 if matched_group else 0.0
    if not matched_group:
        inst_pct = (info.get("heldPercentInstitutions") or 0) * 100
        if 5 <= inst_pct <= 25:
            group_score = 0.2

    return {
        "matched_group":    matched_group,
        "group_score":      round(group_score, 3),
        "is_trusted_group": matched_group is not None,
        "flag":             f"🏛️ {matched_group} Group" if matched_group else None,
    }


def analyze_debt_trajectory(ticker: yf.Ticker, info: dict) -> dict:
    current_debt  = info.get("totalDebt") or 0
    cash          = info.get("totalCash") or 0
    book_val      = info.get("bookValue") or 0
    shares        = info.get("sharesOutstanding") or 0
    total_equity  = book_val * shares if shares > 0 else 0

    current_de   = current_debt / total_equity if total_equity > 0 else 0
    net_debt_cr  = (current_debt - cash) / 1e7
    is_net_cash  = (current_debt - cash) <= 0
    is_near_zero = current_de <= PCFG.near_zero_debt_de

    debt_reducing      = False
    debt_reduction_pct = 0.0
    try:
        bs = ticker.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            debt_rows = [i for i in bs.index if "debt" in str(i).lower() or "borrowing" in str(i).lower()]
            if debt_rows:
                series = bs.loc[debt_rows[0]]
                if len(series) >= 4:
                    recent = float(series.iloc[0]) if not pd.isna(series.iloc[0]) else 0
                    prior  = float(series.iloc[3]) if not pd.isna(series.iloc[3]) else 0
                    if prior > 0 and recent < prior:
                        debt_reduction_pct = (prior - recent) / prior * 100
                        debt_reducing = debt_reduction_pct >= PCFG.debt_reduction_threshold * 100
    except Exception:
        pass

    if is_net_cash:         debt_score = 1.0
    elif is_near_zero:      debt_score = 0.8
    elif debt_reducing:     debt_score = min(debt_reduction_pct / 60, 0.7)
    else:                   debt_score = max(1.0 - current_de / 2.0, 0.0)

    flags = []
    if is_net_cash:       flags.append("🟢 NET CASH (Cash > Debt)")
    elif is_near_zero:    flags.append(f"🟢 NEAR ZERO DEBT (D/E {current_de:.2f})")
    elif debt_reducing:   flags.append(f"📉 Debt Reducing ({debt_reduction_pct:.0f}% in 4 qtrs)")

    return {
        "current_de":         round(current_de, 2),
        "net_debt_cr":        round(net_debt_cr, 0),
        "is_net_cash":        is_net_cash,
        "is_near_zero_debt":  is_near_zero,
        "debt_reducing":      debt_reducing,
        "debt_reduction_pct": round(debt_reduction_pct, 1),
        "debt_score":         round(debt_score, 3),
        "flags":              flags,
    }


def analyze_peer_pe_discount(info: dict) -> dict:
    trailing_pe = info.get("trailingPE")
    forward_pe  = info.get("forwardPE")
    sector      = info.get("sector") or "Default"
    industry    = info.get("industry") or ""

    sector_pe = PCFG.sector_pe_map.get("Default", 25.0)
    for key in PCFG.sector_pe_map:
        if key.lower() in sector.lower() or key.lower() in industry.lower():
            sector_pe = PCFG.sector_pe_map[key]
            break

    best_pe = pe_source = None
    if trailing_pe and 0 < trailing_pe < 200:
        best_pe, pe_source = trailing_pe, "TTM"
    elif forward_pe and 0 < forward_pe < 200:
        best_pe, pe_source = forward_pe, "Forward"

    if best_pe is None:
        return {
            "stock_pe": None, "sector_pe": sector_pe, "pe_ratio_to_sector": None,
            "pe_discount_pct": None, "pe_score": 0.3, "is_deep_value": False,
            "flag": None, "pe_source": None, "sector_mapped": sector,
        }

    ratio           = best_pe / sector_pe
    pe_discount_pct = (1 - ratio) * 100
    pe_score        = max(1.0 - ratio, 0.0)
    is_deep_value   = ratio <= PCFG.peer_pe_discount_threshold

    flag = None
    if ratio <= 0.25:
        flag = f"💎 EXTREME VALUE: P/E {best_pe:.1f} vs sector {sector_pe:.0f} (1/4 of peers!)"
    elif is_deep_value:
        flag = f"💰 DEEP VALUE: P/E {best_pe:.1f} vs sector {sector_pe:.0f} ({pe_discount_pct:.0f}% discount)"

    return {
        "stock_pe":           round(best_pe, 1),
        "sector_pe":          sector_pe,
        "pe_ratio_to_sector": round(ratio, 2),
        "pe_discount_pct":    round(pe_discount_pct, 1),
        "pe_score":           round(pe_score, 3),
        "is_deep_value":      is_deep_value,
        "flag":               flag,
        "pe_source":          pe_source,
        "sector_mapped":      sector,
    }


def detect_catalysts(symbol: str, announcements: list[dict]) -> dict:
    all_text = " ".join(
        (a.get("title", "") + " " + a.get("category", "")).lower()
        for a in announcements
    )

    preferential_found = any(kw in all_text for kw in [
        "preferential allotment", "preferential issue", "warrant",
        "promoter acquiring", "promoter purchase",
    ])
    capex_found   = any(kw in all_text for kw in PCFG.capex_keywords)
    pivot_found   = any(kw in all_text for kw in PCFG.pivot_keywords)
    jv_found      = any(kw in all_text for kw in ["joint venture", " jv ", "collaboration", "partnership"])
    buyback_found = any(kw in all_text for kw in ["buyback", "buy back", "share repurchase"])
    export_found  = any(kw in all_text for kw in ["export", "overseas", "international order"])
    order_found   = any(kw in all_text for kw in ["order win", "order received", "loi", "contract awarded", "new order"])

    found_catalysts = []
    catalyst_score  = 0.0

    if preferential_found:
        found_catalysts.append("📈 Promoter Preferential Allotment (Skin in game)")
        catalyst_score += 0.35
    if capex_found:
        found_catalysts.append("🏭 Capacity Expansion Announced")
        catalyst_score += 0.25
    if pivot_found:
        found_catalysts.append("🔄 New Business / JV Pivot")
        catalyst_score += 0.30
    elif jv_found:
        found_catalysts.append("🤝 Joint Venture / Partnership")
        catalyst_score += 0.20
    if buyback_found:
        found_catalysts.append("💰 Buyback Announced")
        catalyst_score += 0.20
    if export_found:
        found_catalysts.append("🌍 Export / International Order")
        catalyst_score += 0.15
    if order_found:
        found_catalysts.append("📋 New Order Win")
        catalyst_score += 0.20

    recent_30d = [
        a for a in announcements
        if (datetime.now() - datetime.strptime(a["date"], "%Y-%m-%d")).days <= 30
    ]
    if len(recent_30d) >= 3:
        found_catalysts.append(f"⚡ High Corp Activity ({len(recent_30d)} announcements/30d)")
        catalyst_score += 0.10

    return {
        "catalysts":              found_catalysts,
        "catalyst_score":         round(min(catalyst_score, 1.0), 3),
        "promoter_buying_signal": preferential_found,
        "capex_signal":           capex_found,
        "export_signal":          export_found,
        "recent_announcements":   len(announcements),
    }


def classify_price_stage(info: dict, hist: pd.DataFrame) -> str:
    if hist is None or hist.empty:
        return "UNKNOWN"
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    if price == 0:
        return "UNKNOWN"

    c        = hist["Close"]
    high_52w = c.rolling(252).max().iloc[-1]
    low_52w  = c.rolling(252).min().iloc[-1]

    pct_from_low  = (price - low_52w)  / low_52w  * 100 if low_52w  > 0 else 0
    pct_from_high = (high_52w - price) / high_52w * 100 if high_52w > 0 else 0

    if pct_from_low < 20:                        return "🟢 DEEP BASE (near 52W low)"
    elif pct_from_low < 50 and pct_from_high > 30: return "🟡 EARLY UPTREND"
    elif pct_from_high < 10:                     return "🔴 NEAR 52W HIGH (late)"
    else:                                        return "🟠 MID CYCLE"


# ──────────────────────────────────────────────────────
# COMPOSITE SCORE
# ──────────────────────────────────────────────────────

def pre_breakout_composite(
    shareholding: dict, valuation: dict, growth: dict,
    catalyst: dict, group: dict, debt: dict, pe_val: dict,
) -> float:
    score = (
        shareholding["promoter_score"] * PCFG.w_promoter  +
        valuation["valuation_score"]   * PCFG.w_valuation +
        growth["growth_score"]         * PCFG.w_growth    +
        catalyst["catalyst_score"]     * PCFG.w_catalyst  +
        group["group_score"]           * PCFG.w_group     +
        debt["debt_score"]             * PCFG.w_debt      +
        pe_val["pe_score"]             * PCFG.w_pe_value
    )
    score += shareholding["undiscovered_score"] * 0.05

    # "Guess The Gem" trifecta bonus
    if group["is_trusted_group"] and pe_val["is_deep_value"] and catalyst["catalyst_score"] >= 0.3:
        score += 0.08

    return round(min(score, 1.0), 4)


# ──────────────────────────────────────────────────────
# MAIN SCANNER
# ──────────────────────────────────────────────────────

def run_pre_breakout_scanner(symbols: list[str]) -> list[dict]:
    log.info("══════════════════════════════════════════════════════")
    log.info("  PRE-BREAKOUT SCANNER (Layer 2) v2")
    log.info("  Method: value-picks.blogspot.com + @valuepick")
    log.info("  Fixes: holding cos excluded, passive income filtered")
    log.info("══════════════════════════════════════════════════════")
    log.info(f"Scanning {len(symbols)} symbols...")

    results     = []
    excluded_ct = 0

    for i, symbol in enumerate(symbols):
        sym_clean = symbol.replace(".NS", "")
        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.info

            mktcap = (info.get("marketCap") or 0) / 1e7
            price  = info.get("currentPrice") or info.get("regularMarketPrice") or 0

            if not (PCFG.min_market_cap_cr <= mktcap <= PCFG.max_market_cap_cr):
                continue
            if not (PCFG.min_price <= price <= PCFG.max_price):
                continue

            # ── Quality gate (FIX 1 + FIX 2) ──
            skip, reason = should_exclude(info)
            if skip:
                excluded_ct += 1
                log.debug(f"  EXCLUDED {sym_clean}: {reason}")
                continue

            hist         = ticker.history(period="2y")
            shareholding = analyze_shareholding(info)
            valuation    = analyze_asset_value(info)
            growth       = analyze_growth(ticker, info)
            group        = analyze_promoter_group(info)
            debt         = analyze_debt_trajectory(ticker, info)
            pe_val       = analyze_peer_pe_discount(info)

            bse_code      = get_bse_code_from_symbol(symbol)
            announcements = []
            if bse_code:
                announcements = fetch_bse_announcements(bse_code, PCFG.announcement_lookback_days)
                time.sleep(0.3)

            catalyst = detect_catalysts(symbol, announcements)
            stage    = classify_price_stage(info, hist)

            has_signal = (
                shareholding["insider_pct"] >= PCFG.min_promoter_holding_pct or
                valuation["trading_near_book"] or
                growth["growth_accelerating"] or
                catalyst["catalyst_score"] >= 0.2 or
                group["is_trusted_group"] or
                debt["is_near_zero_debt"] or
                debt["debt_reducing"] or
                pe_val["is_deep_value"]
            )
            if not has_signal:
                continue

            composite = pre_breakout_composite(
                shareholding, valuation, growth, catalyst, group, debt, pe_val
            )

            all_flags = []
            if group["flag"]:   all_flags.append(group["flag"])
            all_flags.extend(debt["flags"])
            if pe_val["flag"]:  all_flags.append(pe_val["flag"])
            all_flags.extend(catalyst["catalysts"])

            results.append({
                "symbol":          symbol,
                "price":           round(price, 2),
                "market_cap_cr":   round(mktcap, 0),
                "price_stage":     stage,
                "composite_score": composite,
                "shareholding":    shareholding,
                "valuation":       valuation,
                "growth":          growth,
                "catalyst":        catalyst,
                "group":           group,
                "debt":            debt,
                "pe_value":        pe_val,
                "all_flags":       all_flags,
                "bse_code":        bse_code,
                "scanned_at":      datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            })

            group_tag = f"[{group['matched_group']}]" if group["is_trusted_group"] else ""
            debt_tag  = "DEBT↓" if (debt["debt_reducing"] or debt["is_near_zero_debt"]) else ""
            pe_tag    = f"PE@{pe_val['pe_ratio_to_sector']:.1f}x" if pe_val["stock_pe"] else ""
            log.info(
                f"  [{i+1:04d}] {sym_clean:<14} ₹{price:<7.0f} "
                f"MCap ₹{mktcap:.0f}Cr Score:{composite:.3f} "
                f"{group_tag} {debt_tag} {pe_tag} {stage}"
            )

        except Exception as e:
            log.debug(f"  {sym_clean}: {e}")

        time.sleep(0.4)

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    log.info(
        f"Pre-breakout scan complete. "
        f"Candidates: {len(results)}  |  Excluded (holding/passive): {excluded_ct}"
    )
    return results[:PCFG.top_n]


# ──────────────────────────────────────────────────────
# REPORT GENERATOR
# ──────────────────────────────────────────────────────

def generate_pre_breakout_report(candidates: list[dict]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═" * 72
    lines = [
        sep,
        "  PRE-BREAKOUT WATCHLIST  —  EARLY STAGE MULTIBAGGER CANDIDATES",
        f"  {now}",
        "  Signal framework: value picks",
        sep,
        "  These stocks have NOT broken out yet.",
        "  Monitor for Layer 1 price+volume confirmation before entry.",
        sep,
        "",
    ]

    for i, s in enumerate(candidates, 1):
        sh    = s["shareholding"]
        val   = s["valuation"]
        gr    = s["growth"]
        cat   = s["catalyst"]
        grp   = s.get("group", {})
        dbt   = s.get("debt", {})
        pev   = s.get("pe_value", {})
        flags = s.get("all_flags", [])

        lines += [
            f"#{i:02d}  {s['symbol']:<18}  Score: {s['composite_score']:.3f}",
            f"    Price: ₹{s['price']:<8}  MCap: ₹{s['market_cap_cr']:.0f} Cr",
            f"    Stage: {s['price_stage']}",
            "",
            f"    ── OWNERSHIP ──",
            (f"    Promoter/Insider: {sh['insider_pct']:.1f}%   "
             f"Institutions: {sh['institutional_pct']:.1f}%   "
             f"{'[UNDISCOVERED]' if sh['undiscovered_score'] > 0.6 else ''}"),
            (f"    Group: {grp.get('matched_group') or 'Unknown'}  "
             f"{'🏛️ TRUSTED GROUP' if grp.get('is_trusted_group') else ''}"),
            "",
            f"    ── VALUATION ──",
            (f"    P/B: {val['pb_ratio']:.2f}   "
             f"Net Cash: ₹{val['net_cash_cr']:.0f} Cr   "
             f"Cash/MCap: {val['cash_to_mcap_pct']:.1f}%   "
             f"{'[NEAR BOOK]' if val['trading_near_book'] else ''}"),
            (f"    P/E: {pev.get('stock_pe') or 'N/A'}   "
             f"Sector P/E: {pev.get('sector_pe', '?')}   "
             f"Ratio: {pev.get('pe_ratio_to_sector') or 'N/A'}x   "
             f"{'💎 DEEP VALUE' if pev.get('is_deep_value') else ''}"),
            "",
            f"    ── DEBT ──",
            (f"    D/E: {dbt.get('current_de', '?')}   "
             f"Net Debt: ₹{dbt.get('net_debt_cr', '?')} Cr   "
             f"Reducing: {'✓ ' + str(round(dbt.get('debt_reduction_pct', 0), 0)) + '%' if dbt.get('debt_reducing') else '✗'}   "
             f"{'🟢 NET CASH' if dbt.get('is_net_cash') else ''}"),
            "",
            f"    ── GROWTH ──",
            (f"    Rev: {gr['revenue_growth_pct']:.1f}%   "
             f"Earnings: {gr['earnings_growth_pct']:.1f}%   "
             f"Margin: {gr['profit_margin_pct']:.1f}%   "
             f"Accelerating: {'✓' if gr['growth_accelerating'] else '✗'}"),
            "",
        ]

        if flags:
            lines.append(f"    ── SIGNALS ({len(flags)}) ──")
            for flag in flags[:6]:
                lines.append(f"    ▶  {flag}")
        else:
            lines.append("    ── No specific catalyst in last 90 days ──")

        lines += ["", "─" * 72, ""]

    lines += [
        sep,
        "  ✓ Is there a clear sector tailwind (ageing, EV, green energy, manufacturing)?",
        "  ✓ Is management track record clean? (annual report + news search)",
        "  ✓ Is this the only/first listed company in its niche?",
        "  ✓ Is promoter hiking stake with own money (not ESOPs)?",
        "  ✓ Are concall transcripts specific and confident (not vague)?",
        "  ✓ Wait for Layer 1 breakout confirmation before entry.",
        "  ✓ Have patience. His best picks took 2–10 years to play out.",
        sep,
        "  ⚠  Educational use only. Not financial advice.",
        sep,
    ]
    return "\n".join(lines)
