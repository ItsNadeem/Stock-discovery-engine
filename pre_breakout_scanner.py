"""
pre_breakout_scanner.py — Layer 2: Early-Stage Multibagger Detection

Inspired by value-picks.blogspot.com + @valuepick (Twitter) methodology.
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
     Source: "Guess The Gem" clues always included "P/E at ¼ of industry average"

Data sources (all free, no auth):
  - BSE India corporate announcements
  - yfinance .info + quarterly financials
  - NSE shareholding pattern
"""

import json
import logging
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────
@dataclass
class PreBreakoutConfig:
    # Universe
    max_market_cap_cr: float  = 2_000.0   # Tighter: ≤ ₹2,000 Cr for early-stage
    min_market_cap_cr: float  = 10.0      # Avoid shells
    max_price: float          = 500.0     # Early-stage stocks rarely expensive
    min_price: float          = 5.0

    # Promoter signals
    min_promoter_holding_pct: float = 35.0
    promoter_hike_threshold_pct: float = 2.0   # Meaningful if holding rose ≥2pp QoQ

    # Asset value signal (Max India style)
    mcap_to_book_max: float = 1.5      # Flag if trading near/below book

    # Export growth signal
    min_export_growth_pct: float = 30.0

    # Capacity expansion keyword triggers
    capex_keywords: list = field(default_factory=lambda: [
        "capacity expansion", "new plant", "greenfield", "brownfield",
        "capacity enhancement", "capex", "new facility", "production capacity",
        "setting up", "commissioning", "commercial production"
    ])

    # Sector pivot / new business keywords
    pivot_keywords: list = field(default_factory=lambda: [
        "new business", "diversification", "joint venture", "jv",
        "technology transfer", "new segment", "new vertical",
        "strategic partnership", "new product line", "foray into"
    ])

    # ── NEW: Trusted Promoter Groups (from @valuepick Twitter analysis) ──
    # He specifically called out Murugappa (Shanthi Gears), JK Group (Jay Kay),
    # Cosmo Films group (Cosmo Ferrites), Alembic (Paushak) as minority-friendly.
    # Full list expanded from his 15 years of picks.
    trusted_groups: list = field(default_factory=lambda: [
        # Mentioned directly in his picks
        "murugappa", "jk group", "jk cement", "singhania",
        "alembic", "cosmo films", "max group", "max india",
        # Other clean, minority-friendly groups from Indian market consensus
        "tata", "tvs", "bajaj", "sundaram", "godrej",
        "mahindra", "birla", "wipro", "infosys", "hdfc",
        "kotak", "pi industries", "astral", "aarti",
        "deepak nitrite", "navin fluorine", "fine organics",
        "galaxy surfactants", "alkyl amines", "vinati organics",
    ])

    # ── NEW: Debt elimination thresholds (from Twitter comment threads) ──
    # "Fruits of turnaround only for those with patience"
    # Key signal: D/E falling meaningfully QoQ, or near-zero debt achieved
    debt_reduction_threshold: float = 0.3    # D/E dropped by ≥30% YoY = signal
    near_zero_debt_de: float = 0.15          # D/E ≤ 0.15 = "nearly debt-free" flag

    # ── NEW: Peer P/E discount (from "Guess The Gem" clues) ──
    # He always noted stocks trading at ¼ of sector P/E as a key clue
    peer_pe_discount_threshold: float = 0.50  # Trading at ≤50% of sector median P/E

    # Sector P/E medians (NSE India data, updated periodically)
    # Source: niftyindices.com sector P/E
    sector_pe_map: dict = field(default_factory=lambda: {
        "Chemicals":          28.0,
        "Pharmaceuticals":    32.0,
        "Consumer Goods":     45.0,
        "Industrial":         22.0,
        "Technology":         30.0,
        "Auto Components":    20.0,
        "Metals":             14.0,
        "Textiles":           18.0,
        "Packaging":          25.0,
        "Specialty Chemicals":35.0,
        "Defence":            40.0,
        "Logistics":          28.0,
        "Healthcare":         35.0,
        "Engineering":        24.0,
        "Agro Chemicals":     22.0,
        "Default":            25.0,   # fallback
    })

    # BSE announcement lookback
    announcement_lookback_days: int = 90

    # Scoring weights — rebalanced for 3 new signals
    w_promoter:  float = 0.25
    w_valuation: float = 0.20
    w_growth:    float = 0.20
    w_catalyst:  float = 0.15
    w_group:     float = 0.10   # NEW: trusted group bonus
    w_debt:      float = 0.05   # NEW: debt elimination
    w_pe_value:  float = 0.05   # NEW: peer P/E discount

    top_n: int = 15


PCFG = PreBreakoutConfig()

BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
}


# ──────────────────────────────────────────────────────
# BSE DATA FETCHERS
# ──────────────────────────────────────────────────────

def fetch_bse_bulk_deals(days_back: int = 30) -> pd.DataFrame:
    """
    Fetch BSE bulk deals — signals institutional / promoter accumulation.
    Returns dataframe of recent bulk deals.
    """
    try:
        url = "https://www.bseindia.com/markets/equity/EQReports/bulk_deals.aspx"
        # BSE bulk deals CSV endpoint
        csv_url = "https://www.bseindia.com/download/BhavCopy/Equity/bulk_deals_{date}.csv"
        
        # Use BSE API endpoint for bulk deals
        api_url = "https://api.bseindia.com/BseIndiaAPI/api/BulkDealData/w?flag=0"
        r = requests.get(api_url, headers=BSE_HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                return df
    except Exception as e:
        log.debug(f"BSE bulk deals fetch failed: {e}")

    # Fallback: return empty dataframe
    return pd.DataFrame()


def fetch_bse_announcements(symbol_code: str, days_back: int = 90) -> list[dict]:
    """
    Fetch recent BSE corporate announcements for a stock.
    Returns list of announcement dicts with title + date.
    """
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
        announcements = data.get("Table", [])
        cutoff = datetime.now() - timedelta(days=days_back)
        result = []
        for ann in announcements:
            try:
                dt_str = ann.get("News_submission_dt", "")
                dt = datetime.strptime(dt_str[:10], "%Y-%m-%d") if dt_str else None
                if dt and dt >= cutoff:
                    result.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "title": ann.get("NEWSSUB", "").strip(),
                        "category": ann.get("CATEGORYNAME", "").strip(),
                    })
            except Exception:
                continue
        return result
    except Exception as e:
        log.debug(f"BSE announcements fetch failed for {symbol_code}: {e}")
        return []


def get_bse_code_from_symbol(symbol: str) -> Optional[str]:
    """Map NSE symbol to BSE code using BSE search API."""
    sym_clean = symbol.replace(".NS", "")
    try:
        url = f"https://api.bseindia.com/BseIndiaAPI/api/fetchCompanyList/w?marketcap=&industry=&status=Active&scripcode=&companyname={sym_clean}&segment=Equity"
        r = requests.get(url, headers=BSE_HEADERS, timeout=8)
        if r.status_code == 200:
            data = r.json()
            items = data if isinstance(data, list) else data.get("Table", [])
            for item in items:
                name_match = sym_clean.lower() in str(item.get("SCRIP_CD", "")).lower() or \
                             sym_clean.lower() in str(item.get("Issuer_Name", "")).lower()
                if name_match:
                    return str(item.get("SCRIP_CD", ""))
    except Exception as e:
        log.debug(f"BSE code lookup failed for {sym_clean}: {e}")
    return None


# ──────────────────────────────────────────────────────
# SHAREHOLDING PATTERN ANALYZER
# ──────────────────────────────────────────────────────

def analyze_shareholding(info: dict) -> dict:
    """
    Extract promoter signals from yfinance info.
    yfinance gives heldPercentInsiders as a proxy for promoter holding.
    """
    insider_pct = (info.get("heldPercentInsiders") or 0) * 100
    inst_pct = (info.get("heldPercentInstitutions") or 0) * 100

    # High insider = promoter confidence
    # Low institutional = undiscovered by funds
    promoter_score = min(insider_pct / 75, 1.0)
    undiscovered_score = max(1.0 - inst_pct / 30, 0.0)  # Higher if institutions haven't piled in

    return {
        "insider_pct": round(insider_pct, 1),
        "institutional_pct": round(inst_pct, 1),
        "promoter_score": round(promoter_score, 3),
        "undiscovered_score": round(undiscovered_score, 3),
    }


# ──────────────────────────────────────────────────────
# VALUATION vs ASSET VALUE (Max India Signal)
# ──────────────────────────────────────────────────────

def analyze_asset_value(info: dict) -> dict:
    """
    Check if stock is trading near/below asset value.
    Max India was trading BELOW its cash + monetizable assets.
    """
    mktcap = info.get("marketCap") or 0
    book_val = info.get("bookValue") or 0
    shares = info.get("sharesOutstanding") or 0
    cash = info.get("totalCash") or 0
    total_debt = info.get("totalDebt") or 0

    mktcap_cr = mktcap / 1e7
    book_per_share = book_val
    total_book_cr = (book_per_share * shares) / 1e7 if shares > 0 else 0
    net_cash_cr = (cash - total_debt) / 1e7

    # P/B ratio
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    pb = price / book_per_share if book_per_share > 0 else 99

    # Net cash as % of market cap (Max India signal)
    cash_to_mcap = net_cash_cr / mktcap_cr if mktcap_cr > 0 else 0

    # Score: lower P/B and higher cash coverage = better
    pb_score = max(1.0 - (pb - 0.5) / 2.5, 0.0)   # Best if P/B < 0.5, worst if > 3
    cash_score = min(cash_to_mcap, 1.0)

    valuation_score = pb_score * 0.6 + cash_score * 0.4

    return {
        "market_cap_cr": round(mktcap_cr, 0),
        "pb_ratio": round(pb, 2),
        "net_cash_cr": round(net_cash_cr, 0),
        "cash_to_mcap_pct": round(cash_to_mcap * 100, 1),
        "valuation_score": round(valuation_score, 3),
        "trading_near_book": pb <= PCFG.mcap_to_book_max,
    }


# ──────────────────────────────────────────────────────
# GROWTH TRAJECTORY ANALYZER
# ──────────────────────────────────────────────────────

def analyze_growth(ticker: yf.Ticker, info: dict) -> dict:
    """
    Look for accelerating growth — the Cosmo Ferrites / EKI signal.
    Key: recent quarter growth >> historical average.
    """
    rev_growth = (info.get("revenueGrowth") or 0) * 100
    earn_growth = (info.get("earningsGrowth") or 0) * 100
    profit_margin = (info.get("profitMargins") or 0) * 100
    oper_margin = (info.get("operatingMargins") or 0) * 100

    # Try to get quarterly financials for acceleration check
    accel_signal = False
    quarters_improving = 0
    try:
        qf = ticker.quarterly_financials
        if qf is not None and not qf.empty:
            # Check if last 2 quarters better than prior 2
            rev_row = None
            for idx_name in qf.index:
                if "revenue" in str(idx_name).lower() or "total revenue" in str(idx_name).lower():
                    rev_row = qf.loc[idx_name]
                    break
            if rev_row is not None and len(rev_row) >= 4:
                recent_avg = rev_row.iloc[:2].mean()
                prior_avg = rev_row.iloc[2:4].mean()
                if prior_avg > 0 and recent_avg > prior_avg * 1.15:
                    accel_signal = True
                    quarters_improving = 2
    except Exception:
        pass

    # Score
    rev_score  = min(max(rev_growth, 0) / 60, 1.0)
    earn_score = min(max(earn_growth, 0) / 80, 1.0)
    margin_score = min(max(profit_margin, 0) / 20, 1.0)
    accel_bonus = 0.2 if accel_signal else 0.0

    growth_score = min(rev_score * 0.35 + earn_score * 0.35 + margin_score * 0.30 + accel_bonus, 1.0)

    return {
        "revenue_growth_pct": round(rev_growth, 1),
        "earnings_growth_pct": round(earn_growth, 1),
        "profit_margin_pct": round(profit_margin, 1),
        "operating_margin_pct": round(oper_margin, 1),
        "growth_accelerating": accel_signal,
        "growth_score": round(growth_score, 3),
    }


# ──────────────────────────────────────────────────────
# CATALYST DETECTOR (BSE Announcement Scanner)
# ──────────────────────────────────────────────────────

def detect_catalysts(symbol: str, announcements: list[dict]) -> dict:
    """
    Scan recent BSE announcements for value-picks style catalysts.
    Looks for: capex, JV, new business pivot, capacity expansion, export deals.
    """
    sym_clean = symbol.replace(".NS", "")
    found_catalysts = []
    catalyst_score = 0.0

    all_text = " ".join(
        (a.get("title", "") + " " + a.get("category", "")).lower()
        for a in announcements
    )

    # Check each catalyst type
    capex_found = any(kw in all_text for kw in PCFG.capex_keywords)
    pivot_found = any(kw in all_text for kw in PCFG.pivot_keywords)

    jv_found = any(
        kw in all_text for kw in ["joint venture", " jv ", "collaboration", "partnership"]
    )
    export_found = any(
        kw in all_text for kw in ["export", "overseas", "international order", "foreign"]
    )
    buyback_found = any(
        kw in all_text for kw in ["buyback", "buy back", "share repurchase"]
    )
    preferential_found = any(
        kw in all_text for kw in [
            "preferential allotment", "preferential issue",
            "warrant", "promoter acquiring", "promoter purchase"
        ]
    )
    order_win_found = any(
        kw in all_text for kw in [
            "order win", "order received", "letter of intent",
            "loi", "contract awarded", "new order"
        ]
    )

    if capex_found:
        found_catalysts.append("🏭 Capacity Expansion Announced")
        catalyst_score += 0.25
    if pivot_found:
        found_catalysts.append("🔄 New Business / JV Pivot")
        catalyst_score += 0.30
    if jv_found and not pivot_found:
        found_catalysts.append("🤝 Joint Venture / Partnership")
        catalyst_score += 0.20
    if preferential_found:
        found_catalysts.append("📈 Promoter Preferential Allotment (Skin in game)")
        catalyst_score += 0.35  # Strongest signal per blog
    if buyback_found:
        found_catalysts.append("💰 Buyback Announced")
        catalyst_score += 0.20
    if export_found:
        found_catalysts.append("🌍 Export / International Order")
        catalyst_score += 0.15
    if order_win_found:
        found_catalysts.append("📋 New Order Win")
        catalyst_score += 0.20

    # Recent announcement count as proxy for corporate activity
    recent_30d = [a for a in announcements
                  if (datetime.now() - datetime.strptime(a["date"], "%Y-%m-%d")).days <= 30]
    if len(recent_30d) >= 3:
        found_catalysts.append(f"⚡ High Corp Activity ({len(recent_30d)} announcements/30d)")
        catalyst_score += 0.10

    return {
        "catalysts": found_catalysts,
        "catalyst_score": round(min(catalyst_score, 1.0), 3),
        "capex_signal": capex_found,
        "pivot_signal": pivot_found,
        "promoter_buying_signal": preferential_found,
        "export_signal": export_found,
        "recent_announcements": len(announcements),
    }


# ──────────────────────────────────────────────────────
# NEW DETECTOR 1: TRUSTED PROMOTER GROUP
# Source: @valuepick Twitter — Shanthi Gears (Murugappa), Jay Kay (JK Group),
#         Cosmo Ferrites (Cosmo Films group), Paushak (Alembic group)
# ──────────────────────────────────────────────────────

def analyze_promoter_group(info: dict) -> dict:
    """
    Check if the company belongs to a trusted, minority-friendly business group.
    VALUEPICK consistently picks from clean promoter groups — this is not
    accidental. It filters out fraud risk which is rampant in small caps.
    """
    company_name = (info.get("longName") or info.get("shortName") or "").lower()
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    summary = (info.get("longBusinessSummary") or "").lower()

    full_text = f"{company_name} {sector} {industry} {summary}"

    matched_group = None
    for group in PCFG.trusted_groups:
        if group.lower() in full_text:
            matched_group = group.title()
            break

    # Also check for "group" keyword near clean markers
    # Some companies name themselves after the group directly
    group_score = 0.8 if matched_group else 0.0

    # Partial credit: even without name match, check for structural signals
    # of a professionally-run company (auditor quality, etc.)
    if not matched_group:
        # Proxy: if institutional holding is moderate (5–25%) it suggests
        # some institutional due diligence has been done
        inst_pct = (info.get("heldPercentInstitutions") or 0) * 100
        if 5 <= inst_pct <= 25:
            group_score = 0.2  # Slight confidence from institutional presence

    return {
        "matched_group": matched_group,
        "group_score": round(group_score, 3),
        "is_trusted_group": matched_group is not None,
        "flag": f"🏛️ {matched_group} Group" if matched_group else None,
    }


# ──────────────────────────────────────────────────────
# NEW DETECTOR 2: DEBT ELIMINATION TRAJECTORY
# Source: @valuepick Twitter comments — "closed debt with SBI",
#         "fruits of turnaround only for those with patience"
#         He watched debt paydown for YEARS before the re-rating happened.
# ──────────────────────────────────────────────────────

def analyze_debt_trajectory(ticker: yf.Ticker, info: dict) -> dict:
    """
    Detect companies actively paying down debt — a key re-rating trigger.
    VALUEPICK's turnaround picks always had debt elimination as the thesis.
    The re-rating happens when D/E crosses from "leveraged" to "clean".
    """
    current_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0
    equity = info.get("bookValue") or 0
    shares = info.get("sharesOutstanding") or 0
    total_equity_val = equity * shares if shares > 0 else 0

    current_de = current_debt / total_equity_val if total_equity_val > 0 else 0
    net_debt = current_debt - total_cash
    net_debt_cr = net_debt / 1e7

    is_near_zero_debt = current_de <= PCFG.near_zero_debt_de
    is_net_cash = net_debt <= 0  # Cash > Debt — very strong signal

    # Try to get historical debt trend from balance sheet
    debt_reducing = False
    debt_reduction_pct = 0.0
    prior_de = None

    try:
        bs = ticker.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            debt_rows = [idx for idx in bs.index
                        if "debt" in str(idx).lower() or "borrowing" in str(idx).lower()]
            if debt_rows:
                debt_series = bs.loc[debt_rows[0]]
                # Compare most recent to 4 quarters ago
                if len(debt_series) >= 4:
                    recent = float(debt_series.iloc[0]) if not pd.isna(debt_series.iloc[0]) else 0
                    prior  = float(debt_series.iloc[3]) if not pd.isna(debt_series.iloc[3]) else 0
                    if prior > 0 and recent < prior:
                        debt_reduction_pct = (prior - recent) / prior * 100
                        debt_reducing = debt_reduction_pct >= (PCFG.debt_reduction_threshold * 100)
                        prior_de = prior / total_equity_val if total_equity_val > 0 else None
    except Exception:
        pass

    # Score
    if is_net_cash:
        debt_score = 1.0
    elif is_near_zero_debt:
        debt_score = 0.8
    elif debt_reducing:
        debt_score = min(debt_reduction_pct / 60, 0.7)
    else:
        debt_score = max(1.0 - current_de / 2.0, 0.0)

    flags = []
    if is_net_cash:
        flags.append("🟢 NET CASH (Cash > Debt)")
    elif is_near_zero_debt:
        flags.append(f"🟢 NEAR ZERO DEBT (D/E {current_de:.2f})")
    elif debt_reducing:
        flags.append(f"📉 Debt Reducing ({debt_reduction_pct:.0f}% in 4 qtrs)")

    return {
        "current_de": round(current_de, 2),
        "net_debt_cr": round(net_debt_cr, 0),
        "is_net_cash": is_net_cash,
        "is_near_zero_debt": is_near_zero_debt,
        "debt_reducing": debt_reducing,
        "debt_reduction_pct": round(debt_reduction_pct, 1),
        "debt_score": round(debt_score, 3),
        "flags": flags,
    }


# ──────────────────────────────────────────────────────
# NEW DETECTOR 3: DEEP VALUE vs SECTOR PEERS (P/E Discount)
# Source: @valuepick "Guess The Gem" Twitter posts — he ALWAYS mentioned
#         "P/E at ¼ of industry average" as a key clue. This is the single
#         most consistent valuation signal across his 15-year history.
# ──────────────────────────────────────────────────────

def analyze_peer_pe_discount(info: dict) -> dict:
    """
    Detect stocks trading at deep discount to their sector P/E.
    VALUEPICK's "Guess The Gem" clues repeatedly flagged this:
    "This company trades at P/E of X, while sector average is 4X"

    When a quality company in a good sector trades at ¼ of sector P/E,
    one of two things is true: there's a problem, or it's undiscovered.
    His skill was distinguishing the two. Our job is to surface it.
    """
    trailing_pe = info.get("trailingPE")
    forward_pe  = info.get("forwardPE")
    sector      = info.get("sector") or "Default"
    industry    = info.get("industry") or ""

    # Map to our sector P/E table
    sector_pe = PCFG.sector_pe_map.get("Default", 25.0)
    for key in PCFG.sector_pe_map:
        if key.lower() in sector.lower() or key.lower() in industry.lower():
            sector_pe = PCFG.sector_pe_map[key]
            break

    best_pe = None
    pe_source = None
    if trailing_pe and 0 < trailing_pe < 200:
        best_pe = trailing_pe
        pe_source = "TTM"
    elif forward_pe and 0 < forward_pe < 200:
        best_pe = forward_pe
        pe_source = "Forward"

    if best_pe is None:
        return {
            "stock_pe": None,
            "sector_pe": sector_pe,
            "pe_discount_pct": None,
            "pe_score": 0.3,   # Neutral if no P/E (pre-profit company)
            "is_deep_value": False,
            "flag": None,
            "sector_mapped": sector,
        }

    pe_ratio_to_sector = best_pe / sector_pe
    pe_discount_pct = (1 - pe_ratio_to_sector) * 100

    # Score: best if trading at <50% of sector P/E (the "Guess The Gem" zone)
    pe_score = max(1.0 - pe_ratio_to_sector, 0.0)
    is_deep_value = pe_ratio_to_sector <= PCFG.peer_pe_discount_threshold

    flag = None
    if pe_ratio_to_sector <= 0.25:
        flag = f"💎 EXTREME VALUE: P/E {best_pe:.1f} vs sector {sector_pe:.0f} (¼ of peers!)"
    elif is_deep_value:
        flag = f"💰 DEEP VALUE: P/E {best_pe:.1f} vs sector {sector_pe:.0f} ({pe_discount_pct:.0f}% discount)"

    return {
        "stock_pe": round(best_pe, 1),
        "sector_pe": sector_pe,
        "pe_ratio_to_sector": round(pe_ratio_to_sector, 2),
        "pe_discount_pct": round(pe_discount_pct, 1),
        "pe_score": round(pe_score, 3),
        "is_deep_value": is_deep_value,
        "flag": flag,
        "pe_source": pe_source,
        "sector_mapped": sector,
    }


# ──────────────────────────────────────────────────────
# COMPOSITE PRE-BREAKOUT SCORE (updated for 3 new signals)
# ──────────────────────────────────────────────────────

def pre_breakout_composite(
    shareholding: dict,
    valuation: dict,
    growth: dict,
    catalyst: dict,
    group: dict,
    debt: dict,
    pe_val: dict,
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
    # Undiscovered bonus (small cap inefficiency — low institutional footprint)
    score += shareholding["undiscovered_score"] * 0.05

    # Hard bonus: trusted group + deep value + catalyst all firing = rare gem signal
    if group["is_trusted_group"] and pe_val["is_deep_value"] and catalyst["catalyst_score"] >= 0.3:
        score += 0.08  # The "Guess The Gem" trifecta

    return round(min(score, 1.0), 4)


# ──────────────────────────────────────────────────────
# PRICE STAGE CLASSIFIER
# ──────────────────────────────────────────────────────

def classify_price_stage(info: dict, hist: pd.DataFrame) -> str:
    """
    Classify where the stock is in its cycle — ideally we want
    'BASE' or 'EARLY UPTREND' not 'EXTENDED'.
    """
    if hist is None or hist.empty:
        return "UNKNOWN"

    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    if price == 0:
        return "UNKNOWN"

    c = hist["Close"]
    high_52w = c.rolling(252).max().iloc[-1]
    low_52w = c.rolling(252).min().iloc[-1]

    pct_from_low = (price - low_52w) / low_52w * 100 if low_52w > 0 else 0
    pct_from_high = (high_52w - price) / high_52w * 100 if high_52w > 0 else 0

    if pct_from_low < 20:
        return "🟢 DEEP BASE (near 52W low)"
    elif pct_from_low < 50 and pct_from_high > 30:
        return "🟡 EARLY UPTREND"
    elif pct_from_high < 10:
        return "🔴 NEAR 52W HIGH (late)"
    else:
        return "🟠 MID CYCLE"


# ──────────────────────────────────────────────────────
# MAIN SCANNER
# ──────────────────────────────────────────────────────

def run_pre_breakout_scanner(symbols: list[str]) -> list[dict]:
    """
    Full pre-breakout scan pipeline.
    Returns ranked list of early-stage candidates.
    """
    log.info("══════════════════════════════════════════════════════")
    log.info("  PRE-BREAKOUT SCANNER (Layer 2) — EARLY STAGE HUNT")
    log.info("  Method: value-picks.blogspot.com + @valuepick framework")
    log.info("  NEW: Trusted Group + Debt Elimination + Peer P/E Discount")
    log.info("══════════════════════════════════════════════════════")
    log.info(f"Scanning {len(symbols)} symbols for pre-breakout signals...")

    results = []

    for i, symbol in enumerate(symbols):
        sym_clean = symbol.replace(".NS", "")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Quick universe filters
            mktcap = (info.get("marketCap") or 0) / 1e7   # ₹ Cr
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

            if not (PCFG.min_market_cap_cr <= mktcap <= PCFG.max_market_cap_cr):
                continue
            if not (PCFG.min_price <= price <= PCFG.max_price):
                continue

            # Get history for stage classification
            hist = ticker.history(period="2y")

            # ── Run all signal analyzers ──
            shareholding = analyze_shareholding(info)
            valuation    = analyze_asset_value(info)
            growth       = analyze_growth(ticker, info)
            group        = analyze_promoter_group(info)        # NEW
            debt         = analyze_debt_trajectory(ticker, info)  # NEW
            pe_val       = analyze_peer_pe_discount(info)      # NEW

            # BSE announcements
            bse_code = get_bse_code_from_symbol(symbol)
            announcements = []
            if bse_code:
                announcements = fetch_bse_announcements(
                    bse_code, PCFG.announcement_lookback_days
                )
                time.sleep(0.3)

            catalyst = detect_catalysts(symbol, announcements)
            stage    = classify_price_stage(info, hist)

            # Must have at least one meaningful signal to qualify
            has_signal = (
                shareholding["insider_pct"] >= PCFG.min_promoter_holding_pct or
                valuation["trading_near_book"] or
                growth["growth_accelerating"] or
                catalyst["catalyst_score"] >= 0.2 or
                group["is_trusted_group"] or           # NEW
                debt["is_near_zero_debt"] or           # NEW
                debt["debt_reducing"] or               # NEW
                pe_val["is_deep_value"]                # NEW
            )
            if not has_signal:
                continue

            composite = pre_breakout_composite(
                shareholding, valuation, growth, catalyst, group, debt, pe_val
            )

            # Collect all active flags for report
            all_flags = []
            if group["flag"]:
                all_flags.append(group["flag"])
            all_flags.extend(debt["flags"])
            if pe_val["flag"]:
                all_flags.append(pe_val["flag"])
            all_flags.extend(catalyst["catalysts"])

            results.append({
                "symbol":           symbol,
                "price":            round(price, 2),
                "market_cap_cr":    round(mktcap, 0),
                "price_stage":      stage,
                "composite_score":  composite,
                "shareholding":     shareholding,
                "valuation":        valuation,
                "growth":           growth,
                "catalyst":         catalyst,
                "group":            group,
                "debt":             debt,
                "pe_value":         pe_val,
                "all_flags":        all_flags,
                "bse_code":         bse_code,
                "scanned_at":       datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            })

            # Log with new signal indicators
            group_tag  = f"[{group['matched_group']}]" if group["is_trusted_group"] else ""
            debt_tag   = "DEBT↓" if debt["debt_reducing"] or debt["is_near_zero_debt"] else ""
            pe_tag     = f"PE@{pe_val['pe_ratio_to_sector']:.1f}x" if pe_val["stock_pe"] else ""
            log.info(
                f"  [{i+1:04d}] {sym_clean:<14} ₹{price:<7.0f} MCap ₹{mktcap:.0f}Cr "
                f"Score:{composite:.3f} {group_tag} {debt_tag} {pe_tag} {stage}"
            )

        except Exception as e:
            log.debug(f"  {sym_clean}: {e}")

        time.sleep(0.4)

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    log.info(f"Pre-breakout scan complete. Candidates: {len(results)}")
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
        "  Signal framework: value-picks.blogspot.com + @valuepick (30yr veteran)",
        "  Signals: Promoter Group | Debt Elimination | Peer P/E | Catalyst",
        sep,
        "  These stocks have NOT broken out yet.",
        "  Monitor for Layer 1 price+volume confirmation before entry.",
        sep,
        "",
    ]

    for i, s in enumerate(candidates, 1):
        sh   = s["shareholding"]
        val  = s["valuation"]
        gr   = s["growth"]
        cat  = s["catalyst"]
        grp  = s.get("group", {})
        dbt  = s.get("debt", {})
        pev  = s.get("pe_value", {})
        flags = s.get("all_flags", cat.get("catalysts", []))

        lines += [
            f"#{i:02d}  {s['symbol']:<18}  Score: {s['composite_score']:.3f}",
            f"    Price: ₹{s['price']:<8}  MCap: ₹{s['market_cap_cr']:.0f} Cr",
            f"    Stage: {s['price_stage']}",
            "",
            f"    ── OWNERSHIP ──",
            f"    Promoter/Insider: {sh['insider_pct']:.1f}%   "
            f"Institutions: {sh['institutional_pct']:.1f}%   "
            f"{'[UNDISCOVERED ✓]' if sh['undiscovered_score'] > 0.6 else ''}",
            f"    Group: {grp.get('matched_group') or 'Unknown'}  "
            f"{'🏛️ TRUSTED GROUP' if grp.get('is_trusted_group') else ''}",
            "",
            f"    ── VALUATION ──",
            f"    P/B: {val['pb_ratio']:.2f}   "
            f"Net Cash: ₹{val['net_cash_cr']:.0f} Cr   "
            f"Cash/MCap: {val['cash_to_mcap_pct']:.1f}%   "
            f"{'[NEAR BOOK ✓]' if val['trading_near_book'] else ''}",
            f"    P/E (stock): {pev.get('stock_pe') or 'N/A'}   "
            f"Sector P/E: {pev.get('sector_pe', '?')}   "
            f"Ratio: {pev.get('pe_ratio_to_sector') or 'N/A'}x   "
            f"{'💎 DEEP VALUE' if pev.get('is_deep_value') else ''}",
            "",
            f"    ── DEBT ──",
            f"    D/E: {dbt.get('current_de', '?')}   "
            f"Net Debt: ₹{dbt.get('net_debt_cr', '?')} Cr   "
            f"Reducing: {'✓ ' + str(round(dbt.get('debt_reduction_pct',0),0)) + '%' if dbt.get('debt_reducing') else '✗'}   "
            f"{'🟢 NET CASH' if dbt.get('is_net_cash') else ''}",
            "",
            f"    ── GROWTH ──",
            f"    Rev: {gr['revenue_growth_pct']:.1f}%   "
            f"Earnings: {gr['earnings_growth_pct']:.1f}%   "
            f"Margin: {gr['profit_margin_pct']:.1f}%   "
            f"Accelerating: {'✓' if gr['growth_accelerating'] else '✗'}",
            "",
        ]

        if flags:
            lines.append(f"    ── SIGNALS ({len(flags)}) ──")
            for flag in flags[:6]:
                lines.append(f"    ▶  {flag}")
        else:
            lines.append("    ── No specific catalyst in last 90 days ──")

        lines.append("")
        lines.append("─" * 72)
        lines.append("")

    lines += [
        sep,
        "  THE VALUEPICK CHECKLIST (apply manually after scan):",
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
