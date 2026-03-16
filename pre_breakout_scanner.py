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

from screener_fetcher import enrich_with_screener

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

    # ─────────────────────────────────────────────────────────
    # SCORING WEIGHTS — v3 rebalance
    #
    # Problem diagnosed from real runs:
    #   • Promoter holding (25%) fires on almost every Indian small cap → not differentiated
    #   • Valuation/P/B (20%) fires on everything near 52W low → not differentiated
    #   • Catalyst (15%) is the rarest and most predictive signal — severely underweighted
    #   • Group (10%) fires for a small minority — correctly sized
    #
    # Fix: raise catalyst to 30%, cut promoter to 15%, cut valuation to 12%.
    # Catalyst is now the dominant signal, not a tiebreaker.
    # This mirrors VALUEPICK's actual method: catalyst-first, valuation-second.
    # ─────────────────────────────────────────────────────────
    w_catalyst:  float = 0.30   # raised from 0.15 — now the dominant signal
    w_growth:    float = 0.22   # raised from 0.20 — earnings quality matters
    w_group:     float = 0.15   # raised from 0.10 — group pedigree is a real filter
    w_promoter:  float = 0.15   # cut from 0.25 — ubiquitous, not differentiated
    w_valuation: float = 0.12   # cut from 0.20 — P/B fires on everything at 52W low
    w_debt:      float = 0.03   # unchanged in spirit
    w_pe_value:  float = 0.03   # unchanged in spirit

    # ─────────────────────────────────────────────────────────
    # TIERED QUALIFICATION GATE — replaces the old flat has_signal check
    #
    # TIER 1 (Always qualify): Hard catalyst events — these are rare and
    #   high-conviction by definition. Any stock with one qualifies immediately.
    #
    # TIER 2 (Need 2+ signals): Soft signals that are common alone but meaningful
    #   in combination. Avoids the "cheap but static" trap.
    #
    # TIER 3 (Hard minimum quality): Even if tiers 1/2 pass, reject stocks
    #   with clearly deteriorating revenue (3 consecutive quarterly declines).
    # ─────────────────────────────────────────────────────────

    # Tier 1 catalyst threshold — score ≥ this = auto-qualify
    tier1_catalyst_threshold: float = 0.30   # preferential allotment, capex, JV

    # Tier 2 — need at least this many soft signals
    tier2_min_signals: int = 2

    # Hard minimum ROCE from Screener (applied post-enrichment re-rank)
    min_roce_pct: float = 10.0   # below this = likely value trap

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


# ──────────────────────────────────────────────────────
# NEW: FREE CASH FLOW YIELD
# Source: 2025 academic study — strongest predictor of multibagger outperformance
# among 464 confirmed multibagger stocks (stronger than P/E, P/B, earnings growth)
# ──────────────────────────────────────────────────────

def analyze_fcf(info: dict) -> dict:
    """
    FCF Yield = Free Cash Flow / Market Cap.
    High FCF yield on a small cap = the business generates real cash,
    not accounting profits. This is what separates EKI (real carbon credits
    generating real USD cash) from earnings-manipulated stories.
    """
    free_cf  = info.get("freeCashflow")
    mktcap   = info.get("marketCap") or 0
    revenue  = info.get("totalRevenue") or 0
    oper_cf  = info.get("operatingCashflow")

    fcf_yield  = None
    ocf_margin = None

    if free_cf and mktcap > 0:
        fcf_yield = round((free_cf / mktcap) * 100, 2)

    if oper_cf and revenue > 0:
        ocf_margin = round((oper_cf / revenue) * 100, 1)

    # Score: best if FCF yield > 5%, excellent if > 10%
    if fcf_yield is None:
        fcf_score = 0.3   # neutral — data not available
    elif fcf_yield <= 0:
        fcf_score = 0.0   # negative FCF = cash burning
    else:
        fcf_score = min(fcf_yield / 10, 1.0)

    flag = None
    if fcf_yield and fcf_yield > 8:
        flag = f"💵 HIGH FCF YIELD: {fcf_yield}% (strong cash generation)"
    elif fcf_yield and fcf_yield > 4:
        flag = f"💵 FCF Yield: {fcf_yield}%"

    return {
        "fcf_yield_pct":  fcf_yield,
        "ocf_margin_pct": ocf_margin,
        "fcf_score":      round(fcf_score, 3),
        "flag":           flag,
    }


# ──────────────────────────────────────────────────────
# NEW: PIOTROSKI F-SCORE
# 9-point financial health check. Score 8-9 = strong improving financials.
# Computed entirely from yfinance data. Proven quality filter for Indian small caps.
# Source: Piotroski (2000) — "Value Investing: The Use of Historical Financial Statements"
# ──────────────────────────────────────────────────────

def calc_piotroski(ticker: yf.Ticker, info: dict) -> dict:
    """
    9 binary signals (0 or 1 each), max score = 9.

    Profitability (4 points):
      F1. ROA positive this year
      F2. Operating cash flow positive
      F3. ROA improved year-over-year
      F4. Accruals: OCF/Assets > ROA (cash earnings > accounting earnings)

    Leverage / Liquidity (3 points):
      F5. Long-term debt ratio decreased YoY
      F6. Current ratio improved YoY
      F7. No new shares issued (dilution)

    Operating Efficiency (2 points):
      F8. Gross margin improved YoY
      F9. Asset turnover improved YoY (revenue / total assets)
    """
    score  = 0
    points = {}

    try:
        bs  = ticker.balance_sheet
        fin = ticker.financials
        cf  = ticker.cashflow

        if bs is None or fin is None or cf is None:
            return {"piotroski_score": None, "piotroski_details": {}, "piotroski_flag": None}
        if bs.empty or fin.empty or cf.empty:
            return {"piotroski_score": None, "piotroski_details": {}, "piotroski_flag": None}

        def row(df, *keys):
            for k in keys:
                for idx in df.index:
                    if k.lower() in str(idx).lower():
                        row_data = df.loc[idx]
                        vals = [float(v) if not pd.isna(v) else None for v in row_data.iloc[:2]]
                        return vals[0], vals[1] if len(vals) > 1 else None
            return None, None

        # Balance sheet
        total_assets_now, total_assets_prior  = row(bs, "total assets")
        lt_debt_now,      lt_debt_prior       = row(bs, "long term debt", "longterm debt")
        curr_assets_now,  curr_assets_prior   = row(bs, "current assets", "total current assets")
        curr_liab_now,    curr_liab_prior      = row(bs, "current liabilities", "total current liabilities")
        shares_now,       shares_prior         = row(bs, "ordinary shares", "common stock", "share issued")

        # Income statement
        net_income_now,   net_income_prior     = row(fin, "net income")
        gross_profit_now, gross_profit_prior   = row(fin, "gross profit")
        revenue_now,      revenue_prior        = row(fin, "total revenue", "revenue")

        # Cash flow
        ocf_now, _ = row(cf, "operating cash flow", "total cash from operating")

        # Compute ratios
        roa_now   = net_income_now   / total_assets_now   if (net_income_now and total_assets_now) else None
        roa_prior = net_income_prior / total_assets_prior if (net_income_prior and total_assets_prior) else None
        ocf_roa   = ocf_now / total_assets_now            if (ocf_now and total_assets_now) else None
        gm_now    = gross_profit_now / revenue_now        if (gross_profit_now and revenue_now) else None
        gm_prior  = gross_profit_prior / revenue_prior    if (gross_profit_prior and revenue_prior) else None
        at_now    = revenue_now / total_assets_now        if (revenue_now and total_assets_now) else None
        at_prior  = revenue_prior / total_assets_prior    if (revenue_prior and total_assets_prior) else None
        lev_now   = lt_debt_now / total_assets_now        if (lt_debt_now and total_assets_now) else 0
        lev_prior = lt_debt_prior / total_assets_prior    if (lt_debt_prior and total_assets_prior) else 0
        cr_now    = curr_assets_now / curr_liab_now       if (curr_assets_now and curr_liab_now) else None
        cr_prior  = curr_assets_prior / curr_liab_prior   if (curr_assets_prior and curr_liab_prior) else None

        # F1: ROA > 0
        f1 = 1 if (roa_now and roa_now > 0) else 0
        # F2: OCF > 0
        f2 = 1 if (ocf_now and ocf_now > 0) else 0
        # F3: ROA improving
        f3 = 1 if (roa_now and roa_prior and roa_now > roa_prior) else 0
        # F4: Cash earnings > accounting earnings (OCF/Assets > ROA)
        f4 = 1 if (ocf_roa and roa_now and ocf_roa > roa_now) else 0
        # F5: Leverage reduced
        f5 = 1 if (lev_now < lev_prior) else 0
        # F6: Current ratio improved
        f6 = 1 if (cr_now and cr_prior and cr_now > cr_prior) else 0
        # F7: No dilution (shares not increased)
        f7 = 1 if (shares_now and shares_prior and shares_now <= shares_prior * 1.01) else 0
        # F8: Gross margin improved
        f8 = 1 if (gm_now and gm_prior and gm_now > gm_prior) else 0
        # F9: Asset turnover improved
        f9 = 1 if (at_now and at_prior and at_now > at_prior) else 0

        score   = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        points  = {
            "F1_roa_positive":     f1, "F2_ocf_positive":      f2,
            "F3_roa_improving":    f3, "F4_cash_gt_accrual":   f4,
            "F5_debt_reduced":     f5, "F6_curr_ratio_up":     f6,
            "F7_no_dilution":      f7, "F8_gross_margin_up":   f8,
            "F9_asset_turnover_up": f9,
        }

    except Exception as e:
        log.debug(f"Piotroski calc failed: {e}")
        return {"piotroski_score": None, "piotroski_details": {}, "piotroski_flag": None}

    flag = None
    if score >= 8:
        flag = f"🏆 PIOTROSKI {score}/9 — Exceptional financial health"
    elif score >= 6:
        flag = f"✅ Piotroski {score}/9 — Improving financials"

    return {
        "piotroski_score":   score,
        "piotroski_details": points,
        "piotroski_flag":    flag,
    }


# ──────────────────────────────────────────────────────
# NEW: P/E EXPANSION TRAJECTORY
# Is P/E expanding quarter-over-quarter? Expansion = market re-rating begun.
# Source: Research — "multibaggers need both earnings growth AND P/E expansion"
# ──────────────────────────────────────────────────────

def analyze_pe_trajectory(ticker: yf.Ticker, info: dict) -> dict:
    """
    Detect if P/E has been expanding over recent quarters.
    A stock going from P/E 5 → 7 → 10 means the market is paying
    more per unit of earnings — the re-rating has already started.

    Fix: ticker.quarterly_earnings is deprecated in yfinance ≥ 0.2.x.
    Now uses ticker.quarterly_income_stmt and extracts Net Income directly,
    then divides by shares outstanding to compute quarterly EPS.
    """
    try:
        current_pe = info.get("trailingPE")
        if not current_pe or current_pe <= 0 or current_pe > 200:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        shares = info.get("sharesOutstanding") or 0
        if shares <= 0:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        # ── Use quarterly_income_stmt (replaces deprecated quarterly_earnings) ──
        qis = ticker.quarterly_income_stmt
        if qis is None or qis.empty:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        # Find the Net Income row — yfinance uses various label spellings
        net_income_row = None
        for idx in qis.index:
            if "net income" in str(idx).lower():
                net_income_row = qis.loc[idx]
                break

        if net_income_row is None or len(net_income_row) < 4:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        # quarterly_income_stmt columns are datetime — most recent first
        # Compute quarterly EPS = net_income / shares
        quarterly_eps = []
        for val in net_income_row.iloc[:6]:   # up to 6 quarters back
            try:
                eps = float(val) / shares if not pd.isna(val) else None
                quarterly_eps.append(eps)
            except Exception:
                quarterly_eps.append(None)

        if len([v for v in quarterly_eps if v is not None]) < 4:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        # Get quarter-end closing prices (3-month interval, last 1.5 years)
        hist = ticker.history(period="18mo", interval="3mo")
        if hist is None or hist.empty or len(hist) < 3:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        prices = hist["Close"].dropna()

        # Build rolling 4-quarter trailing EPS and pair with quarter-end price
        pe_history = []
        for i in range(min(3, len(prices) - 1)):
            eps_window = [v for v in quarterly_eps[i:i+4] if v is not None]
            if len(eps_window) < 4:
                continue
            trailing_eps = sum(eps_window)
            if trailing_eps <= 0:
                continue
            price_at_period = float(prices.iloc[-(i + 1)])
            pe_history.append(round(price_at_period / trailing_eps, 1))

        pe_history.reverse()   # oldest → newest

        if len(pe_history) < 2:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        pe_expanding  = all(pe_history[j] < pe_history[j+1] for j in range(len(pe_history)-1))
        pe_trajectory = " → ".join(str(p) for p in pe_history) + f" → {round(current_pe, 1)} (now)"

        flag = None
        if pe_expanding:
            flag = f"📈 P/E EXPANDING: {pe_trajectory}"

        return {
            "pe_expanding":   pe_expanding,
            "pe_trajectory":  pe_trajectory,
            "pe_traj_flag":   flag,
        }

    except Exception as e:
        log.debug(f"P/E trajectory failed: {e}")
        return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}


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
    fcf: dict, piotroski: dict, pe_traj: dict,
) -> float:
    # Base score from original signals
    score = (
        shareholding["promoter_score"] * PCFG.w_promoter  +
        valuation["valuation_score"]   * PCFG.w_valuation +
        growth["growth_score"]         * PCFG.w_growth    +
        catalyst["catalyst_score"]     * PCFG.w_catalyst  +
        group["group_score"]           * PCFG.w_group     +
        debt["debt_score"]             * PCFG.w_debt      +
        pe_val["pe_score"]             * PCFG.w_pe_value
    )
    # Undiscovered bonus
    score += shareholding["undiscovered_score"] * 0.05

    # NEW: FCF yield bonus (strongest academic predictor)
    score += fcf["fcf_score"] * 0.06

    # NEW: Piotroski bonus — strong improving financials
    if piotroski["piotroski_score"] is not None:
        piotroski_norm = piotroski["piotroski_score"] / 9
        score += piotroski_norm * 0.05

    # NEW: P/E expansion bonus — re-rating already started
    if pe_traj["pe_expanding"]:
        score += 0.04

    # "Guess The Gem" trifecta: trusted group + deep value + catalyst
    if group["is_trusted_group"] and pe_val["is_deep_value"] and catalyst["catalyst_score"] >= 0.3:
        score += 0.08

    return round(min(score, 1.0), 4)


# ──────────────────────────────────────────────────────
# MAIN SCANNER
# ──────────────────────────────────────────────────────

def run_pre_breakout_scanner(symbols: list[str]) -> list[dict]:
    log.info("══════════════════════════════════════════════════════")
    log.info("  PRE-BREAKOUT SCANNER (Layer 2) v3")
    log.info("  Method: value-picks.blogspot.com + @valuepick")
    log.info("  v3: Catalyst-first sort | Tiered gate | ROCE filter")
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
            fcf          = analyze_fcf(info)                        # NEW
            piotroski    = calc_piotroski(ticker, info)             # NEW
            pe_traj      = analyze_pe_trajectory(ticker, info)      # NEW

            bse_code      = get_bse_code_from_symbol(symbol)
            announcements = []
            if bse_code:
                announcements = fetch_bse_announcements(bse_code, PCFG.announcement_lookback_days)
                time.sleep(0.3)

            catalyst = detect_catalysts(symbol, announcements)
            stage    = classify_price_stage(info, hist)

            # ── TIERED QUALIFICATION GATE ──────────────────────────────
            # Tier 1: Hard catalyst = auto-qualify (rare, high-conviction)
            #   Preferential allotment, capex announcement, JV pivot etc.
            tier1 = catalyst["catalyst_score"] >= PCFG.tier1_catalyst_threshold

            # Tier 2: Need at least 2 soft signals firing together
            #   Any single signal alone (high promoter, low P/B, near zero debt)
            #   is too common to be meaningful. Two together is interesting.
            soft_signals = [
                shareholding["insider_pct"] >= PCFG.min_promoter_holding_pct,
                valuation["trading_near_book"],
                growth["growth_accelerating"],
                group["is_trusted_group"],
                debt["is_near_zero_debt"] or debt["debt_reducing"],
                pe_val["is_deep_value"],
                piotroski.get("piotroski_score", 0) is not None and
                    (piotroski.get("piotroski_score") or 0) >= 6,
                fcf.get("fcf_score", 0) >= 0.4,
            ]
            tier2 = sum(1 for s in soft_signals if s) >= PCFG.tier2_min_signals

            if not (tier1 or tier2):
                continue

            # Log why it qualified
            qual_reason = "CATALYST" if tier1 else f"{sum(soft_signals)} soft signals"

            composite = pre_breakout_composite(
                shareholding, valuation, growth, catalyst, group, debt, pe_val,
                fcf, piotroski, pe_traj,
            )

            all_flags = []
            if group["flag"]:           all_flags.append(group["flag"])
            all_flags.extend(debt["flags"])
            if pe_val["flag"]:          all_flags.append(pe_val["flag"])
            if fcf["flag"]:             all_flags.append(fcf["flag"])          # NEW
            if piotroski["piotroski_flag"]: all_flags.append(piotroski["piotroski_flag"])  # NEW
            if pe_traj["pe_traj_flag"]: all_flags.append(pe_traj["pe_traj_flag"])  # NEW
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
                "fcf":             fcf,           # NEW
                "piotroski":       piotroski,     # NEW
                "pe_trajectory":   pe_traj,       # NEW
                "all_flags":       all_flags,
                "bse_code":        bse_code,
                "scanned_at":      datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            })

            group_tag  = f"[{group['matched_group']}]" if group["is_trusted_group"] else ""
            debt_tag   = "DEBT↓" if (debt["debt_reducing"] or debt["is_near_zero_debt"]) else ""
            pe_tag     = f"PE@{pe_val['pe_ratio_to_sector']:.1f}x" if pe_val["stock_pe"] else ""
            cat_tag    = f"CAT:{catalyst['catalyst_score']:.2f}" if catalyst["catalyst_score"] > 0 else ""
            log.info(
                f"  [{i+1:04d}] {sym_clean:<14} ₹{price:<7.0f} "
                f"MCap ₹{mktcap:.0f}Cr Score:{composite:.3f} "
                f"({qual_reason}) {group_tag} {cat_tag} {debt_tag} {pe_tag}"
            )

        except Exception as e:
            log.debug(f"  {sym_clean}: {e}")

        time.sleep(0.4)

    # ── Sort: catalyst-first, composite-second ──────────────────
    # A stock with a confirmed catalyst (preferential allotment, capex)
    # ranks above a stock with a slightly higher composite score but no catalyst.
    # This directly mirrors VALUEPICK's method: catalyst drives the pick,
    # valuation confirms it.
    results.sort(
        key=lambda x: (
            -x["catalyst"]["catalyst_score"],   # primary: highest catalyst first
            -x["composite_score"],               # secondary: highest composite
        )
    )
    top_results = results[:PCFG.top_n]

    log.info(
        f"Pre-breakout scan complete. "
        f"Candidates: {len(results)}  |  Excluded (holding/passive): {excluded_ct}  |  "
        f"With catalyst: {sum(1 for r in results if r['catalyst']['catalyst_score'] >= PCFG.tier1_catalyst_threshold)}"
    )

    # ── Enrich top candidates with Screener.in data ──────────────
    log.info(f"Enriching top {len(top_results)} candidates with Screener.in...")
    top_results = enrich_with_screener(top_results, max_candidates=PCFG.top_n)

    # ── Post-Screener ROCE filter + re-rank ──────────────────────
    # Now that we have accurate ROCE from Screener filings, drop obvious
    # value traps (low ROCE = capital being destroyed, not compounded).
    # Only apply if Screener data was successfully fetched.
    screener_filtered = []
    roce_rejected     = 0
    for c in top_results:
        sc   = c.get("screener") or {}
        roce = sc.get("roce_pct")
        if roce is not None and roce < PCFG.min_roce_pct:
            roce_rejected += 1
            log.info(f"  ROCE filter: dropping {c['symbol']} (ROCE {roce:.1f}% < {PCFG.min_roce_pct}%)")
            continue
        screener_filtered.append(c)

    if roce_rejected > 0:
        log.info(f"  ROCE filter removed {roce_rejected} value traps from final watchlist")

    # Final re-sort after ROCE filter (same order: catalyst-first)
    screener_filtered.sort(
        key=lambda x: (
            -x["catalyst"]["catalyst_score"],
            -x["composite_score"],
        )
    )
    return screener_filtered


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
        "  v2: Holding companies & passive income businesses excluded",
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
        sc    = s.get("screener")   # Screener.in enrichment (may be None)

        # Use Screener data where available, fall back to yfinance
        roce_str     = f"{sc['roce_pct']:.1f}%" if sc and sc.get("roce_pct") else "N/A"
        cagr_str     = f"{sc['rev_cagr_3yr_pct']:.1f}%" if sc and sc.get("rev_cagr_3yr_pct") else "N/A"
        promoter_str = f"{sc['promoter_pct']:.1f}%" if sc and sc.get("promoter_pct") else f"{sh['insider_pct']:.1f}%"
        de_str       = f"{sc['de_ratio']:.2f}" if sc and sc.get("de_ratio") is not None else f"{dbt.get('current_de', '?')}"
        data_src     = "Screener.in ✓" if sc else "yfinance"

        lines += [
            f"#{i:02d}  {s['symbol']:<18}  Score: {s['composite_score']:.3f}  [{data_src}]",
            f"    Price: ₹{s['price']:<8}  MCap: ₹{s['market_cap_cr']:.0f} Cr",
            f"    Stage: {s['price_stage']}",
            "",
            f"    ── OWNERSHIP ──",
            (f"    Promoter: {promoter_str}   "
             f"Institutions: {sh['institutional_pct']:.1f}%   "
             f"{'[UNDISCOVERED]' if sh['undiscovered_score'] > 0.6 else ''}"),
            (f"    Group: {grp.get('matched_group') or 'Unknown'}  "
             f"{'🏛️ TRUSTED GROUP' if grp.get('is_trusted_group') else ''}"),
            "",
            f"    ── VALUATION ──",
            (f"    P/B: {val['pb_ratio']:.2f}   "
             f"P/E: {pev.get('stock_pe') or 'N/A'}   "
             f"Sector P/E: {pev.get('sector_pe', '?')}   "
             f"{'💎 DEEP VALUE' if pev.get('is_deep_value') else ''}"),
            "",
            f"    ── FUNDAMENTALS ({data_src}) ──",
            (f"    ROCE: {roce_str}   "
             f"D/E: {de_str}   "
             f"OPM: {gr['profit_margin_pct']:.1f}%"),
            (f"    Rev Growth (YoY): {gr['revenue_growth_pct']:.1f}%   "
             f"3Yr CAGR: {cagr_str}   "
             f"PAT Growth: {gr['earnings_growth_pct']:.1f}%   "
             f"Accel: {'✓' if gr['growth_accelerating'] else '✗'}"),
            "",
            f"    ── DEBT ──",
            (f"    D/E: {de_str}   "
             f"Net Debt: ₹{dbt.get('net_debt_cr', '?')} Cr   "
             f"Reducing: {'✓ ' + str(round(dbt.get('debt_reduction_pct', 0), 0)) + '%' if dbt.get('debt_reducing') else '✗'}   "
             f"{'🟢 NET CASH' if dbt.get('is_net_cash') else ''}"),
            "",
        ]

        if flags:
            lines.append(f"    ── SIGNALS ({len(flags)}) ──")
            for flag in flags[:8]:   # show more signals now
                lines.append(f"    ▶  {flag}")
        else:
            lines.append("    ── No specific catalyst in last 90 days ──")

        # Show Piotroski score + PE trajectory explicitly if notable
        p = s.get("piotroski", {})
        pt = s.get("pe_trajectory", {})
        if p.get("piotroski_score") is not None:
            lines.append(f"    Piotroski F-Score: {p['piotroski_score']}/9   "
                         f"P/E Expanding: {'✓  ' + (pt.get('pe_trajectory') or '') if pt.get('pe_expanding') else '✗'}")

        lines += ["", "─" * 72, ""]

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
