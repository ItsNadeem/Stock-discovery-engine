"""
pre_breakout_scanner.py — Layer 2: Early-Stage Multibagger Detection

Inspired by value-picks.blogspot.com + @valuepick (Twitter) methodology.
30-year Indian market veteran. Verified calls: Paushak ₹74→₹10,000 (140x),
Tasty Bite ₹165→₹9,420, EKI ₹150→₹10,000 (66x in 9 months).

FIX v3 (2026-03) — three bugs causing the whole watchlist to score 0.60–0.63:

FIX A: CATALYST DETECTION WAS SILENTLY RETURNING ZERO FOR EVERY STOCK.
  The BSE API was called with strSearch=P (price-sensitive only).
  Most small-cap announcements are filed as general corporate disclosures,
  not price-sensitive — so the API returned empty lists.
  Catalyst weight is 28% of score. Zero catalyst = every stock scores ~0.60.
  Fix: also call strSearch=C (corporate actions) as a second pass, and
  read the Screener.in concalls list which contains earnings-call events.
  Also added a debug log showing announcement count per stock.

FIX B: HARD PIOTROSKI GATE ADDED (min 5/9 required to enter watchlist).
  Previously Piotroski was soft-scored but not gated.
  A stock with Piotroski 3/9 (UYFINCORE) was ranking above Piotroski 9/9
  (DRCSYSTEMS) because a suspiciously low PEG from bad yfinance data
  inflated the Lynch score. Hard floor eliminates junk quality stocks
  before scoring even starts.

FIX C: ROCE HARD GATE (min 10%) applied after Screener enrichment.
  Low-ROCE stocks may score well on PEG/valuation but are value traps.
  Gate applied in the final ranking step post-enrichment.
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
from public_data_fetcher import fetch_all_public_signals

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────

@dataclass
class PreBreakoutConfig:
    # Universe
    max_market_cap_cr: float = 2_000.0
    min_market_cap_cr: float = 10.0
    max_price: float = 500.0
    min_price: float = 5.0

    # FIX 1: Excluded industries (holding/investment companies)
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

    # FIX 2: Passive income thresholds
    max_operating_margin_pct: float = 85.0
    max_believable_revenue_growth_pct: float = 300.0

    # FIX B: Minimum Piotroski score to enter watchlist at all
    min_piotroski: int = 5   # was unset — stocks with 3/9 were ranking above 9/9
    # FIX C: Minimum ROCE after Screener enrichment
    min_roce_pct: float = 10.0

    # Promoter signals
    min_promoter_holding_pct: float = 35.0
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

    # Trusted promoter groups
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
    near_zero_debt_de: float = 0.15

    # Peer P/E discount
    peer_pe_discount_threshold: float = 0.50
    sector_pe_map: dict = field(default_factory=lambda: {
        "Chemicals": 28.0, "Pharmaceuticals": 32.0, "Consumer Goods": 45.0,
        "Industrial": 22.0, "Technology": 30.0, "Auto Components": 20.0,
        "Metals": 14.0, "Textiles": 18.0, "Packaging": 25.0,
        "Specialty Chemicals": 35.0, "Defence": 40.0, "Logistics": 28.0,
        "Healthcare": 35.0, "Engineering": 24.0, "Agro Chemicals": 22.0,
        "Default": 25.0,
    })

    # FIX A: announcement lookback — also look at corporate actions (strSearch=C)
    announcement_lookback_days: int = 90

    # Scoring weights v3
    w_catalyst:   float = 0.28
    w_growth:     float = 0.20
    w_group:      float = 0.14
    w_promoter:   float = 0.14
    w_valuation:  float = 0.11
    w_public:     float = 0.08
    w_debt:       float = 0.03
    w_pe_value:   float = 0.02

    # Tiered qualification gate
    tier1_catalyst_threshold: float = 0.30
    tier2_min_signals: int = 2

    top_n: int = 15


PCFG = PreBreakoutConfig()

BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
}


# ──────────────────────────────────────────────────────
# UNIVERSE QUALITY GATE (FIX 1 + FIX 2 unchanged)
# ──────────────────────────────────────────────────────

def should_exclude(info: dict) -> tuple[bool, str]:
    industry = (info.get("industry") or "").lower()
    sector   = (info.get("sector") or "").lower()
    summary  = (info.get("longBusinessSummary") or "").lower()
    company_name = (info.get("longName") or "").lower()

    for excl in PCFG.excluded_industries:
        if excl in industry or excl in sector:
            return True, f"Excluded industry: '{excl}'"

    for kw in PCFG.excluded_description_keywords:
        if kw in summary:
            return True, f"Holding company description keyword: '{kw}'"

    for pattern in PCFG.excluded_name_patterns:
        if pattern in company_name:
            return True, f"Holding company name pattern: '{pattern}'"

    oper_margin = (info.get("operatingMargins") or 0) * 100
    if oper_margin > PCFG.max_operating_margin_pct:
        return True, f"Passive income: op margin {oper_margin:.0f}%"

    rev_growth = (info.get("revenueGrowth") or 0) * 100
    if rev_growth > PCFG.max_believable_revenue_growth_pct:
        return True, f"Likely data artefact: revenue growth {rev_growth:.0f}%"

    return False, ""


# ──────────────────────────────────────────────────────
# BSE DATA FETCHERS — FIX A: dual search mode
# ──────────────────────────────────────────────────────

def fetch_bse_announcements(symbol_code: str, days_back: int = 90) -> list[dict]:
    """
    FIX A: Fetch BSE announcements using BOTH strSearch=P (price-sensitive)
    AND strSearch=C (corporate actions).

    Previously only P was used — most small-cap announcements (capex, JV,
    order wins, debt repayment) are filed as corporate actions, not as
    price-sensitive disclosures. This caused catalyst_score=0 for almost
    every stock, killing 28% of the composite score.
    """
    cutoff = datetime.now() - timedelta(days=days_back)
    result = []
    seen_titles = set()

    for search_type in ["P", "C"]:  # FIX A: was only "P"
        try:
            url = (
                f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
                f"?strCat=-1&strPrevDate=&strScrip={symbol_code}"
                f"&strSearch={search_type}&strToDate=&strType=C&subcategory=-1"
            )
            r = requests.get(url, headers=BSE_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            for ann in data.get("Table", []):
                try:
                    dt_str = ann.get("News_submission_dt", "")
                    dt = datetime.strptime(dt_str[:10], "%Y-%m-%d") if dt_str else None
                    if dt and dt >= cutoff:
                        title = ann.get("NEWSSUB", "").strip()
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            result.append({
                                "date":     dt.strftime("%Y-%m-%d"),
                                "title":    title,
                                "category": ann.get("CATEGORYNAME", "").strip(),
                                "search":   search_type,
                            })
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"BSE announcements ({search_type}) failed for {symbol_code}: {e}")

    log.debug(f"BSE {symbol_code}: {len(result)} announcements found (P+C combined)")
    return result


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
            data = r.json()
            items = data if isinstance(data, list) else data.get("Table", [])
            for item in items:
                if (sym_clean.lower() in str(item.get("SCRIP_CD", "")).lower() or
                        sym_clean.lower() in str(item.get("Issuer_Name", "")).lower()):
                    return str(item.get("SCRIP_CD", ""))
    except Exception as e:
        log.debug(f"BSE code lookup failed for {sym_clean}: {e}")
    return None


def fetch_screener_concalls(screener_data: Optional[dict]) -> list[str]:
    """
    FIX A: Extract concall / earnings call events from Screener.in response.
    These are additional catalyst signals not available via BSE API.
    Returns list of concall description strings.
    """
    if not screener_data:
        return []
    concalls = screener_data.get("concalls") or []
    results = []
    cutoff = datetime.now() - timedelta(days=90)
    for cc in concalls:
        try:
            dt_str = cc.get("date") or ""
            dt = datetime.strptime(dt_str[:10], "%Y-%m-%d") if dt_str else None
            if dt and dt >= cutoff:
                results.append(cc.get("title") or cc.get("description") or "Earnings Concall")
        except Exception:
            pass
    return results


# ──────────────────────────────────────────────────────
# SIGNAL ANALYZERS (unchanged from original)
# ──────────────────────────────────────────────────────

def analyze_shareholding(info: dict) -> dict:
    insider_pct = (info.get("heldPercentInsiders") or 0) * 100
    inst_pct    = (info.get("heldPercentInstitutions") or 0) * 100
    promoter_score     = min(insider_pct / 75, 1.0)
    undiscovered_score = max(1.0 - inst_pct / 30, 0.0)
    return {
        "insider_pct":        round(insider_pct, 1),
        "institutional_pct":  round(inst_pct, 1),
        "promoter_score":     round(promoter_score, 3),
        "undiscovered_score": round(undiscovered_score, 3),
    }


def analyze_asset_value(info: dict) -> dict:
    mktcap   = info.get("marketCap") or 0
    book_val = info.get("bookValue") or 0
    cash     = info.get("totalCash") or 0
    debt     = info.get("totalDebt") or 0
    price    = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    mktcap_cr    = mktcap / 1e7
    net_cash_cr  = (cash - debt) / 1e7
    cash_to_mcap = net_cash_cr / mktcap_cr if mktcap_cr > 0 else 0
    pb = price / book_val if book_val > 0 else 99

    pb_score       = max(1.0 - (pb - 0.5) / 2.5, 0.0)
    cash_score     = min(max(cash_to_mcap, 0), 1.0)
    valuation_score = pb_score * 0.6 + cash_score * 0.4

    return {
        "market_cap_cr":    round(mktcap_cr, 0),
        "pb_ratio":         round(pb, 2),
        "net_cash_cr":      round(net_cash_cr, 0),
        "cash_to_mcap_pct": round(cash_to_mcap * 100, 1),
        "valuation_score":  round(valuation_score, 3),
        "trading_near_book": pb <= PCFG.mcap_to_book_max,
    }


def analyze_growth(ticker: yf.Ticker, info: dict) -> dict:
    rev_growth  = min((info.get("revenueGrowth") or 0) * 100, PCFG.max_believable_revenue_growth_pct)
    earn_growth = (info.get("earningsGrowth") or 0) * 100
    profit_mgn  = (info.get("profitMargins") or 0) * 100

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


def analyze_fcf(info: dict) -> dict:
    free_cf = info.get("freeCashflow")
    mktcap  = info.get("marketCap") or 0
    revenue = info.get("totalRevenue") or 0
    oper_cf = info.get("operatingCashflow")

    fcf_yield = None
    ocf_margin = None
    if free_cf and mktcap > 0:
        fcf_yield = round((free_cf / mktcap) * 100, 2)
    if oper_cf and revenue > 0:
        ocf_margin = round((oper_cf / revenue) * 100, 1)

    if fcf_yield is None:
        fcf_score = 0.3
    elif fcf_yield <= 0:
        fcf_score = 0.0
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


def calc_piotroski(ticker: yf.Ticker, info: dict) -> dict:
    score = 0
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

        total_assets_now,  total_assets_prior  = row(bs, "total assets")
        lt_debt_now,       lt_debt_prior        = row(bs, "long term debt", "longterm debt")
        curr_assets_now,   curr_assets_prior    = row(bs, "current assets", "total current assets")
        curr_liab_now,     curr_liab_prior      = row(bs, "current liabilities", "total current liabilities")
        shares_now,        shares_prior         = row(bs, "ordinary shares", "common stock", "share issued")
        net_income_now,    net_income_prior     = row(fin, "net income")
        gross_profit_now,  gross_profit_prior   = row(fin, "gross profit")
        revenue_now,       revenue_prior        = row(fin, "total revenue", "revenue")
        ocf_now,           _                    = row(cf, "operating cash flow", "total cash from operating")

        roa_now  = net_income_now  / total_assets_now  if (net_income_now  and total_assets_now)  else None
        roa_prior= net_income_prior/ total_assets_prior if (net_income_prior and total_assets_prior) else None
        ocf_roa  = ocf_now         / total_assets_now  if (ocf_now         and total_assets_now)  else None
        gm_now   = gross_profit_now / revenue_now      if (gross_profit_now and revenue_now)       else None
        gm_prior = gross_profit_prior/revenue_prior    if (gross_profit_prior and revenue_prior)   else None
        at_now   = revenue_now     / total_assets_now  if (revenue_now     and total_assets_now)  else None
        at_prior = revenue_prior   / total_assets_prior if (revenue_prior  and total_assets_prior) else None
        lev_now  = lt_debt_now     / total_assets_now  if (lt_debt_now     and total_assets_now)  else 0
        lev_prior= lt_debt_prior   / total_assets_prior if (lt_debt_prior  and total_assets_prior) else 0
        cr_now   = curr_assets_now / curr_liab_now     if (curr_assets_now and curr_liab_now)     else None
        cr_prior = curr_assets_prior/curr_liab_prior   if (curr_assets_prior and curr_liab_prior)  else None

        f1 = 1 if (roa_now  and roa_now > 0)                                else 0
        f2 = 1 if (ocf_now  and ocf_now > 0)                                else 0
        f3 = 1 if (roa_now  and roa_prior and roa_now > roa_prior)          else 0
        f4 = 1 if (ocf_roa  and roa_now   and ocf_roa > roa_now)            else 0
        f5 = 1 if (lev_now < lev_prior)                                     else 0
        f6 = 1 if (cr_now   and cr_prior  and cr_now > cr_prior)            else 0
        f7 = 1 if (shares_now and shares_prior and shares_now <= shares_prior * 1.01) else 0
        f8 = 1 if (gm_now   and gm_prior  and gm_now > gm_prior)            else 0
        f9 = 1 if (at_now   and at_prior  and at_now > at_prior)            else 0

        score = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        points = {
            "F1_roa_positive": f1, "F2_ocf_positive": f2,
            "F3_roa_improving": f3, "F4_cash_gt_accrual": f4,
            "F5_debt_reduced": f5, "F6_curr_ratio_up": f6,
            "F7_no_dilution": f7, "F8_gross_margin_up": f8,
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

    return {"piotroski_score": score, "piotroski_details": points, "piotroski_flag": flag}


def analyze_pe_trajectory(ticker: yf.Ticker, info: dict) -> dict:
    try:
        current_pe = info.get("trailingPE")
        if not current_pe or current_pe <= 0 or current_pe > 200:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        shares = info.get("sharesOutstanding") or 0
        if shares <= 0:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        qis = ticker.quarterly_income_stmt
        if qis is None or qis.empty:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        net_income_row = None
        for idx in qis.index:
            if "net income" in str(idx).lower():
                net_income_row = qis.loc[idx]
                break

        if net_income_row is None or len(net_income_row) < 4:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        quarterly_eps = []
        for val in net_income_row.iloc[:6]:
            try:
                eps = float(val) / shares if not pd.isna(val) else None
                quarterly_eps.append(eps)
            except Exception:
                quarterly_eps.append(None)

        if len([v for v in quarterly_eps if v is not None]) < 4:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        hist = ticker.history(period="18mo", interval="3mo")
        if hist is None or hist.empty or len(hist) < 3:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        prices = hist["Close"].dropna()
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

        pe_history.reverse()
        if len(pe_history) < 2:
            return {"pe_expanding": False, "pe_trajectory": None, "pe_traj_flag": None}

        pe_expanding  = all(pe_history[j] < pe_history[j+1] for j in range(len(pe_history)-1))
        pe_trajectory = " → ".join(str(p) for p in pe_history) + f" → {round(current_pe, 1)} (now)"

        flag = f"📈 P/E EXPANDING: {pe_trajectory}" if pe_expanding else None
        return {"pe_expanding": pe_expanding, "pe_trajectory": pe_trajectory, "pe_traj_flag": flag}

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
        "matched_group":   matched_group,
        "group_score":     round(group_score, 3),
        "is_trusted_group":matched_group is not None,
        "flag":            f"🏛️ {matched_group} Group" if matched_group else None,
    }


def analyze_debt_trajectory(ticker: yf.Ticker, info: dict) -> dict:
    current_debt = info.get("totalDebt") or 0
    cash         = info.get("totalCash") or 0
    book_val     = info.get("bookValue") or 0
    shares       = info.get("sharesOutstanding") or 0
    total_equity = book_val * shares if shares > 0 else 0

    current_de   = current_debt / total_equity if total_equity > 0 else 0
    net_debt_cr  = (current_debt - cash) / 1e7
    is_net_cash  = (current_debt - cash) <= 0
    is_near_zero = current_de <= PCFG.near_zero_debt_de

    debt_reducing     = False
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

    if is_net_cash:    debt_score = 1.0
    elif is_near_zero: debt_score = 0.8
    elif debt_reducing: debt_score = min(debt_reduction_pct / 60, 0.7)
    else:              debt_score = max(1.0 - current_de / 2.0, 0.0)

    flags = []
    if is_net_cash:    flags.append("🟢 NET CASH (Cash > Debt)")
    elif is_near_zero: flags.append(f"🟢 NEAR ZERO DEBT (D/E {current_de:.2f})")
    elif debt_reducing: flags.append(f"📉 Debt Reducing ({debt_reduction_pct:.0f}% in 4 qtrs)")

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

    ratio          = best_pe / sector_pe
    pe_discount_pct = (1 - ratio) * 100
    pe_score       = max(1.0 - ratio, 0.0)
    is_deep_value  = ratio <= PCFG.peer_pe_discount_threshold

    flag = None
    if ratio <= 0.25:
        flag = f"💎 EXTREME VALUE: P/E {best_pe:.1f} vs sector {sector_pe:.0f} (1/4 of peers!)"
    elif is_deep_value:
        flag = f"💰 DEEP VALUE: P/E {best_pe:.1f} vs sector {sector_pe:.0f} ({pe_discount_pct:.0f}% discount)"

    return {
        "stock_pe": round(best_pe, 1), "sector_pe": sector_pe,
        "pe_ratio_to_sector": round(ratio, 2), "pe_discount_pct": round(pe_discount_pct, 1),
        "pe_score": round(pe_score, 3), "is_deep_value": is_deep_value,
        "flag": flag, "pe_source": pe_source, "sector_mapped": sector,
    }


def detect_catalysts(symbol: str, announcements: list[dict],
                     screener_data: Optional[dict] = None) -> dict:
    """
    FIX A: Also check Screener.in concalls list for catalyst signals.
    Added debug log showing total announcements checked.
    """
    all_text = " ".join(
        (a.get("title", "") + " " + a.get("category", "")).lower()
        for a in announcements
    )

    # FIX A: also pull concall signals from Screener
    concall_events = fetch_screener_concalls(screener_data)
    if concall_events:
        log.debug(f"  {symbol}: {len(concall_events)} concall events from Screener")

    log.debug(f"  {symbol}: {len(announcements)} BSE announcements checked for catalysts")

    preferential_found = any(kw in all_text for kw in [
        "preferential allotment", "preferential issue", "warrant",
        "promoter acquiring", "promoter purchase",
    ])
    capex_found   = any(kw in all_text for kw in PCFG.capex_keywords)
    pivot_found   = any(kw in all_text for kw in PCFG.pivot_keywords)
    jv_found      = any(kw in all_text for kw in ["joint venture", " jv ", "collaboration", "partnership"])
    buyback_found = any(kw in all_text for kw in ["buyback", "buy back", "share repurchase"])
    export_found  = any(kw in all_text for kw in ["export", "overseas", "international order"])
    order_found   = any(kw in all_text for kw in [
        "order win", "order received", "loi", "contract awarded", "new order"
    ])
    debt_repay_found = any(kw in all_text for kw in [
        "repayment", "debt free", "loan closed", "ncd redemption",
        "debenture redemption", "term loan repaid",
    ])

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
    if debt_repay_found:  # FIX A: new keyword — valuepick explicitly tracks this
        found_catalysts.append("🟢 Debt Repayment / Loan Closure")
        catalyst_score += 0.20
    if concall_events:   # FIX A: concall = management engagement signal
        found_catalysts.append(f"🎙️ Recent Earnings Concall ({len(concall_events)})")
        catalyst_score += 0.10

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
    c = hist["Close"]
    high_52w = c.rolling(252).max().iloc[-1]
    low_52w  = c.rolling(252).min().iloc[-1]
    pct_from_low  = (price - low_52w)  / low_52w  * 100 if low_52w  > 0 else 0
    pct_from_high = (high_52w - price) / high_52w * 100 if high_52w > 0 else 0
    if pct_from_low  < 20:              return "🟢 DEEP BASE (near 52W low)"
    elif pct_from_low < 50 and pct_from_high > 30: return "🟡 EARLY UPTREND"
    elif pct_from_high < 10:            return "🔴 NEAR 52W HIGH (late)"
    else:                               return "🟠 MID CYCLE"


# ──────────────────────────────────────────────────────
# PETER LYNCH / GARP SCORING MODULE (unchanged)
# ──────────────────────────────────────────────────────

def calc_lynch_score(ticker: yf.Ticker, info: dict) -> tuple[float, dict]:
    details = {
        "peg_ratio": None, "inst_own_pct": None, "net_cash_pct": None,
        "eps_consistent": None, "inv_health": None, "lynch_flag": None,
    }

    def g(key, default=None):
        v = info.get(key, default)
        if v is None or (isinstance(v, float) and (v != v)):
            return default
        return v

    lynch_score = 0.0
    flags = []

    # 1. PEG ratio
    peg_score   = 0.3
    trailing_pe = g("trailingPE")
    rev_growth  = g("revenueGrowth")
    forward_pe  = g("forwardPE")

    if trailing_pe and trailing_pe > 0 and trailing_pe < 200:
        if rev_growth and rev_growth > 0.05:
            rev_growth_pct = rev_growth * 100
            peg = trailing_pe / rev_growth_pct
            details["peg_ratio"] = round(peg, 2)
            if peg < 0.5:
                peg_score = 1.0
                flags.append(f"🔥 Lynch PEG {peg:.2f} — Deeply undervalued vs growth")
            elif peg < 1.0:
                peg_score = 0.85
                flags.append(f"✅ Lynch PEG {peg:.2f} — Ideal GARP zone (<1)")
            elif peg < 1.5:
                peg_score = 0.60
                flags.append(f"📊 Lynch PEG {peg:.2f} — Acceptable GARP (<1.5)")
            elif peg < 2.0:
                peg_score = 0.35
            else:
                peg_score = 0.10
        elif trailing_pe < 12:
            peg_score = 0.55
        elif forward_pe and 0 < forward_pe < 20:
            peg_score = 0.65

    lynch_score += peg_score * 0.30

    # 2. Institutional ownership
    inst_own_raw = g("institutionsPercentHeld") or g("heldPercentInstitutions") or None
    inst_score = 0.3
    if inst_own_raw is not None:
        inst_pct = inst_own_raw * 100 if inst_own_raw < 1.0 else inst_own_raw
        details["inst_own_pct"] = round(inst_pct, 1)
        if inst_pct < 5:
            inst_score = 1.0
            flags.append(f"👁️ Undiscovered: {inst_pct:.1f}% institutional holding")
        elif inst_pct < 15:
            inst_score = 0.80
            flags.append(f"🌱 Early stage: {inst_pct:.1f}% institutional (Lynch zone)")
        elif inst_pct < 30:
            inst_score = 0.55
        elif inst_pct < 50:
            inst_score = 0.30
        else:
            inst_score = 0.10
    lynch_score += inst_score * 0.20

    # 3. Net cash / Market cap
    cash   = g("totalCash", 0) or 0
    debt   = g("totalDebt", 0) or 0
    mktcap = g("marketCap", 0) or 0
    net_cash_score = 0.2
    if mktcap > 0:
        net_cash     = cash - debt
        net_cash_pct = net_cash / mktcap * 100
        details["net_cash_pct"] = round(net_cash_pct, 1)
        if net_cash_pct > 30:
            net_cash_score = 1.0
            flags.append(f"💰 Net cash {net_cash_pct:.0f}% of MCap — Lynch floor ✓")
        elif net_cash_pct > 15:
            net_cash_score = 0.75
            flags.append(f"💰 Net cash {net_cash_pct:.0f}% of MCap")
        elif net_cash_pct > 0:
            net_cash_score = 0.50
        else:
            net_cash_score = 0.20
    lynch_score += net_cash_score * 0.20

    # 4. EPS consistency (4+ quarters positive)
    eps_score = 0.3
    try:
        qis = ticker.quarterly_income_stmt
        if qis is not None and not qis.empty:
            ni_row = None
            for idx in qis.index:
                if "net income" in str(idx).lower():
                    ni_row = qis.loc[idx]
                    break
            if ni_row is not None:
                vals = [float(v) for v in ni_row.iloc[:6] if not pd.isna(v)]
                pos_count = sum(1 for v in vals if v > 0)
                details["eps_consistent"] = pos_count
                if pos_count >= 5:
                    eps_score = 1.0
                    flags.append(f"📊 {pos_count}/6 quarters profitable")
                elif pos_count >= 4:
                    eps_score = 0.75
                elif pos_count >= 3:
                    eps_score = 0.50
                else:
                    eps_score = 0.10
    except Exception:
        pass
    lynch_score += eps_score * 0.15

    # 5. Inventory health (sales growing faster than inventory)
    inv_score = 0.5
    try:
        bs  = ticker.balance_sheet
        fin = ticker.financials
        if bs is not None and fin is not None and not bs.empty and not fin.empty:
            inv_now = inv_prior = rev_now = rev_prior = None
            for idx in bs.index:
                if "inventor" in str(idx).lower():
                    row_data = bs.loc[idx]
                    inv_now  = float(row_data.iloc[0]) if not pd.isna(row_data.iloc[0]) else None
                    inv_prior = float(row_data.iloc[1]) if len(row_data) > 1 and not pd.isna(row_data.iloc[1]) else None
                    break
            for idx in fin.index:
                if "revenue" in str(idx).lower() or "sales" in str(idx).lower():
                    row_data  = fin.loc[idx]
                    rev_now   = float(row_data.iloc[0]) if not pd.isna(row_data.iloc[0]) else None
                    rev_prior = float(row_data.iloc[1]) if len(row_data) > 1 and not pd.isna(row_data.iloc[1]) else None
                    break
            if all(v is not None and v > 0 for v in [inv_now, inv_prior, rev_now, rev_prior]):
                inv_growth = (inv_now - inv_prior) / inv_prior
                rev_growth_calc = (rev_now - rev_prior) / rev_prior
                details["inv_health"] = round(inv_growth - rev_growth_calc, 3)
                if inv_growth < rev_growth_calc:
                    inv_score = 0.80
                    flags.append("📦 Healthy: sales growing faster than inventory")
                elif inv_growth > rev_growth_calc * 1.5:
                    inv_score = 0.20
    except Exception:
        pass
    lynch_score += inv_score * 0.15

    # Lynch stock classification
    trailing_pe2 = g("trailingPE")
    roe          = g("returnOnEquity", 0) or 0
    lynch_class  = "Unknown"
    if rev_growth and rev_growth > 0.20 and trailing_pe2 and trailing_pe2 < 30:
        lynch_class = "Fast Grower"
    elif rev_growth and 0.05 < rev_growth <= 0.20:
        lynch_class = "Stalwart"
    elif rev_growth and rev_growth <= 0.05:
        lynch_class = "Slow Grower"
    elif roe < 0:
        lynch_class = "Turnaround"

    details["lynch_class"] = lynch_class
    details["lynch_score"] = round(min(lynch_score, 1.0), 3)
    details["lynch_flags"] = flags
    details["lynch_flag"]  = " | ".join(flags[:3]) if flags else None

    return round(min(lynch_score, 1.0), 3), details


# ──────────────────────────────────────────────────────
# MAIN SCANNER — with FIX B (Piotroski gate)
# ──────────────────────────────────────────────────────

def analyse_pre_breakout(
    symbol: str,
    nse_session=None,
    universe_deals: dict = None,
) -> Optional[dict]:
    """
    Full pre-breakout analysis for a single symbol.

    FIX B: Hard Piotroski gate — stocks with score < PCFG.min_piotroski
    are rejected before scoring. This prevents low-quality stocks with
    a lucky PEG ratio from ranking above high-quality stocks.
    """
    sym_clean = symbol.replace(".NS", "")

    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}
    except Exception as e:
        log.debug(f"{symbol}: yf.Ticker failed — {e}")
        return None

    if not info or not info.get("marketCap"):
        return None

    # ── Universe filters ──
    mktcap_cr = (info.get("marketCap") or 0) / 1e7
    price     = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    if not (PCFG.min_market_cap_cr <= mktcap_cr <= PCFG.max_market_cap_cr):
        return None
    if not (PCFG.min_price <= price <= PCFG.max_price):
        return None

    exclude, reason = should_exclude(info)
    if exclude:
        log.debug(f"  {sym_clean}: excluded — {reason}")
        return None

    # ── Historical data ──
    try:
        hist = ticker.history(period="2y", interval="1d")
    except Exception:
        hist = pd.DataFrame()

    # ── Piotroski (FIX B: gate before expensive analysis) ──
    piotroski_data = calc_piotroski(ticker, info)
    p_score = piotroski_data.get("piotroski_score")

    if p_score is not None and p_score < PCFG.min_piotroski:
        log.debug(
            f"  {sym_clean}: Piotroski {p_score}/9 < {PCFG.min_piotroski} — rejected"
        )
        return None

    # ── All signal modules ──
    shareholding = analyze_shareholding(info)
    valuation    = analyze_asset_value(info)
    growth       = analyze_growth(ticker, info)
    fcf_data     = analyze_fcf(info)
    debt         = analyze_debt_trajectory(ticker, info)
    group        = analyze_promoter_group(info)
    peer_pe      = analyze_peer_pe_discount(info)
    pe_traj      = analyze_pe_trajectory(ticker, info)
    lynch_score, lynch_details = calc_lynch_score(ticker, info)

    # ── BSE announcements (FIX A: both P and C search types) ──
    bse_code     = get_bse_code_from_symbol(symbol)
    announcements = []
    if bse_code:
        announcements = fetch_bse_announcements(bse_code, PCFG.announcement_lookback_days)

    # Pass None for screener_data at this stage — enrichment happens later
    catalyst     = detect_catalysts(symbol, announcements, screener_data=None)

    # ── Public data (insider trades, pledge, bulk deals) ──
    public_data  = None
    if universe_deals is not None:
        try:
            public_data = fetch_all_public_signals(
                symbol, info, universe_deals, session=nse_session
            )
        except Exception as e:
            log.debug(f"  {sym_clean}: public data failed — {e}")

    # ── Price stage ──
    price_stage  = classify_price_stage(info, hist)
    if "NEAR 52W HIGH" in price_stage:
        return None   # already late — skip

    # ── Tiered qualification gate ──
    soft_signals = 0
    if shareholding["insider_pct"] >= PCFG.min_promoter_holding_pct:
        soft_signals += 1
    if valuation["trading_near_book"]:
        soft_signals += 1
    if group["is_trusted_group"]:
        soft_signals += 1
    if debt["is_net_cash"] or debt["debt_reducing"]:
        soft_signals += 1
    if growth["growth_accelerating"]:
        soft_signals += 1
    if peer_pe["is_deep_value"]:
        soft_signals += 1

    tier1_pass = catalyst["catalyst_score"] >= PCFG.tier1_catalyst_threshold
    tier2_pass = soft_signals >= PCFG.tier2_min_signals

    if not tier1_pass and not tier2_pass:
        log.debug(
            f"  {sym_clean}: failed tiered gate "
            f"(catalyst={catalyst['catalyst_score']:.2f}, soft_signals={soft_signals})"
        )
        return None

    # ── Composite score ──
    public_score = (public_data or {}).get("public_score", 0.3)

    composite = (
        catalyst["catalyst_score"]    * PCFG.w_catalyst  +
        growth["growth_score"]         * PCFG.w_growth    +
        group["group_score"]           * PCFG.w_group     +
        shareholding["promoter_score"] * PCFG.w_promoter  +
        valuation["valuation_score"]   * PCFG.w_valuation +
        public_score                   * PCFG.w_public    +
        debt["debt_score"]             * PCFG.w_debt      +
        peer_pe["pe_score"]            * PCFG.w_pe_value
    )

    # Collect all flags for the report
    all_flags = []
    for src in [catalyst, fcf_data, piotroski_data, pe_traj, group, debt]:
        flag = src.get("piotroski_flag") or src.get("flag") or src.get("pe_traj_flag")
        if flag:
            all_flags.append(flag)
    for f in debt.get("flags", []):
        all_flags.append(f)
    if public_data:
        all_flags.extend(public_data.get("public_flags", []))
    if lynch_details.get("lynch_flags"):
        all_flags.extend(lynch_details["lynch_flags"][:2])

    return {
        "symbol":          symbol,
        "price":           round(price, 2),
        "market_cap_cr":   round(mktcap_cr, 0),
        "price_stage":     price_stage,
        "composite_score": round(composite, 4),
        "scanned_at":      datetime.utcnow().isoformat(),

        # Signal breakdowns
        "catalyst":        catalyst,
        "shareholding":    shareholding,
        "valuation":       valuation,
        "growth":          growth,
        "fcf":             fcf_data,
        "debt":            debt,
        "group":           group,
        "peer_pe":         peer_pe,
        "piotroski":       piotroski_data,
        "pe_trajectory":   pe_traj,
        "lynch":           lynch_details,
        "public_data":     public_data,

        # Flat fields for quick report access
        "all_flags":       all_flags,
        "bse_code":        bse_code,
        "announcement_count": len(announcements),

        # screener enriched later
        "screener":        None,
        "screener_flags":  [],
        "screener_roce":   None,
    }


# ──────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────

def run_pre_breakout_scanner(
    symbols: list[str],
    nse_session=None,
    universe_deals: dict = None,
    regime: dict = None,
) -> list[dict]:
    """
    Run Layer 2 pre-breakout scan on full universe.
    Returns top-N candidates, enriched with Screener.in data.
    """
    log.info(f"Layer 2: scanning {len(symbols)} symbols for pre-breakout setups...")

    candidates = []
    batch_size = 50
    sleep_between = 1.5

    for i, symbol in enumerate(symbols):
        try:
            result = analyse_pre_breakout(
                symbol,
                nse_session=nse_session,
                universe_deals=universe_deals,
            )
            if result:
                candidates.append(result)
                log.info(
                    f"  [{i+1}/{len(symbols)}] ✓ {symbol} "
                    f"score={result['composite_score']:.3f} "
                    f"P={result['piotroski'].get('piotroski_score','?')}/9 "
                    f"catalyst={result['catalyst']['catalyst_score']:.2f} "
                    f"anns={result['announcement_count']}"
                )
        except Exception as e:
            log.debug(f"  {symbol}: unhandled error — {e}")

        if (i + 1) % batch_size == 0:
            time.sleep(sleep_between)

    # Sort by composite score
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    # ── FIX C: ROCE gate post-enrichment ──
    # Screener enrichment runs on top-N, then we re-apply ROCE filter
    top_pre = candidates[:PCFG.top_n * 3]  # fetch more, filter down
    enriched = enrich_with_screener(top_pre, max_candidates=len(top_pre))

    # Re-filter by ROCE after Screener data is in
    final = []
    for c in enriched:
        roce = c.get("screener_roce")
        if roce is not None and roce < PCFG.min_roce_pct:
            log.info(
                f"  POST-FILTER: {c['symbol']} removed — "
                f"ROCE {roce:.1f}% < {PCFG.min_roce_pct}% (value trap risk)"
            )
            continue

        # FIX A (post-enrichment): re-run catalyst detection with Screener concalls
        sc_data = c.get("screener")
        if sc_data and not c["catalyst"]["catalysts"]:
            refreshed = detect_catalysts(
                c["symbol"],
                [],                   # BSE already fetched above
                screener_data=sc_data,
            )
            if refreshed["catalyst_score"] > c["catalyst"]["catalyst_score"]:
                c["catalyst"] = refreshed
                # Recompute composite with updated catalyst score
                public_score = (c.get("public_data") or {}).get("public_score", 0.3)
                c["composite_score"] = round(
                    refreshed["catalyst_score"]             * PCFG.w_catalyst  +
                    c["growth"]["growth_score"]              * PCFG.w_growth    +
                    c["group"]["group_score"]                * PCFG.w_group     +
                    c["shareholding"]["promoter_score"]      * PCFG.w_promoter  +
                    c["valuation"]["valuation_score"]        * PCFG.w_valuation +
                    public_score                             * PCFG.w_public    +
                    c["debt"]["debt_score"]                  * PCFG.w_debt      +
                    c["peer_pe"]["pe_score"]                 * PCFG.w_pe_value,
                    4,
                )

        final.append(c)
        if len(final) >= PCFG.top_n:
            break

    final.sort(key=lambda x: x["composite_score"], reverse=True)

    log.info(
        f"Layer 2 complete: {len(candidates)} passed gates, "
        f"{len(final)} after ROCE filter"
    )
    return final


# ──────────────────────────────────────────────────────
# REPORT GENERATOR
# ──────────────────────────────────────────────────────

def generate_pre_breakout_report(candidates: list[dict]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═" * 72
    lines = [
        sep,
        f"  LAYER 2: PRE-BREAKOUT WATCHLIST — {now}",
        "  Lynch GARP + VALUEPICK + Piotroski + Public Data",
        f"  (min Piotroski {PCFG.min_piotroski}/9 required | min ROCE {PCFG.min_roce_pct}% post-enrichment)",
        sep, "",
    ]

    if not candidates:
        lines.append("  No pre-breakout candidates found today.")
        return "\n".join(lines)

    for i, s in enumerate(candidates, 1):
        cat    = s["catalyst"]
        sh     = s["shareholding"]
        val    = s["valuation"]
        gr     = s["growth"]
        dbt    = s["debt"]
        grp    = s["group"]
        pio    = s["piotroski"]
        fcf    = s["fcf"]
        pe_tr  = s["pe_trajectory"]
        lynch  = s["lynch"]
        pub    = s.get("public_data") or {}
        sc     = s.get("screener") or {}
        sf     = s.get("screener_flags") or []

        roce_str = f"ROCE {sc.get('roce_pct'):.1f}%" if sc.get("roce_pct") else "ROCE N/A"

        lines += [
            f"  #{i:02d} {s['symbol']:<16} ₹{s['price']:<8} "
            f"MCap ₹{s['market_cap_cr']:.0f}Cr  Score: {s['composite_score']:.3f}",
            f"  Stage: {s['price_stage']}",
            f"  Piotroski: {pio.get('piotroski_score','?')}/9  "
            f"{roce_str}  "
            f"FCF Yield: {fcf.get('fcf_yield_pct','N/A')}%  "
            f"PE Expanding: {'✓' if pe_tr.get('pe_expanding') else '✗'}",
            f"  Promoter: {sh['insider_pct']:.1f}%  "
            f"P/B: {val['pb_ratio']:.2f}  "
            f"RevGrowth: {gr['revenue_growth_pct']:+.1f}%  "
            f"D/E: {dbt['current_de']:.2f}",
            f"  Lynch: PEG {lynch.get('peg_ratio','N/A')}  "
            f"Inst%: {lynch.get('inst_own_pct','N/A')}  "
            f"Class: {lynch.get('lynch_class','?')}",
        ]

        if grp.get("matched_group"):
            lines.append(f"  Group: 🏛️ {grp['matched_group']}")

        if cat["catalysts"]:
            lines.append(f"  Catalysts ({cat['recent_announcements']} BSE anns):")
            for c_item in cat["catalysts"][:4]:
                lines.append(f"    ▶ {c_item}")
        else:
            lines.append(
                f"  ⚠ No catalysts detected ({cat['recent_announcements']} BSE anns scanned)"
            )

        if sf:
            lines.append(f"  Screener: {' | '.join(sf[:3])}")

        pub_parts = []
        if pub.get("promoter_buying"):
            pub_parts.append(f"🧑‍💼 Insider ₹{pub.get('insider_value_cr',0):.1f}Cr")
        if pub.get("institutional_buying"):
            pub_parts.append("🏦 Bulk Buy")
        if pub.get("pledge_pct") is not None:
            if pub["pledge_pct"] <= 0:
                pub_parts.append("✅ No Pledge")
            elif pub.get("is_high_pledge"):
                pub_parts.append(f"🚨 Pledge {pub['pledge_pct']:.0f}%")
        if pub_parts:
            lines.append(f"  Public: {' | '.join(pub_parts)}")

        lines.append("")

    lines += [
        sep,
        "  KEY:",
        "  Piotroski ≥8 + ROCE >20% + Catalyst = highest conviction",
        "  PEG <1.0 + Inst% <10% = Lynch undiscovered compounder",
        "  🧑‍💼 insider buying = VALUEPICK signature — prioritise these",
        sep,
    ]

    return "\n".join(lines)
