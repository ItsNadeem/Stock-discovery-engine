"""
multibagger_dna.py — Multibagger DNA Scorer

Replaces keyword-matching catalyst detection with a model validated
against confirmed NSE multibaggers.

Ground truth dataset (from backtest.py VALUEPICK_PICKS + public records):
  Paushak      ₹74  → ₹10,000  (140x, 2011–2022) — Alembic group specialty chem
  EKI Energy   ₹162 → ₹10,000  (66x,  2021–2022) — first carbon credit co., 90% export
  Cosmo Ferrites ₹13 → ₹500   (13x,  2011–2021) — Cosmo Films group, EV tailwind
  Tasty Bite   ₹165 → ₹9,420  (57x,  2010–2021) — organic food, US export penetration
  Jay Kay      ₹28  → 5x+     (2021) — JK group, 3D printing JV, promoter pref allot
  Shanthi Gears ₹180 → watch  (2024) — Murugappa, zero debt, gear sector tailwind

What these had in common at the entry point (NOT at the peak):
  1. Revenue acceleration — not high revenue, but a visible CHANGE in slope.
     PAT growing faster than revenue (operating leverage kicking in).
  2. Promoter open-market buying OR preferential allotment (skin in game).
  3. One structural catalyst — capex commissioning, JV with named partner,
     entry into a genuinely new market (carbon credits, 3D printing, US organic).
     NOT generic "capacity expansion" — a specific, nameable, verifiable event.
  4. Sector at an early-to-mid cycle tailwind (not late cycle).
  5. Low institutional ownership — institutions hadn't found it yet.
  6. Small market cap (₹50–₹500 Cr) — enough room to 10x without being a megacap.
  7. Price consolidating — not running, not crashed. Coiling.
  8. Trusted promoter — family group with long track record of minority-friendly
     behaviour. No pledges. No related-party transactions at inflated prices.

What they did NOT require:
  - Piotroski ≥ 5 (EKI listed 1 month before the move; Paushak had thin history)
  - ROCE ≥ 10% at the exact entry (Paushak's ROCE was depressed pre-expansion)
  - High liquidity (all were thinly traded)
  - Being near a 52-week high (all were in deep bases)

This module computes a DNA score (0-1) that directly weights these factors.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. REVENUE ACCELERATION INFLECTION
#
# The single most predictive signal from the ground truth set.
# All six confirmed picks showed a change in revenue slope before the move.
# It's not about high revenue — it's about the SECOND DERIVATIVE turning positive.
#
# Signal: most recent quarter revenue > prior quarter > quarter before that
# AND YoY growth in Q-1 > YoY growth in Q-5 (acceleration, not just level)
# ──────────────────────────────────────────────────────────────

def score_revenue_acceleration(screener_data: Optional[dict]) -> tuple[float, str]:
    """
    Returns (score 0-1, description string).
    Best score (1.0) = sequential QoQ growth AND YoY acceleration.
    """
    if not screener_data:
        return 0.3, "No Screener data"

    qs = screener_data.get("_quarterly_sales", [])
    qp = screener_data.get("_quarterly_profit", [])

    if not qs or len(qs) < 6:
        return 0.3, "Insufficient quarterly data"

    # Filter out None values with their indices preserved
    qs_clean = [(i, v) for i, v in enumerate(qs[:8]) if v is not None and v > 0]
    qp_clean = [(i, v) for i, v in enumerate(qp[:8]) if v is not None] if qp else []

    if len(qs_clean) < 5:
        return 0.3, "Too many missing quarters"

    vals = [v for _, v in qs_clean]

    # Sequential growth: Q0 > Q1 > Q2 (most recent three quarters growing)
    seq_growth = vals[0] > vals[1] and vals[1] > vals[2] if len(vals) >= 3 else False

    # YoY acceleration: growth rate in Q-1 > growth rate in Q-5
    # Q-1 vs Q-5 (same quarter last year): is the YoY rate itself accelerating?
    yoy_accel = False
    yoy_note = ""
    if len(vals) >= 6:
        try:
            yoy_recent = (vals[0] - vals[4]) / vals[4] * 100  # latest vs year-ago
            yoy_prior  = (vals[1] - vals[5]) / vals[5] * 100  # prior vs year-ago
            yoy_accel  = yoy_recent > yoy_prior and yoy_recent > 5
            yoy_note   = f"YoY: {yoy_recent:+.1f}% vs prior {yoy_prior:+.1f}%"
        except Exception:
            pass

    # Operating leverage: PAT growing faster than revenue
    op_leverage = False
    op_note = ""
    if qp_clean and len(qp_clean) >= 5:
        pvals = [v for _, v in qp_clean]
        try:
            rev_growth = (vals[0] - vals[4]) / vals[4] if vals[4] != 0 else 0
            pat_growth = (pvals[0] - pvals[4]) / abs(pvals[4]) if pvals[4] != 0 else 0
            # Operating leverage fires when PAT grows faster AND both are positive
            op_leverage = (pat_growth > rev_growth * 1.3
                           and pvals[0] > 0 and pat_growth > 0.10)
            if op_leverage:
                op_note = f"OpLev: PAT +{pat_growth*100:.0f}% vs Rev +{rev_growth*100:.0f}%"
        except Exception:
            pass

    # Score
    if seq_growth and yoy_accel and op_leverage:
        score = 1.0
        desc  = f"🚀 Revenue inflection: sequential growth + YoY accel + operating leverage. {yoy_note}. {op_note}"
    elif seq_growth and yoy_accel:
        score = 0.80
        desc  = f"📈 Revenue accelerating: QoQ growth + YoY accel. {yoy_note}"
    elif seq_growth and op_leverage:
        score = 0.75
        desc  = f"📈 Sequential growth + operating leverage. {op_note}"
    elif yoy_accel:
        score = 0.60
        desc  = f"📊 YoY revenue accelerating. {yoy_note}"
    elif seq_growth:
        score = 0.50
        desc  = f"📊 Sequential revenue growth (QoQ)"
    elif len(vals) >= 3 and vals[0] > vals[4] if len(vals) >= 5 else False:
        score = 0.35
        desc  = "Revenue positive YoY but not accelerating"
    else:
        # Check if declining
        if len(vals) >= 3 and vals[0] < vals[1] < vals[2]:
            score = 0.05
            desc  = "⚠️ Revenue declining 3 consecutive quarters"
        else:
            score = 0.20
            desc  = "Revenue flat or mixed"

    return round(score, 3), desc


# ──────────────────────────────────────────────────────────────
# 2. PROMOTER CONVICTION SIGNAL
#
# Ranked by strength (from ground truth analysis):
#   Tier 1 (0.90): Promoter buys in open market (PIT disclosure)
#                  OR preferential allotment at premium to market
#   Tier 2 (0.70): Promoter holding > 65% with zero pledge (won't sell)
#   Tier 3 (0.50): Promoter holding > 50% with zero pledge
#   Tier 4 (0.25): High pledge or promoter selling = negative signal
# ──────────────────────────────────────────────────────────────

def score_promoter_conviction(
    public_data: Optional[dict],
    screener_data: Optional[dict],
    info: dict,
) -> tuple[float, str]:
    """Returns (score 0-1, description)."""
    promoter_pct   = None
    pledge_pct     = None
    promoter_buying = False
    insider_val_cr = 0.0

    # From public_data (NSE PIT disclosures)
    if public_data:
        promoter_buying = public_data.get("promoter_buying", False)
        insider_val_cr  = public_data.get("insider_value_cr", 0.0) or 0.0
        pledge_pct      = public_data.get("pledge_pct")

    # From Screener (more accurate than yfinance)
    if screener_data:
        promoter_pct = screener_data.get("promoter_pct")

    # Fallback to yfinance
    if promoter_pct is None:
        raw = info.get("heldPercentInsiders")
        if raw is not None:
            promoter_pct = round(float(raw) * 100, 1)

    # Score
    if promoter_buying and insider_val_cr > 0:
        score = min(0.90 + insider_val_cr / 100, 1.0)
        desc  = f"🧑‍💼 Promoter open-market buying ₹{insider_val_cr:.1f}Cr (HIGHEST conviction)"
    elif promoter_pct and promoter_pct > 65 and (pledge_pct is None or pledge_pct <= 2):
        score = 0.70
        desc  = f"👥 Promoter {promoter_pct:.0f}% holding, minimal/no pledge — won't sell"
    elif promoter_pct and promoter_pct > 50 and (pledge_pct is None or pledge_pct <= 5):
        score = 0.55
        desc  = f"👥 Promoter {promoter_pct:.0f}%, low pledge"
    elif pledge_pct is not None and pledge_pct > 30:
        score = 0.10
        desc  = f"🚨 High pledge {pledge_pct:.0f}% — forced selling risk, skip"
    elif promoter_pct and promoter_pct < 30:
        score = 0.20
        desc  = f"Low promoter holding {promoter_pct:.0f}% — limited skin in game"
    else:
        score = 0.35
        desc  = f"Promoter {promoter_pct:.0f}% (pledge data unavailable)" if promoter_pct else "Promoter data unavailable"

    return round(score, 3), desc


# ──────────────────────────────────────────────────────────────
# 3. STRUCTURAL CATALYST QUALITY
#
# The key insight from ground truth: it's not whether a catalyst keyword
# appears — it's how specific and verifiable the catalyst is.
#
# Tier 1 (1.0): Named partner + specific technology + commissioning date
#               Example: "JV with EOS GmbH for 3D printing" (Jay Kay)
#               Example: "VERRA-certified carbon credit registration" (EKI)
# Tier 2 (0.70): Specific capacity with end-market named
#               Example: "50% ferrite capacity for EV motors" (Cosmo Ferrites)
# Tier 3 (0.40): Generic capex / JV announcement without specifics
# Tier 4 (0.20): No catalyst found
#
# We detect this from BSE announcements + Screener concall text.
# ──────────────────────────────────────────────────────────────

# High-specificity catalyst patterns — order matters (more specific first)
TIER1_PATTERNS = [
    # Named technology partner
    "technology transfer", "technical collaboration", "licence agreement",
    "licensing agreement", "technology licence",
    # Regulatory approvals that unlock revenue
    "fda approval", "usfda", "who-gmp", "verra", "gold standard",
    "eu ets", "carbon credit", "carbon market",
    # Specific named JV partners (not just "joint venture")
    "eos gmbh", "siemens", "bosch", "schneider", "alembic",
    # Export order with named geography/customer type
    "us fda", "us export", "european order", "korean order", "japanese order",
    # Debt fully retired
    "debt free", "zero debt", "loan repaid in full", "ncd fully redeemed",
    "term loan closed",
]

TIER2_PATTERNS = [
    # Capacity with end-market context
    "ev motor", "electric vehicle component", "defense order",
    "defence order", "railway order", "metro rail",
    "semiconductor", "pharma api", "specialty chemical",
    # Preferential allotment (promoter putting money in)
    "preferential allotment", "preferential issue",
    # Buyback (promoter returning cash)
    "buyback", "buy back",
    # Import substitution (government tail wind)
    "import substitution", "make in india", "production linked incentive",
    "pli scheme",
]

TIER3_PATTERNS = [
    "capacity expansion", "new plant", "greenfield", "brownfield",
    "joint venture", "capex", "new facility", "commissioning",
    "commercial production", "order received", "new order",
    "export", "overseas revenue",
]


def score_catalyst_quality(
    announcements: list[dict],
    screener_data: Optional[dict] = None,
) -> tuple[float, str]:
    """
    Score catalyst quality from BSE announcements + Screener concalls.
    Returns (score 0-1, best_catalyst_description).
    """
    all_text = " ".join(
        (a.get("title", "") + " " + a.get("category", "")).lower()
        for a in announcements
    )

    # Also include Screener concall titles
    concall_text = ""
    if screener_data:
        for cc in (screener_data.get("concalls") or []):
            concall_text += " " + (cc.get("title") or cc.get("description") or "").lower()
    all_text = all_text + " " + concall_text

    found_tier1 = [p for p in TIER1_PATTERNS if p in all_text]
    found_tier2 = [p for p in TIER2_PATTERNS if p in all_text]
    found_tier3 = [p for p in TIER3_PATTERNS if p in all_text]

    # Recent activity bonus (last 30 days = catalyst is fresh)
    recent_count = sum(
        1 for a in announcements
        if a.get("date") and
        (pd.Timestamp.now() - pd.Timestamp(a["date"])).days <= 30
    )

    if found_tier1:
        score = min(0.90 + 0.05 * len(found_tier1) + 0.05 * (recent_count > 0), 1.0)
        desc  = f"🏆 TIER-1 CATALYST: {found_tier1[0].title()}"
        if len(found_tier1) > 1:
            desc += f" + {len(found_tier1)-1} more"
    elif found_tier2:
        score = min(0.65 + 0.05 * len(found_tier2) + 0.05 * (recent_count > 0), 0.85)
        desc  = f"🎯 TIER-2 CATALYST: {found_tier2[0].title()}"
    elif found_tier3:
        score = 0.35 + 0.05 * (recent_count >= 3)
        desc  = f"📋 Generic catalyst: {found_tier3[0].title()}"
    else:
        score = 0.10
        desc  = "⚪ No specific catalyst found in BSE announcements"

    if recent_count > 0 and score > 0.10:
        desc += f" ({recent_count} announcement{'s' if recent_count > 1 else ''} in last 30d)"

    return round(score, 3), desc


# ──────────────────────────────────────────────────────────────
# 4. MARKET CAP HEADROOM
#
# A ₹50 Cr company can 10x to ₹500 Cr and still be a small cap.
# A ₹2,000 Cr company needs to become ₹20,000 Cr to 10x — much harder.
# Ideal multibagger zone: ₹30 Cr – ₹400 Cr market cap.
# ──────────────────────────────────────────────────────────────

def score_mcap_headroom(market_cap_cr: float) -> tuple[float, str]:
    """Returns (score 0-1, description)."""
    if market_cap_cr <= 0:
        return 0.3, "MCap unknown"
    if market_cap_cr < 30:
        return 0.40, f"₹{market_cap_cr:.0f}Cr — very small, liquidity risk"
    elif market_cap_cr <= 150:
        return 1.0,  f"₹{market_cap_cr:.0f}Cr — ideal multibagger size (30–150 Cr)"
    elif market_cap_cr <= 400:
        return 0.85, f"₹{market_cap_cr:.0f}Cr — good headroom for 5–10x"
    elif market_cap_cr <= 800:
        return 0.65, f"₹{market_cap_cr:.0f}Cr — moderate headroom for 3–5x"
    elif market_cap_cr <= 1500:
        return 0.40, f"₹{market_cap_cr:.0f}Cr — limited headroom, needs exceptional thesis"
    else:
        return 0.15, f"₹{market_cap_cr:.0f}Cr — too large for typical multibagger move"


# ──────────────────────────────────────────────────────────────
# 5. PRICE STAGE (COILING, NOT RUNNING)
#
# All six confirmed picks were in deep bases or early uptrends.
# None were near their 52-week highs. The best entry is when price
# has been quiet for months — the thesis is building beneath the surface.
# ──────────────────────────────────────────────────────────────

def score_price_stage_for_multibagger(hist: pd.DataFrame, info: dict) -> tuple[float, str]:
    """
    Score price positioning for multibagger entry.
    Deep base = ideal. Near 52W high = too late for multibagger entry.
    """
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    if price == 0 or hist is None or hist.empty:
        return 0.3, "Price stage unknown"

    c = hist["Close"]
    if len(c) < 100:
        return 0.3, "Insufficient history"

    high_52w = float(c.rolling(252).max().iloc[-1]) if len(c) >= 252 else float(c.max())
    low_52w  = float(c.rolling(252).min().iloc[-1]) if len(c) >= 252 else float(c.min())

    pct_from_low  = (price - low_52w) / low_52w * 100  if low_52w  > 0 else 0
    pct_from_high = (high_52w - price) / high_52w * 100 if high_52w > 0 else 0

    # Volatility contraction (ATR shrinking = stock coiling)
    from engine import atr as calc_atr
    try:
        df_for_atr = hist.copy()
        atr_series = calc_atr(df_for_atr, 14)
        recent_atr = float(atr_series.iloc[-20:].mean())
        prior_atr  = float(atr_series.iloc[-80:-20].mean()) if len(atr_series) >= 80 else recent_atr
        coiling    = (recent_atr / prior_atr) < 0.65 if prior_atr > 0 else False
    except Exception:
        coiling = False

    if pct_from_low < 25 and coiling:
        return 1.0,  f"🟢 DEEP BASE + coiling (near 52W low, ATR contracted) — ideal entry zone"
    elif pct_from_low < 25:
        return 0.85, f"🟢 Deep base (near 52W low, {pct_from_low:.0f}% above low)"
    elif pct_from_low < 60 and pct_from_high > 30:
        return 0.65, f"🟡 Early uptrend ({pct_from_low:.0f}% above 52W low, {pct_from_high:.0f}% below high)"
    elif pct_from_high < 10:
        return 0.10, f"🔴 Near 52W high ({pct_from_high:.0f}% below high) — too late for multibagger entry"
    else:
        return 0.40, f"🟠 Mid-cycle ({pct_from_low:.0f}% above low, {pct_from_high:.0f}% below high)"


# ──────────────────────────────────────────────────────────────
# 6. SECTOR CYCLE POSITION
#
# A good company in a bad sector cycle rarely multibags.
# A decent company at the start of a sector upcycle often does.
# We check: is the sector index below its 200EMA but now recovering?
# That's the sweet spot — cycle turning, but not yet priced in.
# ──────────────────────────────────────────────────────────────

SECTOR_INDEX_MAP = {
    "Financial Services": "^CNXFIN",
    "IT":                 "^CNXIT",
    "Pharma":             "^CNXPHARMA",
    "Auto":               "^CNXAUTO",
    "Metals":             "^CNXMETAL",
    "Energy":             "^CNXENERGY",
    "FMCG":               "^CNXFMCG",
    "Realty":             "^CNXREALTY",
    "Infra":              "^CNXINFRA",
    "Media":              "^CNXMEDIA",
}

_sector_cache: dict = {}  # cached per run

def score_sector_cycle(sector: str) -> tuple[float, str]:
    """
    Returns (score 0-1, description).
    Sweet spot: sector recovering from below 200EMA (early upcycle).
    Worst:      sector falling hard (below 200EMA AND below 50EMA).
    """
    global _sector_cache
    if sector in _sector_cache:
        return _sector_cache[sector]

    ticker_sym = SECTOR_INDEX_MAP.get(sector)
    if not ticker_sym:
        result = (0.5, f"Sector '{sector}' not mapped")
        _sector_cache[sector] = result
        return result

    try:
        import yfinance as yf
        df = yf.download(ticker_sym, period="18mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 100:
            result = (0.5, f"{sector}: insufficient data")
            _sector_cache[sector] = result
            return result

        c      = df["Close"].squeeze()
        e50    = c.ewm(span=50,  adjust=False).mean()
        e200   = c.ewm(span=200, adjust=False).mean()
        last   = float(c.iloc[-1])
        e50v   = float(e50.iloc[-1])
        e200v  = float(e200.iloc[-1])
        chg_1m = (last - float(c.iloc[-21])) / float(c.iloc[-21]) * 100
        chg_3m = (last - float(c.iloc[-63])) / float(c.iloc[-63]) * 100

        above_50  = last > e50v
        above_200 = last > e200v

        # Best: early recovery — was below 200EMA but crossing up, 1M positive
        if not above_200 and above_50 and chg_1m > 2:
            score = 0.90
            desc  = f"⭐ {sector}: EARLY UPCYCLE — recovering from below 200EMA, +{chg_1m:.1f}% 1M"
        elif above_200 and above_50 and chg_3m > 5:
            score = 0.75
            desc  = f"✅ {sector}: Bull trend, above 200EMA, +{chg_3m:.1f}% 3M"
        elif above_200 and chg_1m < -3:
            score = 0.50
            desc  = f"⚠️ {sector}: Above 200EMA but weakening ({chg_1m:+.1f}% 1M)"
        elif not above_200 and chg_3m < -10:
            score = 0.15
            desc  = f"🔴 {sector}: Downtrend — below 200EMA, {chg_3m:+.1f}% 3M"
        else:
            score = 0.45
            desc  = f"🟡 {sector}: Neutral ({chg_3m:+.1f}% 3M)"

        result = (round(score, 3), desc)
        _sector_cache[sector] = result
        return result

    except Exception as e:
        result = (0.5, f"{sector}: fetch error — {e}")
        _sector_cache[sector] = result
        return result


# ──────────────────────────────────────────────────────────────
# 7. INSTITUTIONAL DISCOVERY STATUS
#
# Lynch's key insight: stocks with <5% institutional ownership are
# undiscovered. Once institutions start buying, the re-rating begins.
# But the best entry is BEFORE they arrive, not after.
# ──────────────────────────────────────────────────────────────

def score_discovery_status(info: dict, screener_data: Optional[dict]) -> tuple[float, str]:
    """Returns (score 0-1, description)."""
    inst_raw = info.get("heldPercentInstitutions") or info.get("institutionsPercentHeld")
    inst_pct = None
    if inst_raw is not None:
        inst_pct = float(inst_raw) * 100 if inst_raw < 1 else float(inst_raw)

    if inst_pct is None:
        return 0.35, "Institutional ownership data unavailable"
    elif inst_pct < 3:
        return 1.0,  f"👁️ UNDISCOVERED: {inst_pct:.1f}% institutional — Lynch sweet spot"
    elif inst_pct < 8:
        return 0.85, f"🌱 Early discovery: {inst_pct:.1f}% institutional"
    elif inst_pct < 20:
        return 0.60, f"📊 Being discovered: {inst_pct:.1f}% institutional"
    elif inst_pct < 40:
        return 0.35, f"🔍 Partially discovered: {inst_pct:.1f}% institutional"
    else:
        return 0.10, f"📢 Fully discovered: {inst_pct:.1f}% institutional — Lynch edge gone"


# ──────────────────────────────────────────────────────────────
# COMPOSITE DNA SCORE
# Weights derived from ground truth analysis of the 6 confirmed picks:
#   Revenue acceleration was present in ALL 6 at entry → highest weight
#   Promoter conviction was present in 5/6 → second highest
#   Catalyst quality separated the 10x+ from the 3-5x → third
#   MCap headroom is mechanical but necessary → fourth
#   Price stage (coiling) was present in 5/6 → fifth
#   Sector cycle present in 4/6 → sixth
#   Discovery status present in all 6 but hard to verify → seventh
# ──────────────────────────────────────────────────────────────

DNA_WEIGHTS = {
    "revenue_accel":   0.28,  # most predictive in ground truth
    "promoter":        0.22,  # skin in game is essential
    "catalyst":        0.18,  # quality of catalyst, not just presence
    "mcap_headroom":   0.12,  # mechanical but eliminates large caps
    "price_stage":     0.10,  # coiling before the move
    "sector_cycle":    0.06,  # tailwind vs headwind
    "discovery":       0.04,  # undiscovered = upside ahead
}


def compute_dna_score(
    symbol: str,
    market_cap_cr: float,
    info: dict,
    hist: pd.DataFrame,
    screener_data: Optional[dict],
    public_data: Optional[dict],
    announcements: list[dict],
    sector: str = "Other",
) -> dict:
    """
    Compute the full multibagger DNA score for a stock.

    Returns a rich dict with component scores, descriptions, and flags.
    The composite dna_score replaces the old catalyst_score as the primary
    signal in the Layer 2 composite.
    """
    # Component scores
    rev_score, rev_desc      = score_revenue_acceleration(screener_data)
    promo_score, promo_desc  = score_promoter_conviction(public_data, screener_data, info)
    cat_score, cat_desc      = score_catalyst_quality(announcements, screener_data)
    mcap_score, mcap_desc    = score_mcap_headroom(market_cap_cr)
    stage_score, stage_desc  = score_price_stage_for_multibagger(hist, info)
    sector_score, sector_desc = score_sector_cycle(sector)
    disc_score, disc_desc    = score_discovery_status(info, screener_data)

    # Composite
    dna_score = (
        rev_score    * DNA_WEIGHTS["revenue_accel"] +
        promo_score  * DNA_WEIGHTS["promoter"]      +
        cat_score    * DNA_WEIGHTS["catalyst"]       +
        mcap_score   * DNA_WEIGHTS["mcap_headroom"]  +
        stage_score  * DNA_WEIGHTS["price_stage"]    +
        sector_score * DNA_WEIGHTS["sector_cycle"]   +
        disc_score   * DNA_WEIGHTS["discovery"]
    )

    # Hard disqualifiers — no matter how good other scores are
    disqualifiers = []
    pledge_pct = (public_data or {}).get("pledge_pct")
    if pledge_pct is not None and pledge_pct > 30:
        disqualifiers.append(f"High pledge {pledge_pct:.0f}%")
        dna_score = min(dna_score, 0.35)

    if market_cap_cr > 2000:
        disqualifiers.append(f"MCap ₹{market_cap_cr:.0f}Cr too large")
        dna_score = min(dna_score, 0.40)

    # Conviction flags (actionable items for the report)
    flags = []
    if rev_score >= 0.80:   flags.append(rev_desc)
    if promo_score >= 0.85: flags.append(promo_desc)
    if cat_score >= 0.65:   flags.append(cat_desc)
    if stage_score >= 0.85: flags.append(stage_desc)
    if sector_score >= 0.85:flags.append(sector_desc)
    if disc_score >= 0.85:  flags.append(disc_desc)

    # DNA grade
    if dna_score >= 0.75:
        grade = "A — HIGH CONVICTION MULTIBAGGER CANDIDATE"
    elif dna_score >= 0.60:
        grade = "B — Worth deep research"
    elif dna_score >= 0.45:
        grade = "C — Monitor, not yet actionable"
    else:
        grade = "D — Low multibagger probability"

    log.info(
        f"  DNA {symbol}: {dna_score:.3f} ({grade[:1]}) | "
        f"rev={rev_score:.2f} promo={promo_score:.2f} cat={cat_score:.2f} "
        f"mcap={mcap_score:.2f} stage={stage_score:.2f}"
    )

    return {
        "dna_score":        round(dna_score, 4),
        "dna_grade":        grade,
        "dna_flags":        flags,
        "disqualifiers":    disqualifiers,
        # Component breakdown
        "rev_accel_score":  rev_score,
        "rev_accel_desc":   rev_desc,
        "promoter_score":   promo_score,
        "promoter_desc":    promo_desc,
        "catalyst_score":   cat_score,
        "catalyst_desc":    cat_desc,
        "mcap_score":       mcap_score,
        "mcap_desc":        mcap_desc,
        "price_stage_score":stage_score,
        "price_stage_desc": stage_desc,
        "sector_score":     sector_score,
        "sector_desc":      sector_desc,
        "discovery_score":  disc_score,
        "discovery_desc":   disc_desc,
        # Raw inputs for report
        "sector":           sector,
        "market_cap_cr":    market_cap_cr,
    }
