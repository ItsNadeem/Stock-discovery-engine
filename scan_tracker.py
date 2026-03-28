"""
scan_tracker.py — Scan History Tracker

The single highest-ROI improvement to the engine after backtesting.

FIX v2 (2026-03): Bear-regime aware watch_now gate.
  Previously is_watch_now() passed any stock with streak ≥ 3, score not
  falling, and Piotroski ≥ 4 — regardless of market regime.

  In a BEAR regime this produces a watch list of cheap stocks going cheaper
  (SHANTIGOLD -10.3%, SOLARWORLD -10.4%) with no catalyst and no floor.

  Fix: in BEAR or NEUTRAL regime, watch_now requires EITHER:
    - Piotroski ≥ 7, OR
    - catalyst_score > 0.10 (at least one real BSE catalyst detected)
  A stock in a bear market with Piotroski 4/9 and no catalyst is not a
  watch target — it is a value trap. The regime is passed in from run_all.py.

  Also added: price_since_first_pct sign-check — stocks down >15% since
  first seen AND no catalyst are automatically demoted from watch_now in
  a bear regime (they are falling knives, not coiling springs).
"""

import json
import logging
import os
import glob
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

HISTORY_FILE    = "results/history.json"
LOOKBACK_DAYS   = 30
STREAK_THRESHOLD = 3
SCORE_RISE_MIN  = 0.02


# ─────────────────────────────────────────────────────────────
# HISTORY STORE
# ─────────────────────────────────────────────────────────────

def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load history: {e}")
    return {}


def save_history(history: dict):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# SCAN FILE LOADER
# ─────────────────────────────────────────────────────────────

def load_past_scans(lookback_days: int = LOOKBACK_DAYS) -> list[tuple[str, list[dict]]]:
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    files  = sorted(glob.glob("results/watchlist_*.json"), reverse=True)
    scans  = []
    for f in files:
        base = os.path.basename(f)
        try:
            date_str = base.replace("watchlist_", "").replace(".json", "")
            file_dt  = datetime.strptime(date_str, "%Y%m%d")
            if file_dt < cutoff:
                continue
            with open(f) as fh:
                data = json.load(fh)
            scans.append((date_str, data))
        except Exception:
            continue
    return scans


# ─────────────────────────────────────────────────────────────
# CORE TRACKER
# ─────────────────────────────────────────────────────────────

def update_history(
    today_watchlist: list[dict],
    today_date: str,
    history: dict,
) -> dict:
    for stock in today_watchlist:
        sym = stock["symbol"]
        if sym not in history:
            history[sym] = {"first_seen": today_date, "appearances": [], "all_catalysts": []}

        sc_data = stock.get("screener") or {}
        history[sym]["appearances"].append({
            "date":          today_date,
            "price":         stock.get("price"),
            "score":         stock.get("composite_score"),
            "price_stage":   stock.get("price_stage"),
            "piotroski":     stock.get("piotroski", {}).get("piotroski_score"),
            "fcf_yield":     stock.get("fcf", {}).get("fcf_yield_pct"),
            "roce":          sc_data.get("roce_pct"),
            "catalyst_score":stock.get("catalyst", {}).get("catalyst_score", 0),
            "catalysts":     stock.get("catalyst", {}).get("catalysts", []),
            "group":         stock.get("group", {}).get("matched_group"),
            "debt_reducing": stock.get("debt", {}).get("debt_reducing", False),
            "pe_expanding":  stock.get("pe_trajectory", {}).get("pe_expanding", False),
        })

        new_cats = stock.get("catalyst", {}).get("catalysts", [])
        existing = set(history[sym]["all_catalysts"])
        for c in new_cats:
            if c not in existing:
                history[sym]["all_catalysts"].append(c)
                existing.add(c)

        history[sym]["appearances"] = history[sym]["appearances"][-90:]

    return history


def compute_persistence(
    symbol: str,
    history: dict,
    past_scans: list[tuple[str, list[dict]]],
    lookback_days: int = LOOKBACK_DAYS,
) -> dict:
    sym_history  = history.get(symbol, {})
    appearances  = sym_history.get("appearances", [])

    if not appearances:
        return {
            "days_in_watchlist": 0, "streak": 0, "score_trend": "new",
            "score_7d_delta": None, "first_seen": None, "peak_score": None,
            "price_since_first_pct": None, "all_catalysts": [],
            "latest_catalyst_score": 0.0,
        }

    cutoff_dt  = datetime.utcnow() - timedelta(days=lookback_days)
    recent     = [
        a for a in appearances
        if datetime.strptime(a["date"], "%Y%m%d") >= cutoff_dt
    ]
    days_in_watchlist = len(recent)

    # Current streak
    all_scan_dates = sorted({date for date, _ in past_scans}, reverse=True)
    sym_dates = set(a["date"] for a in appearances)
    streak = 0
    for scan_date in all_scan_dates:
        if scan_date in sym_dates:
            streak += 1
        else:
            break

    # Score trend
    scores = [a["score"] for a in appearances if a.get("score") is not None]
    score_7d_delta = None
    score_trend    = "stable"
    if len(scores) >= 2:
        recent_avg = sum(scores[-3:]) / len(scores[-3:])
        prior_avg  = sum(scores[-6:-3]) / len(scores[-6:-3]) if len(scores) >= 6 else scores[0]
        delta      = recent_avg - prior_avg
        score_7d_delta = round(delta, 3)
        if delta > SCORE_RISE_MIN:
            score_trend = "rising"
        elif delta < -SCORE_RISE_MIN:
            score_trend = "falling"

    peak_score = round(max(scores), 3) if scores else None

    # Price change since first seen
    first_price   = appearances[0].get("price")
    current_price = appearances[-1].get("price")
    price_change  = None
    if first_price and current_price and first_price > 0:
        price_change = round((current_price - first_price) / first_price * 100, 1)

    # Latest catalyst score (from most recent appearance)
    latest_catalyst_score = appearances[-1].get("catalyst_score", 0.0) if appearances else 0.0

    return {
        "days_in_watchlist":     days_in_watchlist,
        "streak":                streak,
        "score_trend":           score_trend,
        "score_7d_delta":        score_7d_delta,
        "first_seen":            sym_history.get("first_seen"),
        "peak_score":            peak_score,
        "price_since_first_pct": price_change,
        "all_catalysts":         sym_history.get("all_catalysts", []),
        "latest_catalyst_score": latest_catalyst_score,
    }


def is_watch_now(
    stock: dict,
    persistence: dict,
    regime: str = "NEUTRAL",
) -> tuple[bool, str]:
    """
    FIX: Regime-aware watch_now gate.

    BEAR / NEUTRAL regime: require Piotroski ≥ 7 OR catalyst_score > 0.10.
    This prevents the watch list from filling with cheap stocks going cheaper.

    BULL regime: standard gate (streak ≥ 3, score not falling, Piotroski ≥ 4).
    """
    streak         = persistence.get("streak", 0)
    score_trend    = persistence.get("score_trend", "")
    piotroski      = stock.get("piotroski", {}).get("piotroski_score")
    stage          = stock.get("price_stage", "")
    catalyst_score = persistence.get("latest_catalyst_score", 0.0)
    price_chg      = persistence.get("price_since_first_pct")

    # Base gates (all regimes)
    if streak < STREAK_THRESHOLD:
        return False, f"Streak only {streak} (need {STREAK_THRESHOLD}+)"
    if score_trend == "falling":
        return False, "Score declining"
    if "NEAR 52W HIGH" in stage:
        return False, "Near 52W high — too late"
    if piotroski is not None and piotroski < 4:
        return False, f"Piotroski {piotroski}/9 too low"

    # FIX: Bear/Neutral regime — stricter quality floor
    if regime in ("BEAR", "NEUTRAL"):
        has_quality   = piotroski is not None and piotroski >= 7
        has_catalyst  = catalyst_score > 0.10
        falling_knife = price_chg is not None and price_chg < -15 and not has_catalyst

        if falling_knife:
            return False, f"Down {price_chg:.1f}% since first seen with no catalyst — falling knife in {regime} regime"
        if not has_quality and not has_catalyst:
            return False, (
                f"{regime} regime: need Piotroski ≥7 OR catalyst_score >0.10 "
                f"(got {piotroski}/9 and catalyst={catalyst_score:.2f})"
            )

    # Build reason string
    reasons = [f"Streak {streak} scans"]
    if score_trend == "rising":
        reasons.append(f"score rising (+{persistence.get('score_7d_delta', '?')})")
    if piotroski and piotroski >= 7:
        reasons.append(f"Piotroski {piotroski}/9")
    if catalyst_score > 0.10:
        reasons.append(f"catalyst {catalyst_score:.2f}")
    if persistence.get("all_catalysts"):
        reasons.append(f"{len(persistence['all_catalysts'])} catalyst(s)")

    return True, " | ".join(reasons)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_tracker(
    today_watchlist: list[dict],
    today_l1: list[dict],
    today_date: str,
    regime: dict = None,
) -> dict:
    """
    FIX: accepts regime dict and passes regime string to is_watch_now().
    """
    log.info("▶ Running scan history tracker...")

    regime_str = "NEUTRAL"
    if regime:
        regime_str = regime.get("regime", "NEUTRAL")

    history    = load_history()
    past_scans = load_past_scans(LOOKBACK_DAYS)

    history = update_history(today_watchlist, today_date, history)
    save_history(history)

    enriched  = []
    watch_now = []

    l2_history_symbols = set(history.keys())
    for stock in today_l1:
        sym = stock["symbol"]
        if sym in l2_history_symbols:
            sym_hist = history[sym]
            stock["_prev_l2_appearances"] = len(sym_hist.get("appearances", []))
            stock["_first_seen_l2"]       = sym_hist.get("first_seen")

    for stock in today_watchlist:
        sym         = stock["symbol"]
        persistence = compute_persistence(sym, history, past_scans)
        watch, why  = is_watch_now(stock, persistence, regime=regime_str)

        stock["persistence"] = persistence
        enriched.append(stock)

        if watch:
            watch_now.append({
                "symbol":              sym,
                "price":               stock.get("price"),
                "market_cap_cr":       stock.get("market_cap_cr"),
                "composite_score":     stock.get("composite_score"),
                "price_stage":         stock.get("price_stage"),
                "streak":              persistence["streak"],
                "days_in_watchlist":   persistence["days_in_watchlist"],
                "score_trend":         persistence["score_trend"],
                "score_7d_delta":      persistence["score_7d_delta"],
                "first_seen":          persistence["first_seen"],
                "price_since_first_pct": persistence["price_since_first_pct"],
                "peak_score":          persistence["peak_score"],
                "piotroski":           stock.get("piotroski", {}).get("piotroski_score"),
                "fcf_yield":           stock.get("fcf", {}).get("fcf_yield_pct"),
                "roce":                (stock.get("screener") or {}).get("roce_pct"),
                "all_catalysts":       persistence["all_catalysts"],
                "catalyst_score":      persistence["latest_catalyst_score"],
                "watch_reason":        why,
                "group":               stock.get("group", {}).get("matched_group"),
            })

    watch_now.sort(key=lambda x: (-x["streak"], -x["composite_score"]))

    with open("results/persistence.json", "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    with open("results/watch_now.json", "w") as f:
        json.dump(watch_now, f, indent=2, default=str)

    total_tracked = len(history)
    log.info(
        f"  Tracker complete: {len(enriched)} stocks enriched, "
        f"{len(watch_now)} 'watch now' (regime={regime_str}), "
        f"{total_tracked} total in history"
    )

    return {
        "enriched_watchlist": enriched,
        "watch_now":          watch_now,
        "total_tracked":      total_tracked,
    }


# ─────────────────────────────────────────────────────────────
# REPORT SECTION
# ─────────────────────────────────────────────────────────────

def generate_persistence_section(tracker_result: dict, regime: dict = None) -> str:
    watch_now     = tracker_result.get("watch_now", [])
    total_tracked = tracker_result.get("total_tracked", 0)
    enriched      = tracker_result.get("enriched_watchlist", [])
    regime_str    = (regime or {}).get("regime", "NEUTRAL")

    sep = "═" * 72
    lines = [
        sep,
        "  SECTION D: WATCH NOW — PERSISTENCE TRACKER",
        f"  Regime: {regime_str} | Gate: streak ≥3, score not falling",
        "  BEAR/NEUTRAL: also requires Piotroski ≥7 OR catalyst detected",
        sep,
        f"  Total stocks tracked across all scans: {total_tracked}",
        "",
    ]

    if not watch_now:
        lines += [
            "  No stocks currently meet the watch-now criteria.",
            f"  In {regime_str} regime: need streak ≥3 AND (Piotroski ≥7 OR catalyst >0.10)",
            "  This is the correct behaviour — the list should be empty",
            "  rather than full of cheap stocks with no thesis.",
            "",
        ]
    else:
        for i, w in enumerate(watch_now, 1):
            trend_arrow = {"rising": "↑", "falling": "↓", "stable": "→", "new": "●"}.get(
                w.get("score_trend", ""), "?"
            )
            price_str = (
                f"+{w['price_since_first_pct']:.1f}% since first seen"
                if (w.get("price_since_first_pct") or 0) > 0
                else f"{w.get('price_since_first_pct', '?'):.1f}% since first seen"
                if w.get("price_since_first_pct") is not None else ""
            )

            lines += [
                f"  #{i:02d} {w['symbol']:<16} Streak: {w['streak']} scans "
                f"Score: {w['composite_score']:.3f} {trend_arrow}",
                f"  Price: ₹{w['price']:<8} MCap: ₹{w['market_cap_cr']:.0f} Cr "
                f"{w.get('price_stage', '')}",
                f"  First seen: {w['first_seen']} {price_str}",
                f"  Days in L2 last 30: {w['days_in_watchlist']} "
                f"Peak score: {w['peak_score']} "
                f"Delta 7d: {'+' if (w.get('score_7d_delta') or 0) > 0 else ''}"
                f"{w.get('score_7d_delta', '?')}",
                f"  Catalyst score: {w.get('catalyst_score', 0):.2f}",
            ]

            if w.get("piotroski"):
                lines.append(f"  Piotroski: {w['piotroski']}/9")
            if w.get("roce"):
                lines.append(f"  ROCE: {w['roce']:.1f}% (Screener)")
            if w.get("group"):
                lines.append(f"  Group: {w['group']}")
            if w.get("all_catalysts"):
                lines.append(f"  Catalysts seen: {' | '.join(w['all_catalysts'][:3])}")
            lines.append(f"  Watch reason: {w['watch_reason']}")
            lines.append("")

    # Current watchlist persistence context
    lines += [sep, "  CURRENT WATCHLIST — PERSISTENCE CONTEXT", sep, ""]
    sorted_enriched = sorted(
        enriched,
        key=lambda x: (
            -x.get("persistence", {}).get("streak", 0),
            -x.get("composite_score", 0),
        ),
    )

    for s in sorted_enriched[:15]:
        p      = s.get("persistence", {})
        streak = p.get("streak", 0)
        trend  = {"rising": "↑", "falling": "↓", "stable": "→", "new": "●"}.get(
            p.get("score_trend", "new"), "●"
        )
        first = p.get("first_seen", "today")
        delta = p.get("score_7d_delta")
        delta_str = (
            f"({'+' if (delta or 0) > 0 else ''}{delta:.3f})"
            if delta is not None else ""
        )
        cat_score = p.get("latest_catalyst_score", 0.0)
        pio       = s.get("piotroski", {}).get("piotroski_score", "?")
        new_badge = " [NEW]" if streak == 0 else ""

        lines.append(
            f"  {s['symbol']:<16} streak:{streak:>2} "
            f"score:{s.get('composite_score', 0):.3f} {trend}{delta_str} "
            f"P:{pio}/9 cat:{cat_score:.2f} "
            f"first:{first}{new_badge}"
        )

    lines += [
        "",
        sep,
        "  PERSISTENCE KEY:",
        "  streak = consecutive scans | ↑ rising ↓ falling → stable ● new",
        f"  Watch now ({regime_str}): streak≥3, not falling, P≥7 OR catalyst>0.10",
        sep,
    ]

    return "\n".join(lines)
