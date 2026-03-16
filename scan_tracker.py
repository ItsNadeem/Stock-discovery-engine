"""
scan_tracker.py — Scan History Tracker

The single highest-ROI improvement to the engine after backtesting.

Why persistence matters:
  A stock scoring 0.65 for 10 consecutive weeks is far more interesting
  than one scoring 0.72 once. Persistence means:
    - The thesis signals are structural, not a one-off data quirk
    - Multiple independent scans have validated the same signals
    - The stock has not already run (it would have exited the universe)

  VALUEPICK watched stocks for months before writing about them.
  Persistence tracking makes that behaviour systematic.

What this module does:
  On each run, it reads the last N days of saved watchlist JSON files,
  computes for every symbol currently in the watchlist:
    - days_in_watchlist:  how many of the last 30 scan days it appeared
    - streak:             current consecutive-day streak
    - score_trend:        is the composite score rising, falling, or flat?
    - score_7d_delta:     score change over last 7 appearances
    - first_seen:         date first appeared in watchlist
    - peak_score:         highest score ever recorded
    - price_change_pct:   price change since first appearance
    - catalyst_history:   list of unique catalysts seen across all scans

  It also flags "watch now" stocks — those with streak ≥ 3 AND
  rising score trend AND Piotroski ≥ 6 — as the highest-priority
  manual research targets.

Saves:
  results/history.json          → full history dict (all symbols ever seen)
  results/persistence.json      → today's enriched watchlist with history data
  results/watch_now.json        → "watch now" candidates only
"""

import json
import logging
import os
import glob
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

HISTORY_FILE    = "results/history.json"
LOOKBACK_DAYS   = 30      # how many past scans to analyse
STREAK_THRESHOLD = 3      # min streak to flag as "watch now"
SCORE_RISE_MIN   = 0.02   # score must have risen by this much to be "rising"


# ─────────────────────────────────────────────────────────────
# HISTORY STORE — persistent JSON file updated each run
# ─────────────────────────────────────────────────────────────

def load_history() -> dict:
    """Load the persistent history store. Returns {} if not found."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load history: {e}")
    return {}


def save_history(history: dict):
    """Persist the history store."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# SCAN FILE LOADER — reads past watchlist JSON files
# ─────────────────────────────────────────────────────────────

def load_past_scans(lookback_days: int = LOOKBACK_DAYS) -> list[tuple[str, list[dict]]]:
    """
    Load past watchlist JSON files from the results/ directory.
    Returns list of (date_str, watchlist) tuples, newest first.
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    files  = sorted(glob.glob("results/watchlist_*.json"), reverse=True)
    scans  = []

    for f in files:
        # Extract date from filename: watchlist_20260315.json → 20260315
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
    """
    Update the persistent history store with today's watchlist.
    Each symbol gets a running log of every scan it appeared in.
    """
    today_symbols = {s["symbol"] for s in today_watchlist}

    # Add today's data for each symbol in today's watchlist
    for stock in today_watchlist:
        sym = stock["symbol"]
        if sym not in history:
            history[sym] = {
                "first_seen":        today_date,
                "appearances":       [],
                "all_catalysts":     [],
            }

        sc_data = stock.get("screener") or {}
        history[sym]["appearances"].append({
            "date":             today_date,
            "price":            stock.get("price"),
            "score":            stock.get("composite_score"),
            "price_stage":      stock.get("price_stage"),
            "piotroski":        stock.get("piotroski", {}).get("piotroski_score"),
            "fcf_yield":        stock.get("fcf", {}).get("fcf_yield_pct"),
            "roce":             sc_data.get("roce_pct"),
            "catalyst_score":   stock.get("catalyst", {}).get("catalyst_score", 0),
            "catalysts":        stock.get("catalyst", {}).get("catalysts", []),
            "group":            stock.get("group", {}).get("matched_group"),
            "debt_reducing":    stock.get("debt", {}).get("debt_reducing", False),
            "pe_expanding":     stock.get("pe_trajectory", {}).get("pe_expanding", False),
        })

        # Accumulate unique catalysts seen across all scans
        new_cats = stock.get("catalyst", {}).get("catalysts", [])
        existing = set(history[sym]["all_catalysts"])
        for c in new_cats:
            if c not in existing:
                history[sym]["all_catalysts"].append(c)
                existing.add(c)

        # Keep appearances list bounded (last 90 entries max)
        history[sym]["appearances"] = history[sym]["appearances"][-90:]

    return history


def compute_persistence(
    symbol: str,
    history: dict,
    past_scans: list[tuple[str, list[dict]]],
    lookback_days: int = LOOKBACK_DAYS,
) -> dict:
    """
    Compute all persistence metrics for a single symbol.
    """
    sym_history = history.get(symbol, {})
    appearances = sym_history.get("appearances", [])

    if not appearances:
        return {
            "days_in_watchlist": 0, "streak": 0,
            "score_trend": "new", "score_7d_delta": None,
            "first_seen": None, "peak_score": None,
            "price_since_first_pct": None, "all_catalysts": [],
        }

    # Days in watchlist (last N scan days)
    cutoff_dt = datetime.utcnow() - timedelta(days=lookback_days)
    recent    = [
        a for a in appearances
        if datetime.strptime(a["date"], "%Y%m%d") >= cutoff_dt
    ]
    days_in_watchlist = len(recent)

    # Current streak — consecutive appearances ending today
    sorted_dates = sorted([a["date"] for a in appearances], reverse=True)
    streak = 0
    # Get set of all scan dates (dates where any watchlist was saved)
    all_scan_dates = sorted(
        {date for date, _ in past_scans},
        reverse=True,
    )
    sym_dates = set(a["date"] for a in appearances)

    # Walk back from most recent scan, count consecutive appearances
    for scan_date in all_scan_dates:
        if scan_date in sym_dates:
            streak += 1
        else:
            break   # streak broken

    # Score trend
    scores = [a["score"] for a in appearances if a.get("score") is not None]
    score_7d_delta = None
    score_trend    = "stable"

    if len(scores) >= 2:
        # Compare average of last 3 scores vs prior 3
        recent_avg = sum(scores[-3:]) / len(scores[-3:])
        prior_avg  = sum(scores[-6:-3]) / len(scores[-6:-3]) if len(scores) >= 6 else scores[0]
        delta      = recent_avg - prior_avg
        score_7d_delta = round(delta, 3)

        if delta > SCORE_RISE_MIN:
            score_trend = "rising"
        elif delta < -SCORE_RISE_MIN:
            score_trend = "falling"

    # Peak score
    peak_score = round(max(scores), 3) if scores else None

    # Price change since first seen
    first_price   = appearances[0].get("price")
    current_price = appearances[-1].get("price")
    price_change  = None
    if first_price and current_price and first_price > 0:
        price_change = round((current_price - first_price) / first_price * 100, 1)

    return {
        "days_in_watchlist":    days_in_watchlist,
        "streak":               streak,
        "score_trend":          score_trend,
        "score_7d_delta":       score_7d_delta,
        "first_seen":           sym_history.get("first_seen"),
        "peak_score":           peak_score,
        "price_since_first_pct": price_change,
        "all_catalysts":        sym_history.get("all_catalysts", []),
    }


def is_watch_now(stock: dict, persistence: dict) -> tuple[bool, str]:
    """
    Flag a stock as "watch now" — highest priority for manual research.

    Criteria (all must be met):
      - Streak ≥ 3 (appeared in at least 3 consecutive scans)
      - Score trend is rising or stable (not falling)
      - Piotroski ≥ 6 (if available)
      - Not already near 52W high (not late)

    Returns (is_watch_now, reason_string)
    """
    streak      = persistence.get("streak", 0)
    score_trend = persistence.get("score_trend", "")
    piotroski   = stock.get("piotroski", {}).get("piotroski_score")
    stage       = stock.get("price_stage", "")

    if streak < STREAK_THRESHOLD:
        return False, f"Streak only {streak} (need {STREAK_THRESHOLD}+)"

    if score_trend == "falling":
        return False, "Score declining"

    if "NEAR 52W HIGH" in stage:
        return False, "Near 52W high — too late"

    if piotroski is not None and piotroski < 4:
        return False, f"Piotroski {piotroski}/9 too low"

    # Build reason string
    reasons = [f"Streak {streak} scans"]
    if score_trend == "rising":
        reasons.append(f"score rising (+{persistence.get('score_7d_delta', '?')})")
    if piotroski and piotroski >= 7:
        reasons.append(f"Piotroski {piotroski}/9")
    if persistence.get("all_catalysts"):
        reasons.append(f"{len(persistence['all_catalysts'])} catalyst(s)")

    return True, " | ".join(reasons)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — called from run_all.py after each scan
# ─────────────────────────────────────────────────────────────

def run_tracker(
    today_watchlist: list[dict],
    today_l1: list[dict],
    today_date: str,
) -> dict:
    """
    Main function called after each nightly scan.

    1. Load history
    2. Update with today's results
    3. Compute persistence metrics for today's watchlist
    4. Identify "watch now" candidates
    5. Save updated history + enriched outputs

    Returns enriched watchlist with persistence data attached.
    """
    log.info("▶ Running scan history tracker...")

    history    = load_history()
    past_scans = load_past_scans(LOOKBACK_DAYS)

    # Update history with today
    history = update_history(today_watchlist, today_date, history)
    save_history(history)

    # Enrich each stock with persistence data
    enriched   = []
    watch_now  = []

    # Also track L1 breakouts — did any appear in L2 watchlist previously?
    l2_history_symbols = set(history.keys())
    for stock in today_l1:
        sym = stock["symbol"]
        if sym in l2_history_symbols:
            sym_hist = history[sym]
            stock["_prev_l2_appearances"] = len(sym_hist.get("appearances", []))
            stock["_first_seen_l2"] = sym_hist.get("first_seen")

    for stock in today_watchlist:
        sym         = stock["symbol"]
        persistence = compute_persistence(sym, history, past_scans)
        watch, why  = is_watch_now(stock, persistence)

        stock["persistence"] = persistence
        enriched.append(stock)

        if watch:
            watch_now.append({
                "symbol":             sym,
                "price":              stock.get("price"),
                "market_cap_cr":      stock.get("market_cap_cr"),
                "composite_score":    stock.get("composite_score"),
                "price_stage":        stock.get("price_stage"),
                "streak":             persistence["streak"],
                "days_in_watchlist":  persistence["days_in_watchlist"],
                "score_trend":        persistence["score_trend"],
                "score_7d_delta":     persistence["score_7d_delta"],
                "first_seen":         persistence["first_seen"],
                "price_since_first_pct": persistence["price_since_first_pct"],
                "peak_score":         persistence["peak_score"],
                "piotroski":          stock.get("piotroski", {}).get("piotroski_score"),
                "fcf_yield":          stock.get("fcf", {}).get("fcf_yield_pct"),
                "roce":               (stock.get("screener") or {}).get("roce_pct"),
                "all_catalysts":      persistence["all_catalysts"],
                "watch_reason":       why,
                "group":              stock.get("group", {}).get("matched_group"),
            })

    watch_now.sort(key=lambda x: (-x["streak"], -x["composite_score"]))

    # Save enriched outputs
    with open("results/persistence.json", "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    with open("results/watch_now.json", "w") as f:
        json.dump(watch_now, f, indent=2, default=str)

    total_tracked = len(history)
    log.info(
        f"  Tracker complete: {len(enriched)} stocks enriched, "
        f"{len(watch_now)} 'watch now', "
        f"{total_tracked} total in history"
    )

    return {
        "enriched_watchlist": enriched,
        "watch_now":          watch_now,
        "total_tracked":      total_tracked,
    }


# ─────────────────────────────────────────────────────────────
# REPORT SECTION — injected into daily_report.txt
# ─────────────────────────────────────────────────────────────

def generate_persistence_section(tracker_result: dict) -> str:
    """
    Generate the persistence / watch-now section for the daily report.
    This is the most actionable section — stocks with long streaks and
    rising scores are the highest-priority manual research targets.
    """
    watch_now       = tracker_result.get("watch_now", [])
    total_tracked   = tracker_result.get("total_tracked", 0)
    enriched        = tracker_result.get("enriched_watchlist", [])
    sep = "═" * 72

    lines = [
        sep,
        "  SECTION D: WATCH NOW — PERSISTENCE TRACKER",
        "  Stocks in watchlist for 3+ consecutive scans with rising/stable score",
        "  These are the highest-priority names for manual research",
        sep,
        f"  Total stocks tracked across all scans: {total_tracked}",
        "",
    ]

    if not watch_now:
        lines += [
            "  No stocks currently meet the watch-now criteria.",
            "  (Need: streak ≥ 3 consecutive scans, score not falling, not near 52W high)",
            "",
        ]
    else:
        for i, w in enumerate(watch_now, 1):
            trend_arrow = {"rising": "↑", "falling": "↓", "stable": "→", "new": "●"}.get(
                w.get("score_trend", ""), "?"
            )
            price_str = f"+{w['price_since_first_pct']:.1f}% since first seen" if w.get("price_since_first_pct") else ""
            lines += [
                f"  #{i:02d}  {w['symbol']:<16}  Streak: {w['streak']} scans  "
                f"Score: {w['composite_score']:.3f} {trend_arrow}",
                f"       Price: ₹{w['price']:<8}  MCap: ₹{w['market_cap_cr']:.0f} Cr  "
                f"{w.get('price_stage', '')}",
                f"       First seen: {w['first_seen']}  {price_str}",
                f"       Days in L2 last 30: {w['days_in_watchlist']}  "
                f"Peak score: {w['peak_score']}  "
                f"Delta 7d: {'+' if (w.get('score_7d_delta') or 0) > 0 else ''}{w.get('score_7d_delta', '?')}",
            ]
            if w.get("piotroski"): lines.append(f"       Piotroski: {w['piotroski']}/9", )
            if w.get("roce"):      lines.append(f"       ROCE: {w['roce']:.1f}% (Screener)")
            if w.get("group"):     lines.append(f"       Group: {w['group']}")
            if w.get("all_catalysts"):
                lines.append(f"       Catalysts seen: {' | '.join(w['all_catalysts'][:3])}")
            lines.append(f"       Watch reason: {w['watch_reason']}")
            lines.append("")

    # Also show persistence data for current watchlist
    lines += [
        sep,
        "  CURRENT WATCHLIST — PERSISTENCE CONTEXT",
        sep,
        "",
    ]
    # Sort by streak descending
    sorted_enriched = sorted(
        enriched,
        key=lambda x: (
            -x.get("persistence", {}).get("streak", 0),
            -x.get("composite_score", 0),
        ),
    )
    for s in sorted_enriched[:15]:
        p     = s.get("persistence", {})
        streak = p.get("streak", 0)
        trend  = {"rising": "↑", "falling": "↓", "stable": "→", "new": "●"}.get(
            p.get("score_trend", "new"), "●"
        )
        first = p.get("first_seen", "today")
        delta = p.get("score_7d_delta")
        delta_str = f"({'+' if (delta or 0) > 0 else ''}{delta:.3f})" if delta is not None else ""
        new_badge = " [NEW]" if streak == 0 else ""
        lines.append(
            f"  {s['symbol']:<16}  streak:{streak:>2}  "
            f"score:{s.get('composite_score', 0):.3f} {trend}{delta_str}  "
            f"first:{first}{new_badge}"
        )

    lines += [
        "",
        sep,
        "  PERSISTENCE KEY:",
        "  streak = consecutive scans appeared  |  ↑ rising  ↓ falling  → stable  ● new",
        "  Watch now = streak ≥ 3, score not falling, Piotroski ≥ 4, not near 52W high",
        sep,
    ]
    return "\n".join(lines)
