"""
scan_tracker.py — Scan History Tracker v2

FIX (2026-03-30): v5 pre_breakout_scanner stores group as a plain string/None,
not a dict. All stock.get("group", {}).get("matched_group") calls replaced with
a safe helper that handles both old dict format and new string format.

FIX: regime-aware watch_now gate (from prior session, preserved).
"""

import json
import logging
import os
import glob
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

HISTORY_FILE     = "results/history.json"
LOOKBACK_DAYS    = 30
STREAK_THRESHOLD = 3
SCORE_RISE_MIN   = 0.02


def _group_name(stock: dict) -> Optional[str]:
    """
    Safe group-name extractor that handles both formats:
      v4 format: stock["group"] = {"matched_group": "Murugappa", ...}
      v5 format: stock["group"] = "Murugappa"  OR  None
    """
    g = stock.get("group")
    if g is None:
        return None
    if isinstance(g, dict):
        return g.get("matched_group")
    if isinstance(g, str):
        return g or None
    return None


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


def update_history(today_watchlist: list[dict], today_date: str, history: dict) -> dict:
    for stock in today_watchlist:
        sym = stock["symbol"]
        if sym not in history:
            history[sym] = {"first_seen": today_date, "appearances": [], "all_catalysts": []}

        sc_data = stock.get("screener") or {}

        # DNA-aware catalyst extraction (v5 stores catalysts inside dna dict)
        dna = stock.get("dna") or {}
        cat_data = stock.get("catalyst") or {}
        catalyst_score = (
            dna.get("catalyst_score") or
            cat_data.get("catalyst_score", 0)
        )
        catalysts = (
            dna.get("dna_flags") or
            cat_data.get("catalysts", [])
        )

        history[sym]["appearances"].append({
            "date":           today_date,
            "price":          stock.get("price"),
            "score":          stock.get("composite_score"),
            "price_stage":    stock.get("price_stage"),
            "piotroski":      stock.get("piotroski_score") or
                              (stock.get("piotroski") or {}).get("piotroski_score"),
            "fcf_yield":      (stock.get("fcf") or {}).get("fcf_yield_pct"),
            "roce":           sc_data.get("roce_pct"),
            "catalyst_score": catalyst_score,
            "catalysts":      catalysts,
            "group":          _group_name(stock),
            "debt_reducing":  (stock.get("debt") or {}).get("debt_reducing", False),
            "pe_expanding":   (stock.get("pe_trajectory") or {}).get("pe_expanding", False),
            "dna_score":      dna.get("dna_score"),
            "dna_grade":      dna.get("dna_grade"),
        })

        new_cats = catalysts
        existing = set(history[sym]["all_catalysts"])
        for c in new_cats:
            if c not in existing:
                history[sym]["all_catalysts"].append(c)
                existing.add(c)

        history[sym]["appearances"] = history[sym]["appearances"][-90:]

    return history


def compute_persistence(symbol: str, history: dict,
                        past_scans: list[tuple[str, list[dict]]],
                        lookback_days: int = LOOKBACK_DAYS) -> dict:
    sym_history = history.get(symbol, {})
    appearances = sym_history.get("appearances", [])

    if not appearances:
        return {
            "days_in_watchlist": 0, "streak": 0, "score_trend": "new",
            "score_7d_delta": None, "first_seen": None, "peak_score": None,
            "price_since_first_pct": None, "all_catalysts": [],
            "latest_catalyst_score": 0.0, "latest_dna_score": None,
        }

    cutoff_dt = datetime.utcnow() - timedelta(days=lookback_days)
    recent    = [a for a in appearances
                 if datetime.strptime(a["date"], "%Y%m%d") >= cutoff_dt]
    days_in_watchlist = len(recent)

    all_scan_dates = sorted({date for date, _ in past_scans}, reverse=True)
    sym_dates      = set(a["date"] for a in appearances)
    streak = 0
    for scan_date in all_scan_dates:
        if scan_date in sym_dates:
            streak += 1
        else:
            break

    scores = [a["score"] for a in appearances if a.get("score") is not None]
    score_7d_delta = None
    score_trend    = "stable"
    if len(scores) >= 2:
        recent_avg = sum(scores[-3:]) / len(scores[-3:])
        prior_avg  = sum(scores[-6:-3]) / len(scores[-6:-3]) if len(scores) >= 6 else scores[0]
        delta      = recent_avg - prior_avg
        score_7d_delta = round(delta, 3)
        if delta > SCORE_RISE_MIN:    score_trend = "rising"
        elif delta < -SCORE_RISE_MIN: score_trend = "falling"

    peak_score   = round(max(scores), 3) if scores else None
    first_price  = appearances[0].get("price")
    current_price= appearances[-1].get("price")
    price_change = None
    if first_price and current_price and first_price > 0:
        price_change = round((current_price - first_price) / first_price * 100, 1)

    latest_catalyst_score = appearances[-1].get("catalyst_score", 0.0) or 0.0
    latest_dna_score      = appearances[-1].get("dna_score")

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
        "latest_dna_score":      latest_dna_score,
    }


def is_watch_now(stock: dict, persistence: dict,
                 regime: str = "NEUTRAL") -> tuple[bool, str]:
    """
    Regime-aware watch_now gate.
    BEAR/NEUTRAL: requires Piotroski ≥7 OR DNA grade A OR catalyst_score >0.10
    """
    streak         = persistence.get("streak", 0)
    score_trend    = persistence.get("score_trend", "")
    stage          = stock.get("price_stage", "")
    catalyst_score = persistence.get("latest_catalyst_score", 0.0)
    price_chg      = persistence.get("price_since_first_pct")
    dna_grade      = (stock.get("dna") or {}).get("dna_grade", "")

    # Piotroski — handle both v4 (dict) and v5 (flat) formats
    pio = stock.get("piotroski_score")
    if pio is None:
        pio = (stock.get("piotroski") or {}).get("piotroski_score")

    if streak < STREAK_THRESHOLD:
        return False, f"Streak only {streak} (need {STREAK_THRESHOLD}+)"
    if score_trend == "falling":
        return False, "Score declining"
    if "NEAR 52W HIGH" in stage:
        return False, "Near 52W high — too late"
    if pio is not None and pio < 4:
        return False, f"Piotroski {pio}/9 too low"

    if regime in ("BEAR", "NEUTRAL"):
        has_quality  = pio is not None and pio >= 7
        has_dna_a    = "A" in str(dna_grade)
        has_catalyst = catalyst_score > 0.10
        falling_knife = price_chg is not None and price_chg < -15 and not has_catalyst

        if falling_knife:
            return False, f"Down {price_chg:.1f}% with no catalyst — falling knife"
        if not has_quality and not has_dna_a and not has_catalyst:
            return False, (
                f"{regime} regime: need Piotroski≥7 OR DNA Grade A OR catalyst>0.10 "
                f"(got P={pio}/9, DNA={dna_grade[:1] if dna_grade else '?'}, cat={catalyst_score:.2f})"
            )

    reasons = [f"Streak {streak} scans"]
    if score_trend == "rising":
        reasons.append(f"score rising (+{persistence.get('score_7d_delta','?')})")
    if pio and pio >= 7:
        reasons.append(f"Piotroski {pio}/9")
    if "A" in str(dna_grade):
        reasons.append("DNA Grade A")
    if catalyst_score > 0.10:
        reasons.append(f"catalyst {catalyst_score:.2f}")
    if persistence.get("all_catalysts"):
        reasons.append(f"{len(persistence['all_catalysts'])} catalyst(s)")
    return True, " | ".join(reasons)


def run_tracker(today_watchlist: list[dict], today_l1: list[dict],
                today_date: str, regime: dict = None) -> dict:
    log.info("▶ Running scan history tracker...")
    regime_str = (regime or {}).get("regime", "NEUTRAL")

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
            h = history[sym]
            stock["_prev_l2_appearances"] = len(h.get("appearances", []))
            stock["_first_seen_l2"]       = h.get("first_seen")

    for stock in today_watchlist:
        sym         = stock["symbol"]
        persistence = compute_persistence(sym, history, past_scans)
        watch, why  = is_watch_now(stock, persistence, regime=regime_str)

        stock["persistence"] = persistence
        enriched.append(stock)

        if watch:
            dna    = stock.get("dna") or {}
            pio    = stock.get("piotroski_score") or (stock.get("piotroski") or {}).get("piotroski_score")
            watch_now.append({
                "symbol":                sym,
                "price":                 stock.get("price"),
                "market_cap_cr":         stock.get("market_cap_cr"),
                "composite_score":       stock.get("composite_score"),
                "price_stage":           stock.get("price_stage"),
                "streak":                persistence["streak"],
                "days_in_watchlist":     persistence["days_in_watchlist"],
                "score_trend":           persistence["score_trend"],
                "score_7d_delta":        persistence["score_7d_delta"],
                "first_seen":            persistence["first_seen"],
                "price_since_first_pct": persistence["price_since_first_pct"],
                "peak_score":            persistence["peak_score"],
                "piotroski":             pio,
                "dna_score":             dna.get("dna_score"),
                "dna_grade":             dna.get("dna_grade"),
                "fcf_yield":             (stock.get("fcf") or {}).get("fcf_yield_pct"),
                "roce":                  (stock.get("screener") or {}).get("roce_pct"),
                "all_catalysts":         persistence["all_catalysts"],
                "catalyst_score":        persistence["latest_catalyst_score"],
                "watch_reason":          why,
                "group":                 _group_name(stock),
            })

    watch_now.sort(key=lambda x: (-x["streak"], -x["composite_score"]))

    with open("results/persistence.json", "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    with open("results/watch_now.json", "w") as f:
        json.dump(watch_now, f, indent=2, default=str)

    log.info(
        f"  Tracker: {len(enriched)} enriched, "
        f"{len(watch_now)} watch_now (regime={regime_str}), "
        f"{len(history)} total in history"
    )
    return {"enriched_watchlist": enriched, "watch_now": watch_now,
            "total_tracked": len(history)}


def generate_persistence_section(tracker_result: dict, regime: dict = None) -> str:
    watch_now     = tracker_result.get("watch_now", [])
    total_tracked = tracker_result.get("total_tracked", 0)
    enriched      = tracker_result.get("enriched_watchlist", [])
    regime_str    = (regime or {}).get("regime", "NEUTRAL")

    sep = "═" * 72
    lines = [
        sep,
        "  SECTION D: WATCH NOW — PERSISTENCE TRACKER",
        f"  Regime: {regime_str} | Gate: streak≥3, not falling",
        "  BEAR/NEUTRAL: also requires Piotroski≥7 OR DNA Grade A OR catalyst>0.10",
        sep,
        f"  Total stocks tracked: {total_tracked}",
        "",
    ]

    if not watch_now:
        lines += [
            "  No stocks meet watch-now criteria.",
            f"  In {regime_str}: need streak≥3 AND (P≥7 OR DNA-A OR catalyst>0.10)",
            "",
        ]
    else:
        for i, w in enumerate(watch_now, 1):
            arrow = {"rising":"↑","falling":"↓","stable":"→","new":"●"}.get(
                w.get("score_trend",""), "?")
            p_str = f"+{w['price_since_first_pct']:.1f}%" if (w.get("price_since_first_pct") or 0) > 0 \
                    else (f"{w.get('price_since_first_pct','?'):.1f}%"
                          if w.get("price_since_first_pct") is not None else "")
            lines += [
                f"  #{i:02d} {w['symbol']:<16} Streak:{w['streak']} "
                f"Score:{w['composite_score']:.3f} {arrow}",
                f"  Price:₹{w['price']}  MCap:₹{w.get('market_cap_cr',0):.0f}Cr  {w.get('price_stage','')}",
                f"  First:{w['first_seen']} {p_str}",
            ]
            if w.get("dna_score"):
                lines.append(f"  DNA:{w['dna_score']:.3f} [{(w.get('dna_grade') or '')[:1]}]  "
                             f"cat:{w.get('catalyst_score',0):.2f}")
            if w.get("piotroski"): lines.append(f"  Piotroski:{w['piotroski']}/9")
            if w.get("roce"):      lines.append(f"  ROCE:{w['roce']:.1f}%")
            if w.get("group"):     lines.append(f"  Group:{w['group']}")
            if w.get("all_catalysts"):
                lines.append(f"  Catalysts: {' | '.join(w['all_catalysts'][:2])}")
            lines.append(f"  Watch reason: {w['watch_reason']}")
            lines.append("")

    # Current watchlist context
    lines += [sep, "  CURRENT WATCHLIST — PERSISTENCE CONTEXT", sep, ""]
    sorted_enriched = sorted(enriched,
                             key=lambda x: (-x.get("persistence",{}).get("streak",0),
                                           -x.get("composite_score",0)))
    for s in sorted_enriched[:15]:
        p     = s.get("persistence", {})
        streak= p.get("streak", 0)
        arrow = {"rising":"↑","falling":"↓","stable":"→","new":"●"}.get(
            p.get("score_trend","new"), "●")
        dna   = s.get("dna") or {}
        pio   = s.get("piotroski_score") or (s.get("piotroski") or {}).get("piotroski_score", "?")
        cat   = p.get("latest_catalyst_score", 0.0)
        delta = p.get("score_7d_delta")
        d_str = (f"({'+' if (delta or 0)>0 else ''}{delta:.3f})"
                 if delta is not None else "")
        badge = " [NEW]" if streak == 0 else ""
        lines.append(
            f"  {s['symbol']:<16} streak:{streak:>2} "
            f"score:{s.get('composite_score',0):.3f} {arrow}{d_str} "
            f"DNA:{dna.get('dna_score',0):.3f} P:{pio}/9 cat:{cat:.2f}"
            f"{badge}"
        )

    lines += ["", sep,
              "  KEY: streak=consecutive scans | ↑↓→● trend | DNA Grade A = multibagger pattern",
              sep]
    return "\n".join(lines)
