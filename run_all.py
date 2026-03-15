"""
run_all.py — Master Orchestrator
Runs Layer 1 (Swing Breakout) + Layer 2 (Pre-Breakout) in sequence.
Produces:
  - results/latest.json            → Layer 1 top 20
  - results/watchlist.json         → Layer 2 top 15
  - results/conviction_list.json   → Stocks appearing in BOTH layers (highest conviction)
  - results/daily_report.txt       → Combined human-readable report
"""

import json
import logging
import os
from datetime import datetime

from universe import fetch_nse_symbols
from engine import run_engine
from pre_breakout_scanner import (
    run_pre_breakout_scanner,
    generate_pre_breakout_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("results/scanner.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

os.makedirs("results", exist_ok=True)


def find_conviction_plays(layer1: list[dict], layer2: list[dict]) -> list[dict]:
    """
    Stocks present in BOTH scanners = highest conviction.
    Layer 1 = confirmed breakout. Layer 2 = early thesis.
    Overlap = the sweet spot the blog targets.
    """
    l1_symbols = {s["symbol"] for s in layer1}
    conviction = []
    for s in layer2:
        if s["symbol"] in l1_symbols:
            # Merge both signal sets
            l1_data = next(x for x in layer1 if x["symbol"] == s["symbol"])
            conviction.append({
                "symbol": s["symbol"],
                "price": s["price"],
                "market_cap_cr": s["market_cap_cr"],
                "price_stage": s["price_stage"],
                "layer1_score": l1_data["composite_score"],
                "layer2_score": s["composite_score"],
                "combined_score": round(
                    l1_data["composite_score"] * 0.5 + s["composite_score"] * 0.5, 4
                ),
                "technical": {
                    "rsi": l1_data.get("rsi"),
                    "adx": l1_data.get("adx"),
                    "week52_high": l1_data.get("week52_high"),
                    "breakout_level": l1_data.get("breakout_level"),
                    "vol_surge": l1_data.get("vol_surge"),
                    "ema_bullish": l1_data.get("ema_bullish"),
                    "price_chg_3m_pct": l1_data.get("price_chg_3m_pct"),
                },
                "fundamental": l1_data.get("fundamentals", {}),
                "pre_breakout": {
                    "catalysts": s["catalyst"]["catalysts"],
                    "promoter_pct": s["shareholding"]["insider_pct"],
                    "pb_ratio": s["valuation"]["pb_ratio"],
                    "net_cash_cr": s["valuation"]["net_cash_cr"],
                    "growth_accelerating": s["growth"]["growth_accelerating"],
                    "revenue_growth_pct": s["growth"]["revenue_growth_pct"],
                },
                "scanned_at": s["scanned_at"],
            })

    conviction.sort(key=lambda x: x["combined_score"], reverse=True)
    return conviction


def generate_conviction_report(conviction: list[dict], l1: list[dict], l2: list[dict]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═" * 72
    lines = [
        sep,
        "  MULTIBAGGER DISCOVERY ENGINE — DAILY COMBINED REPORT",
        f"  {now}",
        sep,
        "",
    ]

    # ── Section A: Conviction plays ──
    lines += [
        "  ★  SECTION A: CONVICTION LIST (In Both Layer 1 + Layer 2)",
        "     These have confirmed technical breakout AND early-stage thesis",
        sep,
        "",
    ]

    if conviction:
        for i, s in enumerate(conviction, 1):
            tech = s["technical"]
            pb = s["pre_breakout"]
            lines += [
                f"★ #{i}  {s['symbol']:<16}  Combined Score: {s['combined_score']:.3f}",
                f"     Price: ₹{s['price']:<8}  MCap: ₹{s['market_cap_cr']:.0f} Cr",
                f"     Stage: {s['price_stage']}",
                f"     L1 (Technical): {s['layer1_score']:.3f}  |  L2 (Pre-Breakout): {s['layer2_score']:.3f}",
                f"     RSI: {tech['rsi']}   ADX: {tech['adx']}   3M Return: {tech['price_chg_3m_pct']:+.1f}%",
                f"     52W High: ₹{tech['week52_high']}   Vol Surge: {'✓' if tech['vol_surge'] else '✗'}",
                f"     Promoter: {pb['promoter_pct']:.1f}%   P/B: {pb['pb_ratio']:.2f}   Rev Growth: {pb['revenue_growth_pct']:.1f}%",
            ]
            if pb["catalysts"]:
                lines.append(f"     ▶  {' | '.join(pb['catalysts'][:3])}")
            lines.append("")
    else:
        lines += ["  No overlap today between Layer 1 and Layer 2.", ""]

    # ── Section B: Layer 1 Top 10 ──
    lines += [
        sep,
        "  SECTION B: TOP 10 TECHNICAL BREAKOUTS (Layer 1)",
        sep,
        "",
    ]
    for i, s in enumerate(l1[:10], 1):
        fd = s.get("fundamentals", {})
        lines.append(
            f"  #{i:02d}  {s['symbol']:<16}  ₹{s['last_close']:<8}  "
            f"Score: {s['composite_score']:.3f}  RSI: {s['rsi']}  "
            f"3M: {s['price_chg_3m_pct']:+.1f}%"
        )
    lines.append("")

    # ── Section C: Layer 2 Top 10 ──
    lines += [
        sep,
        "  SECTION C: TOP 10 PRE-BREAKOUT WATCHLIST (Layer 2)",
        sep,
        "",
    ]
    for i, s in enumerate(l2[:10], 1):
        cat_str = s["catalyst"]["catalysts"][0] if s["catalyst"]["catalysts"] else "–"
        lines.append(
            f"  #{i:02d}  {s['symbol']:<16}  ₹{s['price']:<8}  "
            f"Score: {s['composite_score']:.3f}  {s['price_stage']}  {cat_str}"
        )
    lines.append("")

    lines += [
        sep,
        "  HOW TO USE:",
        "  Section A → Highest conviction. Both technicals AND thesis confirmed.",
        "  Section B → Swing trade candidates. Enter on breakout, trail stop-loss.",
        "  Section C → Research these NOW. Wait for breakout before entry.",
        sep,
        "  ⚠  Educational use only. Not financial advice.",
        sep,
    ]
    return "\n".join(lines)


def run_all():
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   MULTIBAGGER DISCOVERY ENGINE — FULL PIPELINE       ║")
    log.info("║   Layer 1: Swing Breakout + Fundamentals             ║")
    log.info("║   Layer 2: Pre-Breakout (value-picks methodology)    ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    date_str = datetime.utcnow().strftime("%Y%m%d")

    # ── Get universe ──
    symbols = fetch_nse_symbols()

    # ── Layer 1: Swing Breakout Engine ──
    log.info("\n▶ Running Layer 1: Swing Breakout Scanner...")
    layer1_results = run_engine()
    if not layer1_results:
        layer1_results = []

    # ── Layer 2: Pre-Breakout Scanner ──
    log.info("\n▶ Running Layer 2: Pre-Breakout Scanner...")
    layer2_results = run_pre_breakout_scanner(symbols)

    # ── Find Conviction Plays (overlap) ──
    log.info("\n▶ Finding conviction plays (overlap between L1 + L2)...")
    conviction = find_conviction_plays(layer1_results, layer2_results)
    log.info(f"  Conviction plays: {len(conviction)}")

    # ── Generate reports ──
    combined_report = generate_conviction_report(conviction, layer1_results, layer2_results)
    pre_breakout_report = generate_pre_breakout_report(layer2_results)

    # Print to Actions log
    print("\n" + combined_report)

    # ── Save all outputs ──
    # Layer 1
    with open(f"results/scan_{date_str}.json", "w") as f:
        json.dump(layer1_results, f, indent=2, default=str)
    with open("results/latest.json", "w") as f:
        json.dump(layer1_results, f, indent=2, default=str)

    # Layer 2
    with open(f"results/watchlist_{date_str}.json", "w") as f:
        json.dump(layer2_results, f, indent=2, default=str)
    with open("results/watchlist.json", "w") as f:
        json.dump(layer2_results, f, indent=2, default=str)

    # Conviction
    with open(f"results/conviction_{date_str}.json", "w") as f:
        json.dump(conviction, f, indent=2, default=str)
    with open("results/conviction.json", "w") as f:
        json.dump(conviction, f, indent=2, default=str)

    # Reports
    with open(f"results/report_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(combined_report + "\n\n" + pre_breakout_report)
    with open("results/daily_report.txt", "w", encoding="utf-8") as f:
        f.write(combined_report + "\n\n" + pre_breakout_report)

    log.info("\n✅ All outputs saved to results/")
    log.info(f"   Layer 1 candidates:   {len(layer1_results)}")
    log.info(f"   Layer 2 watchlist:    {len(layer2_results)}")
    log.info(f"   Conviction plays:     {len(conviction)}")

    return {
        "layer1": layer1_results,
        "layer2": layer2_results,
        "conviction": conviction,
    }


if __name__ == "__main__":
    run_all()
