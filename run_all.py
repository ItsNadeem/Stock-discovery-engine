"""
run_all.py — Master Orchestrator v3
Runs Layer 1 (Swing Breakout) + Layer 2 (Pre-Breakout) in sequence.
NEW in v3: Market regime classifier runs first and banners every report.

Outputs:
  results/latest.json        → Layer 1 top 20
  results/watchlist.json     → Layer 2 top 15
  results/conviction.json    → Stocks in BOTH layers (highest conviction)
  results/daily_report.txt   → Combined human-readable report
"""

import json
import logging
import os
from datetime import datetime

from universe import fetch_nse_symbols
from engine import run_engine, get_market_regime, generate_report as generate_l1_report
from pre_breakout_scanner import (
    run_pre_breakout_scanner,
    generate_pre_breakout_report,
)
from scan_tracker import run_tracker, generate_persistence_section

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
    """Stocks in BOTH layers = highest conviction."""
    l1_symbols = {s["symbol"] for s in layer1}
    conviction = []
    for s in layer2:
        if s["symbol"] in l1_symbols:
            l1_data = next(x for x in layer1 if x["symbol"] == s["symbol"])
            conviction.append({
                "symbol":         s["symbol"],
                "price":          s["price"],
                "market_cap_cr":  s["market_cap_cr"],
                "price_stage":    s["price_stage"],
                "layer1_score":   l1_data["composite_score"],
                "layer2_score":   s["composite_score"],
                "combined_score": round(
                    l1_data["composite_score"] * 0.5 + s["composite_score"] * 0.5, 4
                ),
                "technical": {
                    "rsi":              l1_data.get("rsi"),
                    "adx":              l1_data.get("adx"),
                    "week52_high":      l1_data.get("week52_high"),
                    "breakout_level":   l1_data.get("breakout_level"),
                    "vol_surge":        l1_data.get("vol_surge"),
                    "ema_bullish":      l1_data.get("ema_bullish"),
                    "price_chg_3m_pct": l1_data.get("price_chg_3m_pct"),
                    "obv_bullish":      l1_data.get("obv_bullish"),
                    "rs_days":          l1_data.get("rs_days"),
                    "has_tight_base":   l1_data.get("has_tight_base"),
                },
                "fundamental": l1_data.get("fundamentals", {}),
                "pre_breakout": {
                    "catalysts":          s["catalyst"]["catalysts"],
                    "promoter_pct":       s["shareholding"]["insider_pct"],
                    "pb_ratio":           s["valuation"]["pb_ratio"],
                    "net_cash_cr":        s["valuation"]["net_cash_cr"],
                    "growth_accelerating": s["growth"]["growth_accelerating"],
                    "revenue_growth_pct": s["growth"]["revenue_growth_pct"],
                    "piotroski_score":    s.get("piotroski", {}).get("piotroski_score"),
                    "fcf_yield_pct":      s.get("fcf", {}).get("fcf_yield_pct"),
                    "pe_expanding":       s.get("pe_trajectory", {}).get("pe_expanding"),
                },
                "scanned_at": s["scanned_at"],
            })

    conviction.sort(key=lambda x: x["combined_score"], reverse=True)
    return conviction


def regime_banner(regime: dict) -> list[str]:
    """Format a market regime banner for the top of the report."""
    r     = regime.get("regime", "UNKNOWN")
    emoji = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴", "UNKNOWN": "⚪"}.get(r, "⚪")
    sc    = regime.get("smallcap100") or {}
    n50   = regime.get("nifty50") or {}
    sep   = "═" * 72

    lines = [
        sep,
        f"  MARKET REGIME:  {emoji} {r}",
        f"  {regime.get('regime_note', '')}",
    ]
    if sc:
        lines.append(
            f"  Nifty SC100: {sc.get('last', '?')}  "
            f"1M: {sc.get('chg_1m_pct', '?'):+.1f}%  "
            f"3M: {sc.get('chg_3m_pct', '?'):+.1f}%  "
            f"EMA Stack: {'✓' if sc.get('ema_stacked') else '✗'}  "
            f"Above 200EMA: {'✓' if sc.get('above_200ema') else '✗'}"
        )
    if n50:
        lines.append(
            f"  Nifty 50:    {n50.get('last', '?')}  "
            f"1M: {n50.get('chg_1m_pct', '?'):+.1f}%  "
            f"3M: {n50.get('chg_3m_pct', '?'):+.1f}%"
        )

    # Regime-specific advice
    if r == "BEAR":
        lines += [
            "  ⚠  BEAR REGIME: Layer 1 breakouts have elevated false-positive risk.",
            "     Only act on conviction plays (Section A) with OBV confirmed + RS days ≥ 3.",
        ]
    elif r == "NEUTRAL":
        lines += [
            "  ℹ  NEUTRAL REGIME: Be selective. Favour stocks with Piotroski ≥ 7 and FCF yield > 3%.",
        ]
    elif r == "UNKNOWN":
        lines += [
            "  ⚠  REGIME UNKNOWN: Market data fetch failed — treat as NEUTRAL.",
            "     Check Nifty 50 data above for manual context.",
        ]
    else:  # BULL
        lines += [
            "  ✓  BULL REGIME: Favourable conditions. Standard filters apply.",
        ]
    lines.append(sep)
    return lines


def generate_conviction_report(
    conviction: list[dict],
    l1: list[dict],
    l2: list[dict],
    regime: dict,
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═" * 72
    lines = (
        [sep,
         "  MULTIBAGGER DISCOVERY ENGINE v3 — DAILY COMBINED REPORT",
         f"  {now}",
         sep, ""]
        + regime_banner(regime)
        + [""]
    )

    # ── Section A: Conviction plays ──
    lines += [
        "  ★  SECTION A: CONVICTION LIST (In Both Layer 1 + Layer 2)",
        "     Confirmed technical breakout AND early-stage thesis AND OBV + RS verified",
        sep, "",
    ]

    if conviction:
        for i, s in enumerate(conviction, 1):
            t  = s["technical"]
            pb = s["pre_breakout"]
            fd = s["fundamental"]
            lines += [
                f"★ #{i}  {s['symbol']:<16}  Combined Score: {s['combined_score']:.3f}",
                f"     Price: ₹{s['price']:<8}  MCap: ₹{s['market_cap_cr']:.0f} Cr",
                f"     Stage: {s['price_stage']}",
                f"     L1: {s['layer1_score']:.3f}  |  L2: {s['layer2_score']:.3f}",
                f"     RSI: {t['rsi']}  ADX: {t['adx']}  OBV: {'✓' if t.get('obv_bullish') else '✗'}  "
                f"RS Days: {t.get('rs_days', '?')}  Base: {'Tight ✓' if t.get('has_tight_base') else '–'}",
                f"     3M: {t['price_chg_3m_pct']:+.1f}%  Vol Surge: {'✓' if t['vol_surge'] else '✗'}",
                f"     Promoter: {pb['promoter_pct']:.1f}%  "
                f"FCF Yield: {pb.get('fcf_yield_pct', 'N/A')}%  "
                f"Piotroski: {pb.get('piotroski_score', 'N/A')}/9  "
                f"P/E Expanding: {'✓' if pb.get('pe_expanding') else '✗'}",
                f"     ROE: {fd.get('roe_pct', '?')}%  D/E: {fd.get('de_ratio', '?')}  "
                f"Rev Growth: {fd.get('revenue_growth_pct', '?')}%  "
                f"MCap/Sales: {fd.get('mcap_to_sales', 'N/A')}×",
            ]
            if pb["catalysts"]:
                lines.append(f"     ▶  {' | '.join(pb['catalysts'][:3])}")
            lines.append("")
    else:
        lines += [
            "  No overlap today between Layer 1 and Layer 2.",
            "  This is normal in bear/neutral regimes — monitor Layer 2 watchlist",
            "  and wait for Layer 1 to fire when conditions improve.",
            "",
        ]

    # ── Section B: Layer 1 Top 10 ──
    lines += [sep, "  SECTION B: TOP 10 TECHNICAL BREAKOUTS (Layer 1)", sep, ""]
    if l1:
        for i, s in enumerate(l1[:10], 1):
            fd = s.get("fundamentals", {})
            lines.append(
                f"  #{i:02d}  {s['symbol']:<16}  ₹{s['last_close']:<8}  "
                f"Score: {s['composite_score']:.3f}  RSI: {s['rsi']}  "
                f"OBV: {'✓' if s.get('obv_bullish') else '✗'}  "
                f"RS: {s.get('rs_days', '?')}d  "
                f"3M: {s['price_chg_3m_pct']:+.1f}%  "
                f"FCF: {fd.get('fcf_yield_pct', 'N/A')}%"
            )
    else:
        lines.append("  No breakout candidates today. Market regime may explain this.")
    lines.append("")

    # ── Section C: Layer 2 Top 10 ──
    lines += [sep, "  SECTION C: TOP 10 PRE-BREAKOUT WATCHLIST (Layer 2)", sep, ""]
    for i, s in enumerate(l2[:10], 1):
        p_score = s.get("piotroski", {}).get("piotroski_score")
        fcf_y   = s.get("fcf", {}).get("fcf_yield_pct")
        cat_str = s["catalyst"]["catalysts"][0] if s["catalyst"]["catalysts"] else "–"
        lines.append(
            f"  #{i:02d}  {s['symbol']:<16}  ₹{s['price']:<8}  "
            f"Score: {s['composite_score']:.3f}  "
            f"F-Score: {p_score if p_score is not None else 'N/A'}/9  "
            f"FCF: {fcf_y if fcf_y is not None else 'N/A'}%  "
            f"{s['price_stage']}"
        )
        if cat_str != "–":
            lines.append(f"       ▶ {cat_str}")
    lines.append("")

    lines += [
        sep,
        "  HOW TO USE:",
        "  Section A → Highest conviction. Both technicals AND thesis AND OBV confirmed.",
        "              In bear regime: only act here, nowhere else.",
        "  Section B → Swing trades. Enter on breakout, stop-loss below 52W high.",
        "              Check regime — avoid in BEAR without OBV + RS confirmation.",
        "  Section C → Research queue. Piotroski ≥ 7 + FCF yield = prioritise these.",
        "              Wait for Layer 1 before entry.",
        sep,
        "  ⚠  Educational use only. Not financial advice.",
        sep,
    ]
    return "\n".join(lines)


def run_all():
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  MULTIBAGGER DISCOVERY ENGINE v3 — FULL PIPELINE     ║")
    log.info("║  L1: Breakout + OBV + RS + FCF + MCap/Sales          ║")
    log.info("║  L2: Pre-Breakout + FCF + Piotroski + P/E Trajectory  ║")
    log.info("║  Market regime classifier included                    ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    date_str = datetime.utcnow().strftime("%Y%m%d")

    # ── Market regime (runs first, banners every section) ──
    log.info("\n▶ Classifying market regime...")
    regime = get_market_regime()
    r = regime.get("regime", "UNKNOWN")
    sc = regime.get("smallcap100") or {}
    log.info(f"  Regime: {r}  SC100 1M: {sc.get('chg_1m_pct', '?')}%  3M: {sc.get('chg_3m_pct', '?')}%")
    log.info(f"  {regime.get('regime_note', '')}")

    # ── Get universe ──
    symbols = fetch_nse_symbols()

    # ── Layer 1 ──
    log.info("\n▶ Running Layer 1: Swing Breakout Scanner...")
    layer1_results = run_engine(regime=regime)
    if not layer1_results:
        layer1_results = []

    # ── Layer 2 ──
    log.info("\n▶ Running Layer 2: Pre-Breakout Scanner...")
    layer2_results = run_pre_breakout_scanner(symbols)

    # ── Conviction plays ──
    log.info("\n▶ Finding conviction plays...")
    conviction = find_conviction_plays(layer1_results, layer2_results)
    log.info(f"  Conviction plays: {len(conviction)}")

    # ── Generate reports ──
    combined_report      = generate_conviction_report(conviction, layer1_results, layer2_results, regime)
    pre_breakout_report  = generate_pre_breakout_report(layer2_results)

    # ── Run persistence tracker ──
    log.info("\n▶ Running persistence tracker...")
    tracker_result  = run_tracker(layer2_results, layer1_results, date_str)
    watch_now       = tracker_result["watch_now"]
    persistence_section = generate_persistence_section(tracker_result)

    full_report = combined_report + "\n\n" + pre_breakout_report + "\n\n" + persistence_section

    print("\n" + combined_report)
    if watch_now:
        log.info(f"  Watch now candidates: {[w['symbol'] for w in watch_now]}")

    # ── Save outputs ──
    def save(path, data, is_json=False):
        with open(path, "w", encoding="utf-8") as f:
            if is_json:
                json.dump(data, f, indent=2, default=str)
            else:
                f.write(data)

    save(f"results/scan_{date_str}.json",       layer1_results, is_json=True)
    save("results/latest.json",                 layer1_results, is_json=True)
    save(f"results/watchlist_{date_str}.json",  layer2_results, is_json=True)
    save("results/watchlist.json",              layer2_results, is_json=True)
    save(f"results/conviction_{date_str}.json", conviction,     is_json=True)
    save("results/conviction.json",             conviction,     is_json=True)
    save(f"results/regime_{date_str}.json",     regime,         is_json=True)
    save("results/regime.json",                 regime,         is_json=True)
    save("results/watch_now.json",              watch_now,      is_json=True)
    save(f"results/report_{date_str}.txt",      full_report)
    save("results/daily_report.txt",            full_report)

    log.info("\n✅ All outputs saved to results/")
    log.info(f"   Regime:             {r}")
    log.info(f"   Layer 1 breakouts:  {len(layer1_results)}")
    log.info(f"   Layer 2 watchlist:  {len(layer2_results)}")
    log.info(f"   Conviction plays:   {len(conviction)}")
    log.info(f"   Watch now:          {len(watch_now)}")
    log.info(f"   Total tracked:      {tracker_result['total_tracked']}")

    return {
        "regime":     regime,
        "layer1":     layer1_results,
        "layer2":     layer2_results,
        "conviction": conviction,
    }


if __name__ == "__main__":
    run_all()
