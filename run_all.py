"""
run_all.py — Master Orchestrator v5

Changes from v4:
- Layer 2 now runs multibagger_dna.py + concall_analyser.py
- Conviction report shows DNA grade and concall flags
- regime passed through to tracker
"""

import json
import logging
import os
from datetime import datetime

from universe import fetch_nse_symbols
from engine import run_engine, get_market_regime, generate_report as generate_l1_report
from pre_breakout_scanner import run_pre_breakout_scanner, generate_pre_breakout_report
from scan_tracker import run_tracker, generate_persistence_section
from public_data_fetcher import init_public_data

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


def find_conviction_plays(layer1, layer2):
    l1_syms = {s["symbol"] for s in layer1}
    out = []
    for s in layer2:
        if s["symbol"] in l1_syms:
            l1 = next(x for x in layer1 if x["symbol"] == s["symbol"])
            dna = s.get("dna", {})
            out.append({
                "symbol":       s["symbol"],
                "price":        s["price"],
                "market_cap_cr":s["market_cap_cr"],
                "price_stage":  s["price_stage"],
                "layer1_score": l1["composite_score"],
                "layer2_score": s["composite_score"],
                "combined_score": round(l1["composite_score"]*0.5 + s["composite_score"]*0.5, 4),
                "dna_grade":    dna.get("dna_grade","?"),
                "dna_score":    dna.get("dna_score", 0),
                "rev_accel":    dna.get("rev_accel_score", 0),
                "dna_flags":    dna.get("dna_flags", []),
                "concall":      s.get("concall") or {},
                "piotroski":    s.get("piotroski_score"),
                "screener_roce":(s.get("screener") or {}).get("roce_pct"),
                "technical": {
                    "rsi": l1.get("rsi"), "adx": l1.get("adx"),
                    "vol_surge": l1.get("vol_surge"),
                    "obv_bullish": l1.get("obv_bullish"),
                    "rs_days": l1.get("rs_days"),
                    "price_chg_3m_pct": l1.get("price_chg_3m_pct"),
                },
                "public": {
                    "promoter_buying": (s.get("public_data") or {}).get("promoter_buying"),
                    "insider_value_cr": (s.get("public_data") or {}).get("insider_value_cr"),
                    "pledge_pct": (s.get("public_data") or {}).get("pledge_pct"),
                    "public_flags": (s.get("public_data") or {}).get("public_flags", []),
                },
                "scanned_at": s["scanned_at"],
            })
    out.sort(key=lambda x: x["combined_score"], reverse=True)
    return out


def regime_banner(regime):
    r     = regime.get("regime", "UNKNOWN")
    emoji = {"BULL":"🟢","NEUTRAL":"🟡","BEAR":"🔴","UNKNOWN":"⚪"}.get(r,"⚪")
    sc    = regime.get("smallcap100") or {}
    n50   = regime.get("nifty50") or {}
    sep   = "═"*72
    lines = [sep, f"  MARKET REGIME: {emoji} {r}", f"  {regime.get('regime_note','')}"]
    if sc:
        lines.append(
            f"  Nifty SC100: {sc.get('last','?')} "
            f"1M:{sc.get('chg_1m_pct','?'):+.1f}% 3M:{sc.get('chg_3m_pct','?'):+.1f}% "
            f"EMA Stack:{'✓' if sc.get('ema_stacked') else '✗'} "
            f"Above 200EMA:{'✓' if sc.get('above_200ema') else '✗'}"
        )
    if n50:
        lines.append(f"  Nifty 50: {n50.get('last','?')} "
                     f"1M:{n50.get('chg_1m_pct','?'):+.1f}% 3M:{n50.get('chg_3m_pct','?'):+.1f}%")
    if r=="BEAR":
        lines.append("  ⚠ BEAR: Act only on conviction plays with DNA Grade A + insider buying.")
    elif r=="NEUTRAL":
        lines.append("  ℹ NEUTRAL: DNA Grade A + concall forward guidance required.")
    else:
        lines.append("  ✓ BULL: Standard filters apply.")
    lines.append(sep)
    return lines


def generate_conviction_report(conviction, l1, l2, regime):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═"*72
    lines = ([sep,"  MULTIBAGGER DISCOVERY ENGINE v5 — DAILY REPORT",f"  {now}",sep,""]
             + regime_banner(regime) + [""])

    lines += [
        "  ★ SECTION A: CONVICTION LIST (Layer 1 + Layer 2 overlap)",
        "  Technical breakout confirmed AND multibagger DNA validated",
        sep, "",
    ]
    if conviction:
        for i, s in enumerate(conviction, 1):
            t   = s["technical"]
            pub = s["public"]
            cc  = s.get("concall") or {}
            lines += [
                f"★ #{i} {s['symbol']:<16} Combined:{s['combined_score']:.3f}",
                f"   Price:₹{s['price']}  MCap:₹{s['market_cap_cr']:.0f}Cr  {s['price_stage']}",
                f"   L1:{s['layer1_score']:.3f}  L2:{s['layer2_score']:.3f}",
                f"   DNA:{s['dna_score']:.3f} [{s['dna_grade']}]  RevAccel:{s['rev_accel']:.2f}",
                f"   Piotroski:{s['piotroski']}/9  ROCE:{s['screener_roce'] or 'N/A'}%",
                f"   RSI:{t['rsi']} ADX:{t['adx']} OBV:{'✓' if t.get('obv_bullish') else '✗'}"
                f" RS:{t.get('rs_days','?')}d 3M:{t.get('price_chg_3m_pct',0):+.1f}%",
                f"   Public: Insider:{'✓' if pub.get('promoter_buying') else '–'}"
                f" ₹{pub.get('insider_value_cr',0):.1f}Cr"
                f" Pledge:{pub.get('pledge_pct','N/A')}%",
            ]
            for f in s.get("dna_flags",[])[:2]:
                lines.append(f"   ▶ {f}")
            if cc.get("tier1_signals"):
                lines.append(f"   🎙️ {cc['tier1_signals'][0][:80]}")
                if cc.get("manual_review"):
                    lines.append("   ★ READ FULL CONCALL TRANSCRIPT")
            lines.append("")
    else:
        lines += [
            "  No overlap today — normal in bear/neutral regimes.",
            "  Monitor Layer 2 for DNA Grade A stocks approaching Layer 1 breakout.",
            "",
        ]

    lines += [sep, "  SECTION B: LAYER 1 BREAKOUTS (Technical)", sep, ""]
    for i, s in enumerate(l1[:10], 1):
        lines.append(
            f"  #{i:02d} {s['symbol']:<16} ₹{s['last_close']:<8}"
            f" Score:{s['composite_score']:.3f} RSI:{s['rsi']}"
            f" OBV:{'✓' if s.get('obv_bullish') else '✗'}"
            f" RS:{s.get('rs_days','?')}d 3M:{s['price_chg_3m_pct']:+.1f}%"
        )
    lines.append("")

    lines += [sep, "  SECTION C: LAYER 2 — MULTIBAGGER WATCHLIST (DNA Ranked)", sep, ""]
    for i, s in enumerate(l2[:10], 1):
        dna = s.get("dna",{})
        sc  = s.get("screener") or {}
        cc  = s.get("concall") or {}
        lines.append(
            f"  #{i:02d} {s['symbol']:<16} ₹{s['price']:<8} ₹{s['market_cap_cr']:.0f}Cr"
            f" Score:{s['composite_score']:.3f}"
            f" DNA:{dna.get('dna_score',0):.3f}[{dna.get('dna_grade','?')[:1]}]"
            f" P:{s['piotroski_score']}/9"
            f" ROCE:{sc.get('roce_pct','N/A')}%"
        )
        if dna.get("dna_flags"):
            lines.append(f"       ▶ {dna['dna_flags'][0]}")
        if cc.get("manual_review"):
            lines.append(f"       🎙️ READ CONCALL — forward signals found")
    lines.append("")

    lines += [
        sep,
        "  HOW TO ACT:",
        "  Section A: Both technicals + DNA confirmed. Highest priority.",
        "  Section B: Swing entry. Wait for volume + OBV confirmation.",
        "  Section C: Research queue. DNA Grade A + concall review first.",
        "  DNA Grade A (≥0.75) stocks: read concall before doing anything else.",
        "  🧑‍💼 Insider buying = buy more aggressively, smaller position size.",
        sep,
        "  ⚠ Educational only. Not financial advice.",
        sep,
    ]
    return "\n".join(lines)


def run_all():
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  MULTIBAGGER DISCOVERY ENGINE v5                    ║")
    log.info("║  L1: Momentum  L2: DNA + Concall + Screener        ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    date_str = datetime.utcnow().strftime("%Y%m%d")

    log.info("\n▶ Market regime...")
    regime = get_market_regime()
    log.info(f"  {regime.get('regime')} — {regime.get('regime_note','')}")

    symbols = fetch_nse_symbols()

    log.info("\n▶ Layer 1: Swing Breakout...")
    layer1 = run_engine(regime=regime) or []

    log.info("\n▶ Public data (insider / bulk deals)...")
    nse_session, universe_deals = init_public_data()

    log.info("\n▶ Layer 2: Multibagger DNA scan...")
    layer2 = run_pre_breakout_scanner(
        symbols, nse_session=nse_session,
        universe_deals=universe_deals, regime=regime,
    )

    log.info("\n▶ Conviction plays...")
    conviction = find_conviction_plays(layer1, layer2)
    log.info(f"  {len(conviction)} conviction plays")

    combined  = generate_conviction_report(conviction, layer1, layer2, regime)
    l2_report = generate_pre_breakout_report(layer2)

    log.info("\n▶ Persistence tracker...")
    tracker   = run_tracker(layer2, layer1, date_str, regime=regime)
    watch_now = tracker["watch_now"]
    persist   = generate_persistence_section(tracker, regime=regime)

    full_report = combined + "\n\n" + l2_report + "\n\n" + persist
    print("\n" + combined)

    def save(path, data, is_json=False):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str) if is_json else f.write(data)

    save(f"results/scan_{date_str}.json",       layer1,     is_json=True)
    save("results/latest.json",                 layer1,     is_json=True)
    save(f"results/watchlist_{date_str}.json",  layer2,     is_json=True)
    save("results/watchlist.json",              layer2,     is_json=True)
    save(f"results/conviction_{date_str}.json", conviction, is_json=True)
    save("results/conviction.json",             conviction, is_json=True)
    save(f"results/regime_{date_str}.json",     regime,     is_json=True)
    save("results/regime.json",                 regime,     is_json=True)
    save("results/watch_now.json",              watch_now,  is_json=True)
    save(f"results/report_{date_str}.txt",      full_report)
    save("results/daily_report.txt",            full_report)

    log.info(f"\n✅ Done — L1:{len(layer1)} L2:{len(layer2)} Conviction:{len(conviction)} WatchNow:{len(watch_now)}")
    return {"regime": regime, "layer1": layer1, "layer2": layer2, "conviction": conviction}


if __name__ == "__main__":
    run_all()
