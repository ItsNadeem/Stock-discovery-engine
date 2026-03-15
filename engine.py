"""
engine.py — Multibagger Discovery Engine (yfinance edition)
Pipeline: Swing Breakout Scanner → Fundamental Filter → Top 20
No API keys. No broker account. Runs free on GitHub Actions.
"""

import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from universe import fetch_nse_symbols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("results/scanner.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CONFIG — tweak these to tune the scanner
# ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Universe filters
    min_price: float        = 20.0       # ₹ — ignore penny stocks
    max_price: float        = 2000.0     # ₹ — small/micro cap focus
    min_avg_volume: int     = 75_000     # min 75k avg daily volume
    max_market_cap_cr: float = 8_000.0  # ₹ Cr — small cap ceiling

    # Breakout params
    breakout_lookback_days: int      = 252    # ~1 trading year
    atr_buffer_multiplier: float     = 0.3    # Close > 52W high + 0.3×ATR
    volume_surge_multiplier: float   = 1.5    # Last vol > 1.5× 20-day avg

    # Indicators
    ema_fast: int   = 21
    ema_slow: int   = 55
    rsi_period: int = 14
    rsi_lo: float   = 48.0    # min RSI to qualify
    rsi_hi: float   = 78.0    # max RSI (not overbought)
    adx_min: float  = 18.0    # trend strength floor

    # Fundamental hard filters (from yfinance info)
    max_de_ratio: float         = 1.2
    min_roe_pct: float          = 10.0
    min_revenue_growth_pct: float = 8.0   # yfinance gives TTM growth

    # Scoring weights (must sum to 1.0)
    w_breakout:     float = 0.35
    w_momentum:     float = 0.25
    w_fundamental:  float = 0.30
    w_volume:       float = 0.10

    top_n: int = 20
    batch_size: int = 50       # symbols per yfinance batch download
    sleep_between_batches: float = 2.0


CFG = Config()


# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up   = df["High"].diff()
    down = -df["Low"].diff()
    pdm  = up.where((up > down) & (up > 0), 0.0)
    ndm  = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(df, n)
    pdi  = 100 * pdm.rolling(n).mean() / atr_
    ndi  = 100 * ndm.rolling(n).mean() / atr_
    dx   = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan))
    return dx.rolling(n).mean()


# ─────────────────────────────────────────────────────────────
# STAGE 1 — SWING BREAKOUT ANALYSER (per symbol)
# ─────────────────────────────────────────────────────────────
def analyse_technicals(df: pd.DataFrame, symbol: str) -> Optional[dict]:
    """
    Runs the full technical suite on OHLCV dataframe.
    Returns signal dict or None if stock doesn't qualify.
    """
    min_rows = CFG.breakout_lookback_days + 60
    if df is None or len(df) < min_rows:
        return None

    c = df["Close"]
    v = df["Volume"]

    # Core indicators
    e21    = ema(c, CFG.ema_fast)
    e55    = ema(c, CFG.ema_slow)
    rsi_s  = rsi(c, CFG.rsi_period)
    atr_s  = atr(df)
    adx_s  = adx(df)

    last_close = float(c.iloc[-1])
    last_vol   = float(v.iloc[-1])
    avg_vol    = float(v.rolling(20).mean().iloc[-1])

    # ── Universe pre-filters ──
    if not (CFG.min_price <= last_close <= CFG.max_price):
        return None
    if avg_vol < CFG.min_avg_volume:
        return None

    # ── Breakout logic ──
    rolling_high  = c.rolling(CFG.breakout_lookback_days).max().shift(1)
    atr_now       = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0
    breakout_lvl  = float(rolling_high.iloc[-1]) + atr_now * CFG.atr_buffer_multiplier

    is_breakout   = last_close >= breakout_lvl
    vol_surge     = last_vol   >= avg_vol * CFG.volume_surge_multiplier
    ema_bull      = float(e21.iloc[-1]) > float(e55.iloc[-1])
    last_rsi      = float(rsi_s.iloc[-1])
    rsi_ok        = CFG.rsi_lo <= last_rsi <= CFG.rsi_hi
    last_adx      = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0
    adx_ok        = last_adx >= CFG.adx_min

    # Must pass hard technical gates
    if not (is_breakout and ema_bull and rsi_ok):
        return None

    # ── Scores (0–1) ──
    w52_high = float(rolling_high.iloc[-1])
    breakout_score = (
        min((last_close - w52_high) / w52_high * 20, 1.0)
        if (is_breakout and w52_high > 0) else 0.0
    )

    pct_above_e55  = (last_close - float(e55.iloc[-1])) / float(e55.iloc[-1])
    momentum_score = min(max(pct_above_e55 * 5, 0), 1.0)

    vol_ratio      = last_vol / avg_vol if avg_vol > 0 else 0
    volume_score   = min(vol_ratio / 3, 1.0)

    # 3-month return
    idx_3m     = max(len(c) - 63, 0)
    price_3m   = float(c.iloc[idx_3m])
    chg_3m_pct = (last_close - price_3m) / price_3m * 100 if price_3m > 0 else 0

    return {
        "symbol":          symbol,
        "last_close":      round(last_close, 2),
        "avg_volume":      int(avg_vol),
        "last_volume":     int(last_vol),
        "rsi":             round(last_rsi, 1),
        "atr":             round(atr_now, 2),
        "adx":             round(last_adx, 1),
        "ema21":           round(float(e21.iloc[-1]), 2),
        "ema55":           round(float(e55.iloc[-1]), 2),
        "week52_high":     round(w52_high, 2),
        "breakout_level":  round(breakout_lvl, 2),
        "is_breakout":     is_breakout,
        "vol_surge":       vol_surge,
        "ema_bullish":     ema_bull,
        "rsi_ok":          rsi_ok,
        "adx_ok":          adx_ok,
        "price_chg_3m_pct": round(chg_3m_pct, 1),
        "breakout_score":  round(breakout_score, 3),
        "momentum_score":  round(momentum_score, 3),
        "volume_score":    round(volume_score, 3),
    }


# ─────────────────────────────────────────────────────────────
# STAGE 2 — FUNDAMENTAL FILTER (yfinance .info)
# ─────────────────────────────────────────────────────────────
def get_fundamentals(symbol: str) -> tuple[float, dict]:
    """
    Fetches key ratios from yfinance .info.
    Returns (score 0–1, details dict).
    Score 0.0 = hard fail (stock disqualified).
    """
    try:
        info = yf.Ticker(symbol).info
    except Exception as e:
        log.debug(f"{symbol} info fetch failed: {e}")
        return 0.5, {}   # neutral if unavailable

    def g(key, default=None):
        v = info.get(key, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v

    de          = g("debtToEquity")          # yfinance returns as percentage sometimes
    roe         = g("returnOnEquity")         # decimal e.g. 0.18 = 18%
    rev_growth  = g("revenueGrowth")          # decimal e.g. 0.15 = 15%
    mktcap      = g("marketCap", 0)
    profit_mgn  = g("profitMargins", 0)

    # Convert units
    de_ratio    = (de / 100) if (de is not None and de > 5) else de   # normalize if pct
    roe_pct     = (roe * 100) if roe is not None else None
    rev_gr_pct  = (rev_growth * 100) if rev_growth is not None else None
    mktcap_cr   = mktcap / 1e7 if mktcap else None  # ₹ Cr

    details = {
        "market_cap_cr":    round(mktcap_cr, 0) if mktcap_cr else None,
        "de_ratio":         round(de_ratio, 2) if de_ratio is not None else None,
        "roe_pct":          round(roe_pct, 1) if roe_pct is not None else None,
        "revenue_growth_pct": round(rev_gr_pct, 1) if rev_gr_pct is not None else None,
        "profit_margin_pct":  round(profit_mgn * 100, 1) if profit_mgn else None,
    }

    # ── Hard filters ──
    if mktcap_cr and mktcap_cr > CFG.max_market_cap_cr:
        return 0.0, details   # Too large — not small cap
    if de_ratio is not None and de_ratio > CFG.max_de_ratio:
        return 0.0, details
    if roe_pct is not None and roe_pct < CFG.min_roe_pct:
        return 0.0, details
    if rev_gr_pct is not None and rev_gr_pct < CFG.min_revenue_growth_pct:
        return 0.0, details

    # ── Soft score ──
    roe_score  = min(max((roe_pct or 10) - 10, 0) / 30, 1.0)
    rev_score  = min(max((rev_gr_pct or 0), 0) / 50, 1.0)
    de_score   = 1.0 - min((de_ratio or 0) / CFG.max_de_ratio, 1.0)
    mgn_score  = min(max((profit_mgn or 0) * 100, 0) / 25, 1.0)

    fscore = roe_score * 0.30 + rev_score * 0.35 + de_score * 0.20 + mgn_score * 0.15
    return round(fscore, 3), details


# ─────────────────────────────────────────────────────────────
# COMPOSITE SCORER
# ─────────────────────────────────────────────────────────────
def composite_score(tech: dict, fscore: float) -> float:
    return round(
        tech["breakout_score"] * CFG.w_breakout +
        tech["momentum_score"] * CFG.w_momentum +
        fscore                 * CFG.w_fundamental +
        tech["volume_score"]   * CFG.w_volume,
        4
    )


# ─────────────────────────────────────────────────────────────
# MULTIBAGGER SIGNAL FLAGS
# ─────────────────────────────────────────────────────────────
def multibagger_flags(s: dict) -> list[str]:
    flags = []
    fd = s.get("fundamentals", {})
    if s.get("is_breakout") and s.get("vol_surge"):
        flags.append("🚀 52W Breakout + Volume Surge")
    if s.get("adx_ok") and s.get("adx", 0) > 25:
        flags.append(f"📈 Strong Trend (ADX {s['adx']})")
    if s.get("price_chg_3m_pct", 0) > 25:
        flags.append(f"⚡ 3M Return +{s['price_chg_3m_pct']}%")
    if (fd.get("revenue_growth_pct") or 0) > 25:
        flags.append(f"💰 Rev Growth {fd['revenue_growth_pct']}%")
    if (fd.get("roe_pct") or 0) > 20:
        flags.append(f"✅ ROE {fd['roe_pct']}%")
    if (fd.get("de_ratio") or 1) < 0.3:
        flags.append("🛡️ Nearly Debt-Free")
    if (fd.get("profit_margin_pct") or 0) > 15:
        flags.append(f"🏆 Margin {fd['profit_margin_pct']}%")
    return flags


# ─────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_report(top_n: list[dict]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═" * 72
    lines = [
        sep,
        "  MULTIBAGGER DISCOVERY ENGINE  —  NSE NIGHTLY SCAN",
        f"  {now}  |  yfinance  |  GitHub Actions",
        sep,
        f"  Top {len(top_n)} candidates  (Swing Breakout × Fundamental Quality)",
        sep, "",
    ]

    for i, s in enumerate(top_n, 1):
        fd    = s.get("fundamentals", {})
        flags = multibagger_flags(s)
        mcap  = f"₹{fd['market_cap_cr']} Cr" if fd.get("market_cap_cr") else "N/A"
        lines += [
            f"#{i:02d}  {s['symbol']:<16}  Score: {s['composite_score']:.3f}   MCap: {mcap}",
            f"    Price ₹{s['last_close']:<9}  RSI {s['rsi']:<6}  ADX {s['adx']:<6}  3M: {s['price_chg_3m_pct']:+.1f}%",
            f"    52W High ₹{s['week52_high']:<8}  Breakout @ ₹{s['breakout_level']}",
            f"    EMA21/55: ₹{s['ema21']} / ₹{s['ema55']}   Vol Surge: {'✓' if s['vol_surge'] else '✗'}   Trend: {'Bullish ✓' if s['ema_bullish'] else 'Bearish ✗'}",
            f"    Fundamental Score: {s['fundamental_score']:.2f}   "
            f"ROE: {fd.get('roe_pct','?')}%   "
            f"D/E: {fd.get('de_ratio','?')}   "
            f"Rev Growth: {fd.get('revenue_growth_pct','?')}%",
        ]
        if flags:
            lines.append(f"    ▶  {' | '.join(flags)}")
        lines.append("")

    lines += [sep, "  ⚠  Educational use only. Not financial advice.", sep]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_engine() -> list[dict]:
    log.info("══════════════════════════════════════════════════════")
    log.info("  MULTIBAGGER DISCOVERY ENGINE — NIGHTLY SCAN START")
    log.info("══════════════════════════════════════════════════════")

    symbols = fetch_nse_symbols()
    log.info(f"Universe: {len(symbols)} NSE symbols")

    # ── STAGE 1: Batch-download OHLCV + Technical Breakout Scan ──
    log.info("Stage 1: Swing breakout scan...")
    breakout_candidates = []
    batches = [symbols[i:i + CFG.batch_size] for i in range(0, len(symbols), CFG.batch_size)]

    for b_idx, batch in enumerate(batches):
        log.info(f"  Batch {b_idx+1}/{len(batches)} ({len(batch)} symbols)...")
        try:
            raw = yf.download(
                batch,
                period="2y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            log.warning(f"  Batch download error: {e}")
            time.sleep(5)
            continue

        for sym in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    df = raw[sym].copy() if sym in raw.columns.get_level_values(0) else None

                if df is None or df.empty:
                    continue
                df.dropna(subset=["Close", "Volume"], inplace=True)

                result = analyse_technicals(df, sym)
                if result:
                    breakout_candidates.append(result)
                    log.info(f"    ✓ Breakout: {sym}  ₹{result['last_close']}  RSI={result['rsi']}")
            except Exception as e:
                log.debug(f"  {sym} error: {e}")

        time.sleep(CFG.sleep_between_batches)

    log.info(f"Stage 1 complete. Breakout candidates: {len(breakout_candidates)}")

    if not breakout_candidates:
        log.warning("No breakout candidates today. Exiting.")
        return []

    # ── STAGE 2: Fundamental Filter + Scoring ──
    log.info("Stage 2: Fundamental filter + scoring...")
    ranked = []
    for cand in breakout_candidates:
        sym = cand["symbol"]
        fscore, fd = get_fundamentals(sym)
        if fscore == 0.0:
            log.debug(f"  ✗ Fundamental fail: {sym}")
            continue
        total = composite_score(cand, fscore)
        ranked.append({
            **cand,
            "fundamental_score": fscore,
            "composite_score":   total,
            "fundamentals":      fd,
        })
        log.info(f"  ✓ {sym:16s} composite={total:.3f}  fscore={fscore:.3f}")
        time.sleep(0.3)   # polite to yfinance

    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    top_n = ranked[:CFG.top_n]
    log.info(f"Stage 2 complete. Final candidates: {len(ranked)} → top {len(top_n)}")

    # ── Output ──
    report = generate_report(top_n)
    print("\n" + report)

    date_str = datetime.utcnow().strftime("%Y%m%d")

    with open(f"results/report_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(f"results/scan_{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2, default=str)

    # Always overwrite latest for easy linking
    with open("results/latest.json", "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2, default=str)

    with open("results/latest_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    log.info("Results written to results/")
    return top_n


if __name__ == "__main__":
    run_engine()
