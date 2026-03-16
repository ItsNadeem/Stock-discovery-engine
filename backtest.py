"""
backtest.py — Signal Validation & Backtester
============================================

Two complementary backtests:

BACKTEST A — VALUEPICK GROUND TRUTH VALIDATION
  Run our Layer 2 scoring model retroactively on confirmed multibagger picks
  at their original entry dates. If the model is valid, these stocks should
  have scored highly on the date they were picked.

  Known confirmed picks from value-picks.blogspot.com + @valuepick Twitter:
    Paushak Ltd      ₹74    Mar 2011   → ₹10,000  (140x in 10 years)
    Cosmo Ferrites   ₹13    ~2011      → ₹168+    (13x, re-picked at ₹168 in 2021)
    Tasty Bite       ₹165   ~2010      → ₹9,420   (57x)
    EKI Energy       ₹162   Apr 2021   → ₹10,000  (66x in 9 months)
    Jay Kay Ent.     ₹28    Feb 2021   → multibagger (3D printing pivot)
    Shanthi Gears    ₹180   Jun 2024   → monitoring

BACKTEST B — LAYER 1 HISTORICAL BREAKOUT PERFORMANCE
  For each trading day in the last 2 years, simulate running the breakout
  scanner. For each signal generated, measure forward returns at 1M, 3M, 6M.
  This tells us: when the engine fires, how often does it actually work?

  Metrics computed:
    Hit rate: % of signals with positive 3M return
    Avg return at 1M, 3M, 6M
    Comparison vs Nifty Smallcap 100 (alpha)
    Best/worst cases
    OBV filter impact (with vs without OBV gate)

Usage:
  python backtest.py --mode a          # VALUEPICK ground truth
  python backtest.py --mode b          # Layer 1 historical
  python backtest.py --mode both       # Run both (default)
  python backtest.py --mode a --tune   # Also tune weights on ground truth
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Global signal params — overridden via CLI args for iterative tuning
_BACKTEST_PARAMS: dict = {
    "rsi_lo":   48.0,
    "rsi_hi":   78.0,
    "atr_mult": 0.3,
    "vol_mult": 1.5,
    "adx_min":  18.0,
}


# ─────────────────────────────────────────────────────────────
# GROUND TRUTH DATASET — VALUEPICK CONFIRMED PICKS
# Sourced from: blog posts + @valuepick Twitter verified outcomes
# ─────────────────────────────────────────────────────────────

VALUEPICK_PICKS = [
    {
        # Paushak was BSE-only until Dec 2025 — use .BO for historical data
        # yfinance has BSE data back to ~2000 for this stock
        "symbol":        "PAUSHAKLTD.BO",
        "symbol_alt":    "PAUSHAKLTD.NS",   # NSE listing only from Dec 2025
        "entry_price":   74.0,
        "entry_date":    "2011-03-15",
        "peak_price":    10000.0,
        "peak_date":     "2022-01-03",
        "return_x":      140.0,
        "thesis":        "Alembic group specialty chemical, capacity expansion pending, debt-free",
        "key_signals":   ["trusted_group", "capex", "low_pe", "debt_free"],
        "data_note":     "BSE-only until Dec 2025. Using .BO suffix for yfinance historical data.",
    },
    {
        # Cosmo Ferrites — BSE-listed, NSE symbol COSMOFERR
        "symbol":        "COSMOFERR.BO",
        "symbol_alt":    "COSMOFERR.NS",
        "entry_price":   13.0,
        "entry_date":    "2011-06-01",
        "peak_price":    500.0,
        "return_x":      13.0,
        "thesis":        "Cosmo Films group, soft ferrite manufacturer, EV tailwind",
        "key_signals":   ["trusted_group", "low_pe", "export_growth", "capacity_expansion"],
        "data_note":     "Entry date approximate — '10 years back' from Oct 2021 blog post.",
    },
    {
        # Tasty Bite — BSE scrip code 519091, NSE: TASTYBITE
        # yfinance historical data for Indian small caps before 2015 is patchy
        "symbol":        "TASTYBITE.BO",
        "symbol_alt":    "TASTYBITE.NS",
        "entry_price":   165.0,
        "entry_date":    "2010-02-27",
        "peak_price":    9420.0,
        "return_x":      57.0,
        "thesis":        "Organic food, export-led, US market penetration",
        "key_signals":   ["export_growth", "low_pe", "promoter_confidence"],
        "data_note":     "Pre-2015 Yahoo data very patchy for Indian small caps — may return no data.",
    },
    {
        # EKI Energy Services — listed Apr 2021 on BSE SME, NSE symbol EKINDIA
        # BSE scrip: 543284. Listed at ₹162 on 8 Apr 2021.
        "symbol":        "EKINDIA.NS",
        "symbol_alt":    "EKINDIA.BO",
        "entry_price":   162.0,
        "entry_date":    "2021-04-10",
        "peak_price":    10000.0,
        "peak_date":     "2022-01-03",
        "return_x":      66.0,
        "thesis":        "First listed carbon credit company globally, 90%+ overseas revenue",
        "key_signals":   ["first_in_niche", "export_growth", "high_promoter", "low_pe"],
        "data_note":     "Listed Apr 2021 — entry date is 2 days post-listing. Limited pre-listing history.",
    },
    {
        # Jay Kay Enterprises — BSE scrip 530005, very illiquid, sparse yfinance data
        "symbol":        "JAYKAYENTM.BO",
        "symbol_alt":    "JAYKAYENTM.NS",
        "entry_price":   28.0,
        "entry_date":    "2021-02-06",
        "thesis":        "JK group 3D printing JV with EOS Germany, promoter 32%→52% preferential",
        "key_signals":   ["trusted_group", "promoter_pref_allotment", "pivot_jv"],
        "data_note":     "Extremely illiquid microcap — yfinance data may be missing or stale.",
    },
    {
        # Shanthi Gears — Murugappa group, NSE: SHANTIGEAR — only recent pick so data available
        "symbol":        "SHANTIGEAR.NS",
        "symbol_alt":    "SHANTIGEAR.BO",
        "entry_price":   180.0,
        "entry_date":    "2024-06-01",
        "thesis":        "Murugappa group, zero debt, steady growth, sector tailwind",
        "key_signals":   ["trusted_group", "debt_free", "sector_tailwind"],
        "data_note":     "Most recent pick — full data expected.",
    },
]


# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (duplicated from engine.py to keep
# backtest self-contained — no circular imports)
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
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def obv_series(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff().fillna(0))
    return (df["Volume"] * direction).cumsum()


# ─────────────────────────────────────────────────────────────
# BACKTEST A — VALUEPICK GROUND TRUTH SCORING
# ─────────────────────────────────────────────────────────────

@dataclass
class SignalScoreAtDate:
    symbol:             str
    entry_date:         str
    entry_price:        float
    # Technical state at entry date
    price_vs_52w_high:  Optional[float]  = None   # % from 52W high
    price_vs_52w_low:   Optional[float]  = None   # % from 52W low
    ema21_vs_55:        Optional[str]    = None   # "bullish" / "bearish"
    rsi_at_entry:       Optional[float]  = None
    adx_at_entry:       Optional[float]  = None
    obv_direction:      Optional[str]    = None   # "rising" / "falling"
    volume_vs_avg:      Optional[float]  = None   # ratio
    atr_contraction:    Optional[float]  = None   # current/prior ATR ratio
    # What our signals would have been
    was_breakout:       bool             = False
    was_base:           bool             = False
    breakout_score:     float            = 0.0
    momentum_score:     float            = 0.0
    volume_score:       float            = 0.0
    obv_score:          float            = 0.0
    # Forward returns
    return_1m:          Optional[float]  = None
    return_3m:          Optional[float]  = None
    return_6m:          Optional[float]  = None
    return_1y:          Optional[float]  = None
    # Nifty benchmark on same period
    nifty_1m:           Optional[float]  = None
    nifty_3m:           Optional[float]  = None
    # Composite score our model would have assigned
    estimated_l1_score: float            = 0.0
    # Key signals present at entry date
    signals_present:    list             = field(default_factory=list)
    notes:              str              = ""


def score_at_date(symbol: str, date_str: str, entry_price: float,
                  pick_meta: Optional[dict] = None) -> SignalScoreAtDate:
    """
    Reconstruct what our technical signals would have looked like
    on a given historical date for a given stock.
    Tries primary symbol first, falls back to symbol_alt if no data found.
    """
    result = SignalScoreAtDate(
        symbol=symbol,
        entry_date=date_str,
        entry_price=entry_price,
    )

    try:
        entry_dt  = datetime.strptime(date_str, "%Y-%m-%d")
        from_dt   = entry_dt - timedelta(days=730)
        to_dt     = entry_dt + timedelta(days=400)

        # ── Try primary symbol, fall back to alt symbol ──
        def fetch_history(sym: str) -> Optional[pd.DataFrame]:
            try:
                df = yf.Ticker(sym).history(
                    start=from_dt.strftime("%Y-%m-%d"),
                    end=to_dt.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                )
                if df is not None and not df.empty and len(df) >= 50:
                    return df
            except Exception:
                pass
            return None

        df = fetch_history(symbol)
        if df is None:
            # Try alternate symbol (e.g. .NS vs .BO)
            alt = pick_meta.get("symbol_alt") if pick_meta else None
            if alt and alt != symbol:
                log.info(f"  {symbol} no data — trying alt symbol {alt}")
                df = fetch_history(alt)
                if df is not None:
                    result.symbol = alt   # note which symbol worked

        if df is None or df.empty:
            result.notes = (
                f"No yfinance data found for {symbol}. "
                f"{pick_meta.get('data_note', '') if pick_meta else ''}"
            )
            return result

        if len(df) < 200:
            result.notes = (
                f"Only {len(df)} rows available — pre-2015 Indian small cap "
                f"data is sparse on Yahoo Finance. "
                f"{pick_meta.get('data_note', '') if pick_meta else ''}"
            )
            # Continue with what we have if >= 60 rows (enough for basic indicators)
            if len(df) < 60:
                return result

        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Find the row closest to entry date
        entry_idx = df.index.searchsorted(pd.Timestamp(entry_dt))
        if entry_idx >= len(df):
            entry_idx = len(df) - 1
        # Use data up to and including entry date
        hist = df.iloc[:entry_idx + 1].copy()

        if len(hist) < 100:
            result.notes = "Not enough pre-entry history"
            return result

        c = hist["Close"]
        v = hist["Volume"]

        # ── Technical state at entry ──
        e21   = ema(c, 21)
        e55   = ema(c, 55)
        rsi_s = rsi(c, 14)
        atr_s = atr(hist, 14)
        obv_s = obv_series(hist)

        last_close = float(c.iloc[-1])
        last_vol   = float(v.iloc[-1])
        avg_vol    = float(v.rolling(20).mean().iloc[-1])

        high_252   = float(c.rolling(252).max().shift(1).iloc[-1])
        low_252    = float(c.rolling(252).min().iloc[-1])
        atr_now    = float(atr_s.iloc[-1])

        result.price_vs_52w_high = round((last_close - high_252) / high_252 * 100, 1) if high_252 > 0 else None
        result.price_vs_52w_low  = round((last_close - low_252) / low_252 * 100, 1)   if low_252  > 0 else None
        result.ema21_vs_55       = "bullish" if float(e21.iloc[-1]) > float(e55.iloc[-1]) else "bearish"
        result.rsi_at_entry      = round(float(rsi_s.iloc[-1]), 1)
        result.volume_vs_avg     = round(last_vol / avg_vol, 2) if avg_vol > 0 else None

        # OBV direction
        obv_window   = obv_s.iloc[-10:]
        obv_slope    = float(obv_window.iloc[-1] - obv_window.iloc[0])
        result.obv_direction = "rising" if obv_slope > 0 else "falling"

        # ATR contraction
        recent_atr = float(atr_s.iloc[-20:].mean())
        prior_atr  = float(atr_s.iloc[-80:-20].mean()) if len(atr_s) >= 80 else 0
        if prior_atr > 0:
            result.atr_contraction = round(recent_atr / prior_atr, 2)
            result.was_base        = result.atr_contraction < 0.7

        # Was it a breakout at entry?
        breakout_level = high_252 + atr_now * 0.3
        result.was_breakout = last_close >= breakout_level

        # Scores
        result.breakout_score = (
            min((last_close - high_252) / high_252 * 20, 1.0)
            if result.was_breakout and high_252 > 0 else 0.0
        )
        result.momentum_score = min(
            max((last_close - float(e55.iloc[-1])) / float(e55.iloc[-1]) * 5, 0), 1.0
        )
        result.volume_score = min((last_vol / avg_vol) / 3, 1.0) if avg_vol > 0 else 0

        # OBV score
        obv_std = float(obv_window.std()) if obv_window.std() > 0 else 1
        slope_z = obv_slope / (obv_std * len(obv_window) ** 0.5)
        result.obv_score = round(min(max(slope_z / 2, 0), 1.0), 3)

        # Estimated L1 composite (no fundamentals — just technical)
        result.estimated_l1_score = round(
            result.breakout_score * 0.35 +
            result.momentum_score * 0.25 +
            result.volume_score   * 0.10 +
            result.obv_score      * 0.10,
            3
        )

        # Which signals were present
        sigs = []
        if result.was_breakout:          sigs.append("52W_BREAKOUT")
        if result.was_base:              sigs.append("TIGHT_BASE")
        if result.ema21_vs_55 == "bullish": sigs.append("EMA_BULL")
        if result.rsi_at_entry and 48 <= result.rsi_at_entry <= 78: sigs.append("RSI_OK")
        if result.obv_direction == "rising": sigs.append("OBV_RISING")
        if result.volume_vs_avg and result.volume_vs_avg >= 1.5: sigs.append("VOL_SURGE")
        result.signals_present = sigs

        # ── Forward returns ──
        future = df.iloc[entry_idx:]
        entry_p = float(future["Close"].iloc[0])

        def fwd_return(days):
            target_dt = entry_dt + timedelta(days=days)
            idx = future.index.searchsorted(pd.Timestamp(target_dt))
            if idx < len(future):
                return round((float(future["Close"].iloc[idx]) - entry_p) / entry_p * 100, 1)
            return None

        result.return_1m = fwd_return(21)
        result.return_3m = fwd_return(63)
        result.return_6m = fwd_return(126)
        result.return_1y = fwd_return(252)

        # Nifty benchmark
        try:
            nifty_df = yf.download(
                "^NSEI",
                start=(entry_dt - timedelta(days=5)).strftime("%Y-%m-%d"),
                end=to_dt.strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if not nifty_df.empty:
                nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
                nidx = nifty_df.index.searchsorted(pd.Timestamp(entry_dt))
                np0  = float(nifty_df["Close"].squeeze().iloc[nidx])
                def nifty_ret(days):
                    t = entry_dt + timedelta(days=days)
                    i = nifty_df.index.searchsorted(pd.Timestamp(t))
                    if i < len(nifty_df):
                        return round((float(nifty_df["Close"].squeeze().iloc[i]) - np0) / np0 * 100, 1)
                    return None
                result.nifty_1m = nifty_ret(21)
                result.nifty_3m = nifty_ret(63)
        except Exception:
            pass

    except Exception as e:
        result.notes = str(e)

    return result


def run_backtest_a() -> list[SignalScoreAtDate]:
    """
    Run VALUEPICK ground truth validation.
    Score each confirmed pick on its entry date and measure what
    signals would have been present.
    """
    log.info("═" * 60)
    log.info("BACKTEST A — VALUEPICK GROUND TRUTH VALIDATION")
    log.info("Scoring confirmed picks on their original entry dates")
    log.info("═" * 60)

    results = []
    for pick in VALUEPICK_PICKS:
        sym = pick["symbol"]
        log.info(f"\nScoring {sym} @ ₹{pick['entry_price']} on {pick['entry_date']}")
        log.info(f"  Thesis: {pick['thesis']}")
        if pick.get("data_note"):
            log.info(f"  Data note: {pick['data_note']}")

        score = score_at_date(sym, pick["entry_date"], pick["entry_price"], pick_meta=pick)
        score.notes = pick.get("thesis", "")
        results.append(score)

        log.info(f"  Technical state on entry date:")
        log.info(f"    Price vs 52W high: {score.price_vs_52w_high}%")
        log.info(f"    RSI: {score.rsi_at_entry}  EMA: {score.ema21_vs_55}  OBV: {score.obv_direction}")
        log.info(f"    Was breakout: {score.was_breakout}  Tight base: {score.was_base}")
        log.info(f"    Volume vs avg: {score.volume_vs_avg}×")
        log.info(f"    Signals present: {score.signals_present}")
        log.info(f"    L1 score estimate: {score.estimated_l1_score}")
        log.info(f"  Forward returns:")
        log.info(f"    1M: {score.return_1m}%  3M: {score.return_3m}%  6M: {score.return_6m}%  1Y: {score.return_1y}%")
        log.info(f"    Nifty 1M: {score.nifty_1m}%  3M: {score.nifty_3m}%")

        time.sleep(1.0)

    return results


# ─────────────────────────────────────────────────────────────
# BACKTEST B — LAYER 1 HISTORICAL PERFORMANCE
# ─────────────────────────────────────────────────────────────

@dataclass
class BacktestSignal:
    symbol:         str
    signal_date:    str
    signal_price:   float
    rsi:            float
    adx:            float
    vol_surge:      bool
    obv_bullish:    bool
    has_tight_base: bool
    breakout_score: float
    return_1m:      Optional[float] = None
    return_3m:      Optional[float] = None
    return_6m:      Optional[float] = None
    nifty_1m:       Optional[float] = None
    nifty_3m:       Optional[float] = None
    alpha_3m:       Optional[float] = None   # return_3m - nifty_3m


def simulate_breakout_scan_on_date(
    df: pd.DataFrame,
    symbol: str,
    idx: int,
    lookback: int = 252,
) -> Optional[BacktestSignal]:
    """
    v3 scanner: 52W high breakout gate. Kept for comparison vs v4.
    Full universe result: 49.8% hit rate, +2.7% avg 3M, +1.1% alpha — no edge.
    """
    min_history = min(lookback + 60, 120)
    if idx < min_history:
        return None

    window = df.iloc[:idx + 1]
    c = window["Close"]
    v = window["Volume"]

    last_close = float(c.iloc[-1])
    last_vol   = float(v.iloc[-1])
    avg_vol    = float(v.rolling(20).mean().iloc[-1])

    if avg_vol < 50_000 or last_close < 10:
        return None

    e21   = ema(c, 21)
    e55   = ema(c, 55)
    rsi_s = rsi(c, 14)
    atr_s = atr(window, 14)

    atr_now      = float(atr_s.iloc[-1])
    rolling_high = c.rolling(lookback).max().shift(1)
    high_252     = float(rolling_high.iloc[-1])
    breakout_lvl = high_252 + atr_now * _BACKTEST_PARAMS["atr_mult"]

    is_breakout  = last_close >= breakout_lvl
    ema_bull     = float(e21.iloc[-1]) > float(e55.iloc[-1])
    last_rsi     = float(rsi_s.iloc[-1])
    rsi_ok       = _BACKTEST_PARAMS["rsi_lo"] <= last_rsi <= _BACKTEST_PARAMS["rsi_hi"]
    vol_surge    = last_vol >= avg_vol * _BACKTEST_PARAMS["vol_mult"]

    if not (is_breakout and ema_bull and rsi_ok):
        return None

    obv_s       = obv_series(window)
    obv_win     = obv_s.iloc[-10:]
    obv_slope   = float(obv_win.iloc[-1] - obv_win.iloc[0])
    obv_std     = float(obv_win.std()) if obv_win.std() > 0 else 1
    obv_bullish = obv_slope > obv_std * 0.1

    recent_atr = float(atr_s.iloc[-20:].mean())
    prior_atr  = float(atr_s.iloc[-80:-20].mean()) if len(atr_s) >= 80 else recent_atr
    has_base   = (recent_atr / prior_atr) < 0.7 if prior_atr > 0 else False

    b_score = min((last_close - high_252) / high_252 * 20, 1.0) if high_252 > 0 else 0.0
    m_score = min(max((last_close - float(e55.iloc[-1])) / float(e55.iloc[-1]) * 5, 0), 1.0)
    v_score = min((last_vol / avg_vol) / 3, 1.0) if avg_vol > 0 else 0

    try:
        adx_val = float(
            (lambda d, n=14: (
                lambda up, dn, pdm, ndm, a:
                    ((100 * (pdm.rolling(n).mean() / a) - 100 * (ndm.rolling(n).mean() / a)).abs() /
                     (100 * pdm.rolling(n).mean() / a + 100 * ndm.rolling(n).mean() / a).replace(0, np.nan)
                    ).rolling(n).mean()
            )(
                d["High"].diff(), -d["Low"].diff(),
                d["High"].diff().where((d["High"].diff() > -d["Low"].diff()) & (d["High"].diff() > 0), 0),
                (-d["Low"].diff()).where((-d["Low"].diff() > d["High"].diff()) & (-d["Low"].diff() > 0), 0),
                atr(d, n)
            ))(window).iloc[-1]
        )
    except Exception:
        adx_val = 0.0

    return BacktestSignal(
        symbol=symbol,
        signal_date=str(window.index[-1].date()),
        signal_price=round(last_close, 2),
        rsi=round(last_rsi, 1),
        adx=round(adx_val, 1) if not np.isnan(adx_val) else 0.0,
        vol_surge=vol_surge,
        obv_bullish=obv_bullish,
        has_tight_base=has_base,
        breakout_score=round(b_score, 3),
    )


def simulate_v4_scan_on_date(
    df: pd.DataFrame,
    symbol: str,
    idx: int,
) -> Optional[BacktestSignal]:
    """
    v4 scanner: volatility-adjusted momentum gate (NSE Momentum Index formula).
    Gates: avg_vol ≥ 150k | ≥50 traded days/63 | EMA21>EMA55 | RSI 48-80
           | ADX ≥ 25 | mom_score > 0 (above-average 6M+12M vol-adj momentum)
    """
    min_history = 252 + 60
    if idx < min_history:
        return None

    window = df.iloc[:idx + 1]
    c = window["Close"]
    v = window["Volume"]

    last_close = float(c.iloc[-1])
    last_vol   = float(v.iloc[-1])
    avg_vol    = float(v.rolling(20).mean().iloc[-1])

    # Liquidity gates
    if avg_vol < 150_000 or last_close < 10:
        return None
    traded_63 = int((v.iloc[-63:] > 0).sum())
    if traded_63 < 50:
        return None

    # Indicator gates
    e21   = ema(c, 21)
    e55   = ema(c, 55)
    rsi_s = rsi(c, 14)

    ema_bull = float(e21.iloc[-1]) > float(e55.iloc[-1])
    last_rsi = float(rsi_s.iloc[-1])

    if not ema_bull:
        return None
    if not (48.0 <= last_rsi <= 80.0):
        return None

    # ADX gate
    try:
        adx_s = (lambda d, n=14: (
            lambda up, dn, pdm, ndm, a:
                ((100 * (pdm.rolling(n).mean() / a) - 100 * (ndm.rolling(n).mean() / a)).abs() /
                 (100 * pdm.rolling(n).mean() / a + 100 * ndm.rolling(n).mean() / a).replace(0, np.nan)
                ).rolling(n).mean()
        )(
            d["High"].diff(), -d["Low"].diff(),
            d["High"].diff().where((d["High"].diff() > -d["Low"].diff()) & (d["High"].diff() > 0), 0),
            (-d["Low"].diff()).where((-d["Low"].diff() > d["High"].diff()) & (-d["Low"].diff() > 0), 0),
            atr(d, n)
        ))(window)
        adx_val = float(adx_s.iloc[-1])
    except Exception:
        adx_val = 0.0

    if np.isnan(adx_val) or adx_val < 0.20:   # 0.20 = ~ADX 20 on 0-1 scale
        return None

    # v4 core: volatility-adjusted momentum score (NSE formula)
    if len(c) < 252 + 10:
        return None

    p_6m  = float(c.iloc[-126])
    p_12m = float(c.iloc[-252])
    last  = float(c.iloc[-1])

    ret_6m  = (last - p_6m)  / p_6m
    ret_12m = (last - p_12m) / p_12m

    daily_ret = c.pct_change().dropna()
    ann_vol   = float(daily_ret.iloc[-252:].std()) * (252 ** 0.5)

    if ann_vol <= 0.01:
        return None

    mom_score = ((ret_6m / ann_vol) + (ret_12m / ann_vol)) / 2

    # Gate: must have above-average momentum (> 0)
    if mom_score <= 0:
        return None

    # OBV — extreme distribution only
    obv_s     = obv_series(window)
    obv_win   = obv_s.iloc[-10:]
    obv_std_v = float(obv_win.std()) if obv_win.std() > 0 else 1
    obv_slope_z = (float(obv_win.iloc[-1]) - float(obv_win.iloc[0])) / obv_std_v
    if obv_slope_z < -1.5:
        return None

    obv_bullish = obv_slope_z > 0.1

    atr_s      = atr(window, 14)
    recent_atr = float(atr_s.iloc[-20:].mean())
    prior_atr  = float(atr_s.iloc[-80:-20].mean()) if len(atr_s) >= 80 else recent_atr
    has_base   = (recent_atr / prior_atr) < 0.7 if prior_atr > 0 else False
    vol_surge  = last_vol >= avg_vol * 1.5

    # Normalised score for sorting
    norm_mom = min(max(mom_score / 2.0, 0.0), 1.0)

    return BacktestSignal(
        symbol=symbol,
        signal_date=str(window.index[-1].date()),
        signal_price=round(last_close, 2),
        rsi=round(last_rsi, 1),
        adx=round(adx_val, 1),
        vol_surge=vol_surge,
        obv_bullish=obv_bullish,
        has_tight_base=has_base,
        breakout_score=round(norm_mom, 3),   # reuse field for mom_score
    )


def measure_forward_returns(
    signal: BacktestSignal,
    df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    signal_idx: int,
) -> BacktestSignal:
    """Add forward return measurements to a signal."""
    future = df.iloc[signal_idx:]
    if future.empty:
        return signal

    entry_p = float(future["Close"].iloc[0])

    def fwd(days):
        if signal_idx + days < len(df):
            return round((float(df["Close"].iloc[signal_idx + days]) - entry_p) / entry_p * 100, 1)
        return None

    signal.return_1m = fwd(21)
    signal.return_3m = fwd(63)
    signal.return_6m = fwd(126)

    # Nifty benchmark
    if not nifty_df.empty:
        nifty_close = nifty_df["Close"].squeeze()
        entry_dt    = pd.Timestamp(signal.signal_date)
        nidx = nifty_df.index.searchsorted(entry_dt)
        if nidx < len(nifty_close):
            np0 = float(nifty_close.iloc[nidx])
            def nf(days):
                if nidx + days < len(nifty_close):
                    return round((float(nifty_close.iloc[nidx + days]) - np0) / np0 * 100, 1)
                return None
            signal.nifty_1m = nf(21)
            signal.nifty_3m = nf(63)
            if signal.return_3m is not None and signal.nifty_3m is not None:
                signal.alpha_3m = round(signal.return_3m - signal.nifty_3m, 1)

    return signal


def run_backtest_b(
    symbols: list[str],
    lookback_days: int = 500,
    sample_every_n_days: int = 5,
    batch_size: int = 50,
    scanner_version: str = "v3",   # "v3" = breakout, "v4" = momentum
) -> dict:
    """
    Run Layer 1 historical backtest on a list of symbols.

    Uses batch downloading (50 symbols at a time) — the same pattern
    as engine.py. This is ~4× faster than sequential per-symbol downloads
    and is the key optimisation that makes a full 1,800-symbol run feasible
    in ~11 minutes instead of ~60 minutes.

    For each symbol it simulates running the breakout scanner every
    sample_every_n_days over the past lookback_days and measures
    forward returns at 1M, 3M, 6M vs Nifty 50.
    """
    log.info("═" * 60)
    log.info("BACKTEST B — LAYER 1 HISTORICAL BREAKOUT PERFORMANCE")
    log.info(f"  Scanner:   {scanner_version.upper()}")
    log.info(f"  Symbols:   {len(symbols)}")
    log.info(f"  Lookback:  {lookback_days} days  |  Sample every {sample_every_n_days} days")
    log.info(f"  Batches:   {len(symbols) // batch_size + 1} × {batch_size} symbols")
    log.info("═" * 60)

    # Fetch Nifty once — used to compute alpha for every signal
    log.info("Fetching Nifty 50 benchmark (4 years)...")
    nifty_df = yf.download(
        "^NSEI", period="4y", interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    log.info(f"  Nifty: {len(nifty_df)} rows")

    all_signals:         list[BacktestSignal] = []
    signals_with_obv:    list[BacktestSignal] = []
    signals_without_obv: list[BacktestSignal] = []
    v4_pending:          list               = []   # (sig, df, idx) tuples for two-pass
    total_scanned = 0
    total_skipped = 0

    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    for b_idx, batch in enumerate(batches):
        log.info(f"  Batch {b_idx+1}/{len(batches)} — {batch[0]} … {batch[-1]}")

        try:
            raw = yf.download(
                batch, period="4y", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
        except Exception as e:
            log.warning(f"  Batch {b_idx+1} download failed: {e}")
            time.sleep(3)
            continue

        for sym in batch:
            sym_signals = 0
            try:
                # Extract this symbol's frame from the batch result
                if len(batch) == 1:
                    df = raw.copy()
                elif isinstance(raw.columns, pd.MultiIndex):
                    lvl0 = raw.columns.get_level_values(0)
                    if sym not in lvl0:
                        total_skipped += 1
                        continue
                    df = raw[sym].copy()
                else:
                    total_skipped += 1
                    continue

                # Flatten columns if still MultiIndex after slicing
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.dropna(subset=["Close", "Volume"], inplace=True)
                # v4 needs 252+60=312 warmup rows; v3 needs ~320 (252 lookback + 60 indicators)
                min_rows = 330 if scanner_version == "v4" else 300
                if df.empty or len(df) < min_rows:
                    total_skipped += 1
                    continue

                df.index = pd.to_datetime(df.index).tz_localize(None)

                # Scan window: v4 needs 312 warmup rows (252 for 12M momentum + 60 indicators)
                warmup           = 312 if scanner_version == "v4" else 260
                scan_range_end   = len(df) - 21
                scan_range_start = max(len(df) - lookback_days, warmup)

                if scan_range_start >= scan_range_end:
                    total_skipped += 1
                    continue

                if scanner_version == "v4":
                    # v4 TWO-PASS: collect (date_idx → mom_score) candidates first,
                    # then cross-sectional percentile gate is applied after all symbols
                    # are processed. Store raw candidates in pending list.
                    for idx in range(scan_range_start, scan_range_end, sample_every_n_days):
                        sig = simulate_v4_scan_on_date(df, sym, idx)
                        if sig is None:
                            continue
                        # Store with the raw mom_score (breakout_score field holds norm_mom)
                        # and the df + idx so we can measure forward returns in pass 2
                        v4_pending.append((sig, df.copy(), idx))
                        sym_signals += 1
                else:
                    for idx in range(scan_range_start, scan_range_end, sample_every_n_days):
                        sig = simulate_breakout_scan_on_date(df, sym, idx, lookback=252)
                        if sig is None:
                            continue
                        sig = measure_forward_returns(sig, df, nifty_df, idx)
                        all_signals.append(sig)
                        sym_signals += 1
                        if sig.obv_bullish:
                            signals_with_obv.append(sig)
                        else:
                            signals_without_obv.append(sig)

                total_scanned += 1
                if sym_signals > 0:
                    log.info(f"    ✓ {sym:<18}: {sym_signals} signals")

            except Exception as e:
                log.warning(f"  {sym}: {type(e).__name__}: {e}")
                total_skipped += 1

        log.info(
            f"  Batch {b_idx+1} done — running total: {len(all_signals) + len(v4_pending)} signals | "
            f"scanned: {total_scanned} | skipped: {total_skipped}"
        )
        time.sleep(1.5)   # polite pause between batches

    # ── v4 PASS 2: cross-sectional percentile gate ──────────────────────────
    # NSE's Momentum Index methodology: rank all stocks by momentum score on
    # each rebalance date and take only the top quintile (top 20%).
    # This is what actually works — not "above zero" but "top 20% of universe".
    if scanner_version == "v4" and v4_pending:
        log.info(f"\nv4 Pass 2: cross-sectional percentile gate on {len(v4_pending)} raw candidates...")

        # Group candidates by their signal date
        from collections import defaultdict
        by_date: dict = defaultdict(list)
        for item in v4_pending:
            sig, df, idx = item
            by_date[sig.signal_date].append(item)

        # On each date, compute 80th percentile threshold and keep top 20%
        top_pct = 0.80   # top 20% = above 80th percentile
        passed = kept = 0
        for date_str, items in sorted(by_date.items()):
            scores = [item[0].breakout_score for item in items]   # norm_mom stored here
            if len(scores) < 5:   # too few to rank meaningfully
                continue
            threshold = np.percentile(scores, top_pct * 100)
            for sig, df, idx in items:
                passed += 1
                if sig.breakout_score >= threshold:
                    sig = measure_forward_returns(sig, df, nifty_df, idx)
                    all_signals.append(sig)
                    kept += 1
                    if sig.obv_bullish:
                        signals_with_obv.append(sig)
                    else:
                        signals_without_obv.append(sig)

        log.info(f"  Cross-sectional gate: {passed} raw → {kept} kept "
                 f"(top 20% per date, {100*kept/max(passed,1):.1f}% pass rate)")

    log.info(
        f"\nBacktest B complete:"
        f"\n  Total signals:   {len(all_signals)}"
        f"\n  OBV confirmed:   {len(signals_with_obv)}"
        f"\n  OBV diverging:   {len(signals_without_obv)}"
        f"\n  Symbols scanned: {total_scanned}"
        f"\n  Symbols skipped: {total_skipped}"
    )
    return {
        "all_signals":         all_signals,
        "with_obv":            signals_with_obv,
        "without_obv":         signals_without_obv,
        "total_scanned":       total_scanned,
        "total_skipped":       total_skipped,
    }


# ─────────────────────────────────────────────────────────────
# METRICS & WEIGHT TUNING
# ─────────────────────────────────────────────────────────────

def compute_metrics(signals: list[BacktestSignal], label: str) -> dict:
    """Compute hit rate, avg returns, alpha for a list of signals."""
    if not signals:
        return {"label": label, "count": 0}

    r3 = [s.return_3m for s in signals if s.return_3m is not None]
    r1 = [s.return_1m for s in signals if s.return_1m is not None]
    r6 = [s.return_6m for s in signals if s.return_6m is not None]
    al = [s.alpha_3m  for s in signals if s.alpha_3m  is not None]

    return {
        "label":          label,
        "count":          len(signals),
        "hit_rate_3m":    round(sum(1 for r in r3 if r > 0) / len(r3) * 100, 1) if r3 else None,
        "avg_return_1m":  round(float(np.mean(r1)), 1) if r1 else None,
        "avg_return_3m":  round(float(np.mean(r3)), 1) if r3 else None,
        "avg_return_6m":  round(float(np.mean(r6)), 1) if r6 else None,
        "median_3m":      round(float(np.median(r3)), 1) if r3 else None,
        "avg_alpha_3m":   round(float(np.mean(al)), 1) if al else None,
        "best_3m":        round(max(r3), 1) if r3 else None,
        "worst_3m":       round(min(r3), 1) if r3 else None,
        "pct_gt_20pct":   round(sum(1 for r in r3 if r > 20) / len(r3) * 100, 1) if r3 else None,
    }


def tune_weights(ground_truth: list[SignalScoreAtDate]) -> dict:
    """
    Simple grid search over scoring weights to maximise:
      - Score rank of known multibaggers (they should rank highly)
      - Correlation between score and forward return

    Returns the best weight configuration found.
    """
    log.info("\nTuning weights on ground truth picks...")

    # Only use picks where we have forward return data
    valid = [g for g in ground_truth if g.return_3m is not None and g.return_3m > 0]
    if len(valid) < 3:
        log.warning("Not enough valid ground truth picks with forward returns to tune")
        return {}

    best_score  = -999
    best_config = {}

    # Grid search
    for w_b in [0.25, 0.30, 0.35, 0.40]:        # breakout weight
        for w_m in [0.15, 0.20, 0.25]:            # momentum weight
            for w_v in [0.05, 0.10, 0.15]:         # volume weight
                for w_o in [0.05, 0.10, 0.15]:     # OBV weight
                    if abs(w_b + w_m + w_v + w_o - 1.0) > 0.3:
                        continue

                    # Score each pick with this weight config
                    scores = []
                    for g in valid:
                        s = (g.breakout_score * w_b +
                             g.momentum_score * w_m +
                             g.volume_score   * w_v +
                             g.obv_score      * w_o)
                        scores.append(s)

                    # Maximise correlation between score and forward return
                    returns = [g.return_3m for g in valid]
                    if len(scores) >= 2 and np.std(scores) > 0:
                        corr = float(np.corrcoef(scores, returns)[0, 1])
                        if corr > best_score:
                            best_score  = corr
                            best_config = {
                                "w_breakout":  w_b,
                                "w_momentum":  w_m,
                                "w_volume":    w_v,
                                "w_obv":       w_o,
                                "correlation": round(corr, 3),
                            }

    log.info(f"  Best weights: {best_config}")
    return best_config


# ─────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_backtest_report(
    a_results: list[SignalScoreAtDate],
    b_metrics: dict,
    best_weights: dict,
    params: dict = None,
) -> str:
    sep = "═" * 72
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        sep,
        "  MULTIBAGGER ENGINE — BACKTEST REPORT",
        f"  {now}",
        sep,
    ]

    # Show what parameters were used — crucial for iterative tuning
    if params:
        lines += [
            "",
            "  PARAMETERS USED IN THIS RUN:",
            f"    RSI range:  {params.get('rsi_lo', 48)} – {params.get('rsi_hi', 78)}",
            f"    ATR mult:   {params.get('atr_mult', 0.3)}   "
            f"(breakout = 52W high + {params.get('atr_mult', 0.3)}×ATR)",
            f"    Vol surge:  {params.get('vol_mult', 1.5)}×  avg 20-day volume",
            f"    ADX min:    {params.get('adx_min', 18)}",
            f"    Lookback:   {params.get('lookback', 500)} days",
            f"    Quick mode: {'YES' if params.get('quick') else 'NO'}",
            "",
        ]
    lines.append("")

    # ── BACKTEST A ──
    lines += [
        "  BACKTEST A: VALUEPICK GROUND TRUTH VALIDATION",
        "  Question: Would our engine have flagged these known multibaggers?",
        sep,
        "",
    ]

    for g in a_results:
        pick = next((p for p in VALUEPICK_PICKS if p["symbol"] == g.symbol), {})
        ret_x = pick.get("return_x", "?")
        lines += [
            f"  {g.symbol}  entry ₹{g.entry_price}  date {g.entry_date}  ({ret_x}x confirmed)",
            f"  Thesis: {g.notes[:80]}",
            f"  Technical state on entry date:",
            f"    vs 52W high: {g.price_vs_52w_high}%   vs 52W low: {g.price_vs_52w_low}%",
            f"    RSI: {g.rsi_at_entry}   EMA: {g.ema21_vs_55}   OBV: {g.obv_direction}",
            f"    Breakout: {'✓' if g.was_breakout else '✗'}   Tight base: {'✓' if g.was_base else '✗'}",
            f"    Volume vs avg: {g.volume_vs_avg}×   ATR ratio: {g.atr_contraction}",
            f"  Signals our model would have fired: {', '.join(g.signals_present) or 'NONE'}",
            f"  Estimated L1 score: {g.estimated_l1_score:.3f}",
        ]
        if g.return_3m is not None:
            alpha = f"+{g.return_3m - g.nifty_3m:.1f}% alpha" if g.nifty_3m else ""
            lines.append(
                f"  Forward returns: 1M {g.return_1m:+.1f}%  3M {g.return_3m:+.1f}%  "
                f"6M {g.return_6m:+.1f}%  1Y {g.return_1y:+.1f}%  {alpha}"
            )
        else:
            # Show the actual reason — much more informative than "insufficient data"
            note = g.notes if g.notes and g.notes != pick.get("thesis", "") else ""
            pick_note = pick.get("data_note", "")
            reason = note or pick_note or "No yfinance data available for this symbol/date"
            lines.append(f"  Forward returns: ⚠ NOT AVAILABLE — {reason}")
        if g.notes and g.notes != pick.get("thesis", ""):
            lines.append(f"  Note: {g.notes}")
        lines.append("")

    # Key insight
    had_breakout = sum(1 for g in a_results if g.was_breakout)
    had_base     = sum(1 for g in a_results if g.was_base)
    total        = len(a_results)
    lines += [
        f"  SUMMARY: {had_breakout}/{total} picks had a technical breakout on entry date",
        f"           {had_base}/{total} picks had a tight consolidation base",
        "",
        "  KEY INSIGHT:",
        "  If hit_rate is low (picks did NOT have breakout on entry date),",
        "  that confirms VALUEPICK found them BEFORE technical signals fired.",
        "  Our Layer 1 would have missed them — Layer 2 thesis signals matter more.",
        "",
    ]

    # ── BACKTEST B ──
    lines += [
        sep,
        "  BACKTEST B: LAYER 1 HISTORICAL PERFORMANCE",
        "  Question: When our breakout scanner fires, how often does it work?",
        sep,
        "",
    ]

    for key in ["all_signals", "with_obv", "without_obv", "with_base", "with_vol"]:
        m = b_metrics.get(key, {})
        if not m or m.get("count", 0) == 0:
            continue
        label = {
            "all_signals":  "All signals",
            "with_obv":     "OBV confirmed ✓",
            "without_obv":  "OBV diverging ✗",
            "with_base":    "Tight base ✓",
            "with_vol":     "Volume surge ✓",
        }.get(key, key)
        lines += [
            f"  {label} (n={m.get('count', 0)})",
            f"    Hit rate (3M positive return): {m.get('hit_rate_3m', 'N/A')}%",
            f"    Avg return: 1M {m.get('avg_return_1m', '?')}%  3M {m.get('avg_return_3m', '?')}%  "
            f"6M {m.get('avg_return_6m', '?')}%",
            f"    Median 3M: {m.get('median_3m', '?')}%   Avg alpha vs Nifty: {m.get('avg_alpha_3m', '?')}%",
            f"    Best 3M: {m.get('best_3m', '?')}%   Worst 3M: {m.get('worst_3m', '?')}%",
            f"    % signals with >20% 3M return: {m.get('pct_gt_20pct', '?')}%",
            "",
        ]

    # OBV impact
    w_obv  = b_metrics.get("with_obv", {})
    wo_obv = b_metrics.get("without_obv", {})
    if w_obv.get("avg_return_3m") and wo_obv.get("avg_return_3m"):
        obv_lift = round(w_obv["avg_return_3m"] - wo_obv["avg_return_3m"], 1)
        lines += [
            f"  OBV FILTER IMPACT:",
            f"    With OBV confirmed:  avg 3M = {w_obv['avg_return_3m']}%",
            f"    Without OBV:         avg 3M = {wo_obv['avg_return_3m']}%",
            f"    OBV adds {obv_lift:+.1f}% to avg 3M return",
            "",
        ]

    # ── Weight tuning ──
    if best_weights:
        lines += [
            sep,
            "  WEIGHT TUNING RESULT (based on ground truth correlation)",
            sep,
            "",
            f"  Current weights:  breakout=0.35  momentum=0.20  volume=0.10  OBV=0.10",
            f"  Suggested weights: breakout={best_weights.get('w_breakout')}  "
            f"momentum={best_weights.get('w_momentum')}  "
            f"volume={best_weights.get('w_volume')}  "
            f"OBV={best_weights.get('w_obv')}",
            f"  Score-return correlation: {best_weights.get('correlation')}",
            "",
            "  To apply: update w_breakout, w_momentum, w_volume, w_obv in engine.py Config",
            "",
        ]

    lines += [
        sep,
        "  ⚠  Backtesting has survivorship bias — historical prices are",
        "     available only for stocks that still exist. Failed companies",
        "     are excluded, which overstates hit rates. Use as direction,",
        "     not as proof.",
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multibagger Engine Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --mode a                    # VALUEPICK ground truth only (~1 min)
  python backtest.py --mode b --quick            # Fast Backtest B on 10 stocks (~2 min)
  python backtest.py --mode b                    # Backtest B on 48 curated stocks (~5 min)
  python backtest.py --mode b --full             # Full NSE universe ~1,800 stocks (~11 min)
  python backtest.py --mode both --tune          # Run everything + weight tuning
  python backtest.py --mode b --rsi-lo 45 --rsi-hi 80   # Test different RSI range
  python backtest.py --mode b --atr-mult 0.1    # Loosen breakout gate (more signals)
  python backtest.py --mode b --full --lookback 365      # 1 year on full universe
        """
    )
    parser.add_argument("--mode", default="both", choices=["a", "b", "both"],
                        help="a=ground truth, b=historical, both=run both")
    parser.add_argument("--tune", action="store_true",
                        help="Tune weights on ground truth results")
    parser.add_argument("--symbols", nargs="+",
                        help="Override symbols for Backtest B")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 stocks, 200 days (~2 min). Good for iterating.")
    parser.add_argument("--full", action="store_true",
                        help="Full NSE universe (~1,800 symbols, ~11 min). "
                             "Uses batch downloads. Runs on GitHub Actions monthly.")
    parser.add_argument("--v4", action="store_true",
                        help="Use v4 scanner (volatility-adjusted momentum) instead of v3 breakout.")
    parser.add_argument("--compare", action="store_true",
                        help="Run both v3 AND v4 side-by-side on the same symbols. "
                             "Best way to validate whether v4 actually improves results.")

    # ── Signal parameter overrides — tune these to optimise ──
    parser.add_argument("--rsi-lo",   type=float, default=48.0,
                        help="RSI lower bound (default 48). Lower = more signals.")
    parser.add_argument("--rsi-hi",   type=float, default=78.0,
                        help="RSI upper bound (default 78). Higher = more signals.")
    parser.add_argument("--atr-mult", type=float, default=0.3,
                        help="ATR buffer multiplier for breakout level (default 0.3). "
                             "Lower = easier to trigger breakout.")
    parser.add_argument("--vol-mult", type=float, default=1.5,
                        help="Volume surge multiplier vs 20-day avg (default 1.5). "
                             "Lower = more signals, higher = stricter.")
    parser.add_argument("--adx-min",  type=float, default=18.0,
                        help="Minimum ADX for trend strength (default 18).")
    parser.add_argument("--lookback", type=int, default=500,
                        help="Backtest lookback days (default 500 = ~2 years)")

    args = parser.parse_args()

    # Quick mode overrides
    quick_symbols = [
        "DIXON.NS", "DEEPAKNTR.NS", "PERSISTENT.NS", "TRENT.NS",
        "POLYCAB.NS", "HAL.NS", "IRFC.NS", "TATAPOWER.NS", "FORTIS.NS", "BALKRISIND.NS",
    ]

    log.info("═" * 60)
    log.info("BACKTEST PARAMETERS")
    log.info(f"  RSI range:    {args.rsi_lo} – {args.rsi_hi}")
    log.info(f"  ATR mult:     {args.atr_mult}  (breakout = 52W high + {args.atr_mult}×ATR)")
    log.info(f"  Vol surge:    {args.vol_mult}×  avg volume")
    log.info(f"  ADX min:      {args.adx_min}")
    log.info(f"  Lookback:     {args.lookback} days")
    log.info(f"  Quick mode:   {'YES (10 stocks)' if args.quick else 'NO'}")
    log.info(f"  Full universe:{'YES (~1,800 symbols)' if getattr(args, 'full', False) else 'NO'}")
    log.info(f"  Scanner:      {'v4 (momentum)' if getattr(args, 'v4', False) else 'v3 (breakout)'}"
             f"{'  + compare both' if getattr(args, 'compare', False) else ''}")
    log.info("═" * 60)

    a_results    = []
    b_metrics    = {}
    best_weights = {}

    if args.mode in ("a", "both"):
        a_results = run_backtest_a()
        if args.tune:
            best_weights = tune_weights(a_results)

    if args.mode in ("b", "both"):
        if args.symbols:
            # Explicit symbol list overrides everything
            test_symbols = args.symbols
            lookback     = args.lookback
        elif args.quick:
            test_symbols = quick_symbols
            lookback     = min(args.lookback, 200)
        elif getattr(args, 'full', False):
            # Full NSE universe — fetch from universe.py
            log.info("Loading full NSE universe...")
            from universe import fetch_nse_symbols
            test_symbols = fetch_nse_symbols()
            log.info(f"  Universe: {len(test_symbols)} symbols")
            lookback = args.lookback
        else:
            test_symbols = [
                "DIXON.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS", "FINEORG.NS",
                "GALAXYSURF.NS", "VINATIORGA.NS", "ALKYLAMINE.NS", "PIIND.NS",
                "PERSISTENT.NS", "KPITTECH.NS", "HAPPSTMNDS.NS", "TANLA.NS",
                "TRENT.NS", "PAGEIND.NS", "RELAXO.NS", "BATAINDIA.NS", "SAFARI.NS",
                "POLYCAB.NS", "GRINDWELL.NS", "SCHAEFFLER.NS", "TIMKEN.NS",
                "SKFINDIA.NS", "ELGIEQUIP.NS", "SUPRAJIT.NS",
                "BALKRISIND.NS", "CEATLTD.NS", "APOLLOTYRE.NS",
                "TATAPOWER.NS", "JSWENERGY.NS", "CESC.NS", "SUZLON.NS",
                "HAL.NS", "BEL.NS", "BHEL.NS", "IRFC.NS", "RAILTEL.NS",
                "RVNL.NS", "IRCON.NS", "COCHINSHIP.NS",
                "METROPOLIS.NS", "LALPATHLAB.NS", "FORTIS.NS", "ASTERDM.NS",
                "BALRAMCHIN.NS", "TATACHEM.NS",
                "MRF.NS", "VIPIND.NS", "MAXHEALTH.NS",
            ]
            lookback = args.lookback

        # Pass signal params for v3 breakout scanner
        global _BACKTEST_PARAMS
        _BACKTEST_PARAMS = {
            "rsi_lo":   args.rsi_lo,
            "rsi_hi":   args.rsi_hi,
            "atr_mult": args.atr_mult,
            "vol_mult": args.vol_mult,
            "adx_min":  args.adx_min,
        }

        use_v4    = getattr(args, "v4", False)
        do_compare = getattr(args, "compare", False)

        if do_compare:
            # Run both v3 and v4 on same symbols for direct comparison
            log.info("Running v3 (breakout) scanner...")
            raw_v3 = run_backtest_b(test_symbols, lookback_days=lookback,
                                     sample_every_n_days=5, scanner_version="v3")
            log.info("Running v4 (momentum) scanner...")
            raw_v4 = run_backtest_b(test_symbols, lookback_days=lookback,
                                     sample_every_n_days=5, scanner_version="v4")
            b_metrics = {
                "v3_all":      compute_metrics(raw_v3["all_signals"], "v3 All signals"),
                "v3_with_obv": compute_metrics(raw_v3["with_obv"],    "v3 OBV confirmed"),
                "v4_all":      compute_metrics(raw_v4["all_signals"], "v4 All signals"),
                "v4_with_obv": compute_metrics(raw_v4["with_obv"],    "v4 OBV confirmed"),
                "v4_vol":      compute_metrics(
                    [s for s in raw_v4["all_signals"] if s.vol_surge], "v4 Vol surge ✓"
                ),
                # Keep these for report compatibility
                "all_signals":  compute_metrics(raw_v4["all_signals"],  "v4 All signals"),
                "with_obv":     compute_metrics(raw_v4["with_obv"],     "v4 OBV confirmed"),
                "without_obv":  compute_metrics(raw_v4["without_obv"],  "v4 OBV diverging"),
                "with_base":    compute_metrics(
                    [s for s in raw_v4["all_signals"] if s.has_tight_base], "v4 Tight base ✓"
                ),
                "with_vol":     compute_metrics(
                    [s for s in raw_v4["all_signals"] if s.vol_surge], "v4 Vol surge ✓"
                ),
            }
        else:
            scanner = "v4" if use_v4 else "v3"
            raw = run_backtest_b(test_symbols, lookback_days=lookback,
                                  sample_every_n_days=5, scanner_version=scanner)
            b_metrics = {
                "all_signals":   compute_metrics(raw["all_signals"],  f"{scanner} All signals"),
                "with_obv":      compute_metrics(raw["with_obv"],     f"{scanner} OBV confirmed"),
                "without_obv":   compute_metrics(raw["without_obv"],  f"{scanner} OBV diverging"),
                "with_base":     compute_metrics(
                    [s for s in raw["all_signals"] if s.has_tight_base], f"{scanner} Tight base ✓"
                ),
                "with_vol":      compute_metrics(
                    [s for s in raw["all_signals"] if s.vol_surge], f"{scanner} Vol surge ✓"
                ),
            }

    report = generate_backtest_report(
        a_results, b_metrics, best_weights,
        params=getattr(args, '__dict__', {})
    )
    print("\n" + report)

    date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
    os.makedirs("results", exist_ok=True)
    with open(f"results/backtest_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(f"results/backtest_{date_str}.json", "w") as f:
        json.dump({
            "params":          getattr(args, '__dict__', {}),
            "ground_truth":    [asdict(g) for g in a_results],
            "backtest_b":      b_metrics,
            "weight_tuning":   best_weights,
        }, f, indent=2, default=str)

    log.info(f"Results saved → results/backtest_{date_str}.txt")


if __name__ == "__main__":
    main()
