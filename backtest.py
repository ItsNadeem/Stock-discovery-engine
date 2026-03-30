"""
backtest.py — Signal Validation & Backtester v5

THREE backtests:

BACKTEST A — VALUEPICK GROUND TRUTH VALIDATION (unchanged, always needed)
  Score confirmed multibagger picks at their original entry dates.
  Tells us: would the engine have found them? Which signals fired?
  This is the only backtest that validates the DNA scorer.
  Run after any scoring change to check we don't regress on known picks.

BACKTEST B — LAYER 1 HISTORICAL BREAKOUT PERFORMANCE
  v3 (52W breakout) vs v4 (volatility-adjusted momentum) side-by-side.
  VERDICT from prior runs: v4 modestly outperforms v3 but neither has
  strong edge in a bear market. KEEP BOTH for comparison — the spread
  between v3 and v4 tells us how much the momentum filter is helping.
  v3 is NOT removed — it's the baseline. Without it, we can't measure
  whether v4 is actually better.

BACKTEST C — DNA SCORER VALIDATION (NEW for v5)
  Run the DNA scorer on each confirmed VALUEPICK pick at its entry date
  and measure the component scores. Answers the question we couldn't
  answer before: does the DNA model score known multibaggers highly?
  If DNA score < 0.60 for a confirmed pick, something is wrong with the
  model. This replaces the old weight-tuning logic (--tune flag) which
  was tuning on only 6 data points and had no out-of-sample validity.

CLI:
  python backtest.py --mode a          # Ground truth validation only
  python backtest.py --mode b          # Layer 1 historical (v3 vs v4)
  python backtest.py --mode c          # DNA scorer validation (new)
  python backtest.py --mode both       # A + B (default, ~10 min)
  python backtest.py --mode all        # A + B + C (full, ~15 min)
  python backtest.py --full            # Full 1,800-symbol universe (B only, ~45 min)
  python backtest.py --v4              # B with v4 only
  python backtest.py --compare         # B with v3 vs v4 side-by-side
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_BACKTEST_PARAMS: dict = {
    "rsi_lo": 48.0, "rsi_hi": 78.0,
    "atr_mult": 0.3, "vol_mult": 1.5, "adx_min": 18.0,
}

# ─────────────────────────────────────────────────────────────
# GROUND TRUTH DATASET — VALUEPICK CONFIRMED PICKS
# ─────────────────────────────────────────────────────────────

VALUEPICK_PICKS = [
    {
        "symbol": "PAUSHAKLTD.BO", "symbol_alt": "PAUSHAKLTD.NS",
        "entry_price": 74.0, "entry_date": "2011-03-15",
        "peak_price": 10000.0, "return_x": 140.0,
        "thesis": "Alembic group specialty chemical, capacity expansion, debt-free",
        "key_signals": ["trusted_group", "capex", "low_pe", "debt_free"],
        "data_note": "BSE-only until Dec 2025.",
    },
    {
        "symbol": "COSMOFERR.BO", "symbol_alt": "COSMOFERR.NS",
        "entry_price": 13.0, "entry_date": "2011-06-01",
        "peak_price": 500.0, "return_x": 13.0,
        "thesis": "Cosmo Films group, soft ferrite, EV tailwind",
        "key_signals": ["trusted_group", "low_pe", "export_growth", "capacity_expansion"],
        "data_note": "Entry date approximate.",
    },
    {
        "symbol": "TASTYBITE.BO", "symbol_alt": "TASTYBITE.NS",
        "entry_price": 165.0, "entry_date": "2010-02-27",
        "peak_price": 9420.0, "return_x": 57.0,
        "thesis": "Organic food, export-led, US market penetration",
        "key_signals": ["export_growth", "low_pe", "promoter_confidence"],
        "data_note": "Pre-2015 Yahoo data patchy.",
    },
    {
        "symbol": "EKINDIA.NS", "symbol_alt": "EKINDIA.BO",
        "entry_price": 162.0, "entry_date": "2021-04-10",
        "peak_price": 10000.0, "return_x": 66.0,
        "thesis": "First listed carbon credit company, 90%+ overseas revenue",
        "key_signals": ["first_in_niche", "export_growth", "high_promoter", "low_pe"],
        "data_note": "Listed Apr 2021 — minimal history at entry.",
    },
    {
        "symbol": "JAYKAYENTM.BO", "symbol_alt": "JAYKAYENTM.NS",
        "entry_price": 28.0, "entry_date": "2021-02-06",
        "thesis": "JK group 3D printing JV with EOS Germany, promoter pref allotment",
        "key_signals": ["trusted_group", "promoter_pref_allotment", "pivot_jv"],
        "data_note": "Extremely illiquid microcap.",
    },
    {
        "symbol": "SHANTIGEAR.NS", "symbol_alt": "SHANTIGEAR.BO",
        "entry_price": 180.0, "entry_date": "2024-06-01",
        "thesis": "Murugappa group, zero debt, steady growth, sector tailwind",
        "key_signals": ["trusted_group", "debt_free", "sector_tailwind"],
        "data_note": "Most recent pick — full data expected.",
    },
]

# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (self-contained — no engine.py imports)
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
    return (df["Volume"] * np.sign(df["Close"].diff().fillna(0))).cumsum()

# ─────────────────────────────────────────────────────────────
# BACKTEST A — VALUEPICK GROUND TRUTH
# ─────────────────────────────────────────────────────────────

@dataclass
class SignalScoreAtDate:
    symbol: str
    entry_date: str
    entry_price: float
    thesis: str = ""
    data_note: str = ""
    # Technical state
    price_vs_52w_high: Optional[float] = None
    price_vs_52w_low:  Optional[float] = None
    ema21_vs_55:       Optional[str]   = None
    rsi_at_entry:      Optional[float] = None
    obv_direction:     Optional[str]   = None
    volume_vs_avg:     Optional[float] = None
    atr_contraction:   Optional[float] = None
    was_breakout:      bool = False
    was_base:          bool = False
    signals_present:   list = field(default_factory=list)
    estimated_l1_score:float = 0.0
    # Forward returns
    return_1m:  Optional[float] = None
    return_3m:  Optional[float] = None
    return_6m:  Optional[float] = None
    return_1y:  Optional[float] = None
    nifty_1m:   Optional[float] = None
    nifty_3m:   Optional[float] = None
    notes: str = ""


def score_at_date(symbol: str, date_str: str, entry_price: float,
                  pick_meta: Optional[dict] = None) -> SignalScoreAtDate:
    result = SignalScoreAtDate(
        symbol=symbol, entry_date=date_str, entry_price=entry_price,
        thesis=pick_meta.get("thesis","") if pick_meta else "",
        data_note=pick_meta.get("data_note","") if pick_meta else "",
    )
    try:
        entry_dt = datetime.strptime(date_str, "%Y-%m-%d")
        from_dt  = entry_dt - timedelta(days=730)
        to_dt    = entry_dt + timedelta(days=400)

        def fetch(sym):
            try:
                df = yf.Ticker(sym).history(
                    start=from_dt.strftime("%Y-%m-%d"),
                    end=to_dt.strftime("%Y-%m-%d"),
                    interval="1d", auto_adjust=True,
                )
                if df is not None and not df.empty and len(df) >= 50:
                    return df
            except Exception:
                pass
            return None

        df = fetch(symbol)
        if df is None and pick_meta and pick_meta.get("symbol_alt"):
            alt = pick_meta["symbol_alt"]
            log.info(f"  {symbol}: no data — trying {alt}")
            df = fetch(alt)
            if df is not None:
                result.symbol = alt

        if df is None or df.empty:
            result.notes = f"No data. {pick_meta.get('data_note','') if pick_meta else ''}"
            return result

        df.index = pd.to_datetime(df.index).tz_localize(None)
        entry_idx = min(df.index.searchsorted(pd.Timestamp(entry_dt)), len(df) - 1)
        hist = df.iloc[:entry_idx + 1]

        if len(hist) < 60:
            result.notes = f"Only {len(hist)} rows at entry date"
            return result

        c = hist["Close"]; v = hist["Volume"]
        e21 = ema(c,21); e55 = ema(c,55)
        rsi_s = rsi(c,14); atr_s = atr(hist,14); obv_s = obv_series(hist)

        last_close = float(c.iloc[-1])
        last_vol   = float(v.iloc[-1])
        avg_vol    = float(v.rolling(20).mean().iloc[-1])
        high_252   = float(c.rolling(min(252,len(c))).max().shift(1).iloc[-1])
        low_252    = float(c.rolling(min(252,len(c))).min().iloc[-1])
        atr_now    = float(atr_s.iloc[-1])

        result.price_vs_52w_high = round((last_close-high_252)/high_252*100,1) if high_252>0 else None
        result.price_vs_52w_low  = round((last_close-low_252)/low_252*100,1)   if low_252>0  else None
        result.ema21_vs_55       = "bullish" if float(e21.iloc[-1])>float(e55.iloc[-1]) else "bearish"
        result.rsi_at_entry      = round(float(rsi_s.iloc[-1]),1)
        result.volume_vs_avg     = round(last_vol/avg_vol,2) if avg_vol>0 else None

        obv_win  = obv_s.iloc[-10:]
        obv_slope = float(obv_win.iloc[-1]-obv_win.iloc[0])
        result.obv_direction = "rising" if obv_slope>0 else "falling"

        recent_atr = float(atr_s.iloc[-20:].mean())
        prior_atr  = float(atr_s.iloc[-80:-20].mean()) if len(atr_s)>=80 else recent_atr
        if prior_atr>0:
            result.atr_contraction = round(recent_atr/prior_atr,2)
            result.was_base = result.atr_contraction < 0.7

        result.was_breakout = last_close >= (high_252 + atr_now*0.3)

        sigs = []
        if result.was_breakout:             sigs.append("52W_BREAKOUT")
        if result.was_base:                 sigs.append("TIGHT_BASE")
        if result.ema21_vs_55=="bullish":   sigs.append("EMA_BULL")
        if result.rsi_at_entry and 48<=result.rsi_at_entry<=78: sigs.append("RSI_OK")
        if result.obv_direction=="rising":  sigs.append("OBV_RISING")
        if result.volume_vs_avg and result.volume_vs_avg>=1.5:  sigs.append("VOL_SURGE")
        result.signals_present = sigs

        result.estimated_l1_score = round(
            (0.35 if result.was_breakout else 0) +
            (0.25 * min(max((last_close-float(e55.iloc[-1]))/float(e55.iloc[-1])*5,0),1)) +
            (0.10 * min(last_vol/avg_vol/3,1) if avg_vol>0 else 0) +
            (0.10 * min(max(obv_slope/(float(obv_win.std()) if obv_win.std()>0 else 1)/2,0),1)),
            3
        )

        # Forward returns
        future = df.iloc[entry_idx:]
        entry_p = float(future["Close"].iloc[0])
        def fwd(days):
            t = entry_dt + timedelta(days=days)
            i = future.index.searchsorted(pd.Timestamp(t))
            if i < len(future):
                return round((float(future["Close"].iloc[i])-entry_p)/entry_p*100,1)
            return None
        result.return_1m = fwd(21); result.return_3m = fwd(63)
        result.return_6m = fwd(126); result.return_1y = fwd(252)

        # Nifty benchmark
        try:
            ndf = yf.download("^NSEI",
                start=(entry_dt-timedelta(days=5)).strftime("%Y-%m-%d"),
                end=to_dt.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            if not ndf.empty:
                ndf.index = pd.to_datetime(ndf.index).tz_localize(None)
                nc = ndf["Close"].squeeze()
                ni = ndf.index.searchsorted(pd.Timestamp(entry_dt))
                np0 = float(nc.iloc[ni])
                def nf(days):
                    t = entry_dt + timedelta(days=days)
                    i = ndf.index.searchsorted(pd.Timestamp(t))
                    if i < len(nc): return round((float(nc.iloc[i])-np0)/np0*100,1)
                    return None
                result.nifty_1m = nf(21); result.nifty_3m = nf(63)
        except Exception:
            pass
    except Exception as e:
        result.notes = str(e)
    return result


def run_backtest_a() -> list[SignalScoreAtDate]:
    log.info("═"*60)
    log.info("BACKTEST A — VALUEPICK GROUND TRUTH VALIDATION")
    log.info("═"*60)
    results = []
    for pick in VALUEPICK_PICKS:
        log.info(f"\n{pick['symbol']} @ ₹{pick['entry_price']} on {pick['entry_date']}")
        log.info(f"  Thesis: {pick['thesis']}")
        s = score_at_date(pick["symbol"], pick["entry_date"], pick["entry_price"], pick)
        results.append(s)
        log.info(f"  52W high: {s.price_vs_52w_high}%  52W low: {s.price_vs_52w_low}%")
        log.info(f"  RSI:{s.rsi_at_entry} EMA:{s.ema21_vs_55} OBV:{s.obv_direction} Base:{s.was_base}")
        log.info(f"  Signals: {s.signals_present}")
        log.info(f"  L1 score: {s.estimated_l1_score}")
        log.info(f"  Returns: 1M:{s.return_1m}% 3M:{s.return_3m}% 6M:{s.return_6m}% 1Y:{s.return_1y}%")
        log.info(f"  Nifty:   1M:{s.nifty_1m}% 3M:{s.nifty_3m}%")
        if s.notes: log.info(f"  Note: {s.notes}")
        time.sleep(1.0)
    return results

# ─────────────────────────────────────────────────────────────
# BACKTEST C — DNA SCORER VALIDATION (NEW)
# ─────────────────────────────────────────────────────────────

def run_backtest_c() -> list[dict]:
    """
    Run the DNA scorer on each confirmed pick at its entry date.
    Uses whatever historical data is available via yfinance.
    Primary purpose: validate that DNA model scores known multibaggers ≥0.60.
    If any confirmed pick scores < 0.50, the DNA model has a problem.
    """
    log.info("═"*60)
    log.info("BACKTEST C — DNA SCORER VALIDATION ON CONFIRMED PICKS")
    log.info("Validates multibagger_dna.py against 6 known ground truth picks")
    log.info("═"*60)

    try:
        from multibagger_dna import compute_dna_score, score_revenue_acceleration
        from multibagger_dna import score_promoter_conviction, score_catalyst_quality
        from multibagger_dna import score_mcap_headroom, score_price_stage_for_multibagger
    except ImportError as e:
        log.error(f"Cannot import multibagger_dna: {e}")
        log.error("Make sure multibagger_dna.py is in the repo root.")
        return []

    results = []
    for pick in VALUEPICK_PICKS:
        sym = pick["symbol"]
        log.info(f"\n{sym} @ ₹{pick['entry_price']} — {pick['entry_date']}")

        # Fetch historical data up to entry date
        entry_dt = datetime.strptime(pick["entry_date"], "%Y-%m-%d")
        from_dt  = entry_dt - timedelta(days=730)
        to_dt    = entry_dt + timedelta(days=5)

        hist = None
        for s in [sym, pick.get("symbol_alt","")]:
            if not s: continue
            try:
                df = yf.Ticker(s).history(
                    start=from_dt.strftime("%Y-%m-%d"),
                    end=to_dt.strftime("%Y-%m-%d"),
                    interval="1d", auto_adjust=True,
                )
                if df is not None and not df.empty and len(df) >= 30:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    entry_idx = min(df.index.searchsorted(pd.Timestamp(entry_dt)), len(df)-1)
                    hist = df.iloc[:entry_idx+1]
                    break
            except Exception:
                pass

        # Minimal info dict from what we know
        info = {
            "currentPrice": pick["entry_price"],
            "marketCap":    pick["entry_price"] * 1e5 * 1e7,  # rough estimate
        }
        mcap_cr = (info["marketCap"]) / 1e7

        # Run DNA scorer with no Screener data (as it would be at runtime)
        try:
            dna = compute_dna_score(
                symbol        = sym,
                market_cap_cr = mcap_cr,
                info          = info,
                hist          = hist if hist is not None else pd.DataFrame(),
                screener_data = None,
                public_data   = None,
                announcements = [],
                sector        = "Other",
            )
            result = {
                "symbol":       sym,
                "entry_date":   pick["entry_date"],
                "entry_price":  pick["entry_price"],
                "known_return": pick.get("return_x", "?"),
                "thesis":       pick["thesis"],
                "dna_score":    dna["dna_score"],
                "dna_grade":    dna["dna_grade"],
                "rev_accel":    dna["rev_accel_score"],
                "promoter":     dna["promoter_score"],
                "catalyst":     dna["catalyst_score"],
                "mcap":         dna["mcap_score"],
                "price_stage":  dna["price_stage_score"],
                "pass":         dna["dna_score"] >= 0.50,
                "note":         pick.get("data_note",""),
            }
            results.append(result)

            status = "✅ PASS" if result["pass"] else "❌ FAIL"
            log.info(f"  DNA: {dna['dna_score']:.3f} [{dna['dna_grade'][:1]}] {status}")
            log.info(f"  Components: rev={dna['rev_accel_score']:.2f} promo={dna['promoter_score']:.2f} "
                     f"cat={dna['catalyst_score']:.2f} mcap={dna['mcap_score']:.2f} "
                     f"stage={dna['price_stage_score']:.2f}")
            if dna["dna_flags"]:
                for f in dna["dna_flags"]:
                    log.info(f"    ▶ {f}")
        except Exception as e:
            log.warning(f"  DNA scoring failed: {e}")
            results.append({"symbol": sym, "error": str(e)})

        time.sleep(1.0)

    # Summary
    log.info("\n" + "═"*60)
    log.info("BACKTEST C SUMMARY — DNA Scorer Validation")
    passed = sum(1 for r in results if r.get("pass"))
    log.info(f"  Passed (DNA ≥0.50): {passed}/{len(results)}")
    log.info("  Per-pick scores:")
    for r in results:
        if "error" in r:
            log.info(f"  {r['symbol']:<20} ERROR: {r['error']}")
        else:
            status = "✅" if r["pass"] else "❌"
            log.info(f"  {r['symbol']:<20} {status} DNA:{r['dna_score']:.3f} "
                     f"({r['known_return']}x confirmed)")

    if passed < len(results):
        log.warning("\n  ⚠ Some confirmed picks scored below 0.50.")
        log.warning("  This means the DNA model would have missed them.")
        log.warning("  Check data availability — yfinance historical data for")
        log.warning("  pre-2015 Indian small caps is often sparse.")

    return results

# ─────────────────────────────────────────────────────────────
# BACKTEST B — LAYER 1 HISTORICAL PERFORMANCE
# ─────────────────────────────────────────────────────────────

@dataclass
class BacktestSignal:
    symbol: str
    signal_date: str
    signal_price: float
    rsi: float
    adx: float
    vol_surge: bool
    obv_bullish: bool
    has_tight_base: bool
    breakout_score: float
    macd_expanding: bool = False
    has_hh_hl: bool = False
    bb_squeeze_exp: bool = False
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_6m: Optional[float] = None
    nifty_1m:  Optional[float] = None
    nifty_3m:  Optional[float] = None
    alpha_3m:  Optional[float] = None


def simulate_v3(df, symbol, idx, lookback=252):
    """v3: 52W breakout scanner. Kept as baseline — do NOT remove."""
    if idx < min(lookback+60, 120): return None
    w = df.iloc[:idx+1]
    c = w["Close"]; v = w["Volume"]
    last_close = float(c.iloc[-1])
    avg_vol = float(v.rolling(20).mean().iloc[-1])
    if avg_vol < 50_000 or last_close < 10: return None
    e21 = ema(c,21); e55 = ema(c,55)
    rsi_s = rsi(c,14); atr_s = atr(w,14)
    atr_now = float(atr_s.iloc[-1])
    high_252 = float(c.rolling(lookback).max().shift(1).iloc[-1])
    if not (last_close >= high_252+atr_now*_BACKTEST_PARAMS["atr_mult"]): return None
    if not float(e21.iloc[-1])>float(e55.iloc[-1]): return None
    last_rsi = float(rsi_s.iloc[-1])
    if not (_BACKTEST_PARAMS["rsi_lo"]<=last_rsi<=_BACKTEST_PARAMS["rsi_hi"]): return None
    last_vol = float(v.iloc[-1])
    obv_s = obv_series(w); obv_w = obv_s.iloc[-10:]
    obv_bull = float(obv_w.iloc[-1]-obv_w.iloc[0]) > float(obv_w.std() if obv_w.std()>0 else 1)*0.1
    recent_atr = float(atr_s.iloc[-20:].mean())
    prior_atr  = float(atr_s.iloc[-80:-20].mean()) if len(atr_s)>=80 else recent_atr
    has_base   = (recent_atr/prior_atr)<0.7 if prior_atr>0 else False
    b_score    = min((last_close-high_252)/high_252*20,1.0) if high_252>0 else 0
    return BacktestSignal(
        symbol=symbol, signal_date=str(w.index[-1].date()),
        signal_price=round(last_close,2), rsi=round(last_rsi,1),
        adx=0.0, vol_surge=last_vol>=avg_vol*_BACKTEST_PARAMS["vol_mult"],
        obv_bullish=obv_bull, has_tight_base=has_base, breakout_score=round(b_score,3),
    )


def simulate_v4(df, symbol, idx):
    """v4: volatility-adjusted momentum. Keep alongside v3 for comparison."""
    if idx < 312: return None
    w = df.iloc[:idx+1]
    c = w["Close"]; v = w["Volume"]
    last_close = float(c.iloc[-1])
    avg_vol = float(v.rolling(20).mean().iloc[-1])
    if avg_vol < 150_000 or last_close < 10: return None
    if int((v.iloc[-63:]>0).sum()) < 50: return None
    e21 = ema(c,21); e55 = ema(c,55); rsi_s = rsi(c,14)
    if not float(e21.iloc[-1])>float(e55.iloc[-1]): return None
    last_rsi = float(rsi_s.iloc[-1])
    if not (48.0<=last_rsi<=80.0): return None
    if len(c)<262: return None
    ret_6m = (float(c.iloc[-1])-float(c.iloc[-126]))/float(c.iloc[-126])
    ret_12m= (float(c.iloc[-1])-float(c.iloc[-252]))/float(c.iloc[-252])
    ann_vol= float(c.pct_change().dropna().iloc[-252:].std())*(252**0.5)
    if ann_vol<=0.01: return None
    mom    = ((ret_6m/ann_vol)+(ret_12m/ann_vol))/2
    if mom<=0: return None
    obv_s  = obv_series(w); obv_w = obv_s.iloc[-10:]
    obv_slope_z = (float(obv_w.iloc[-1]-obv_w.iloc[0]))/(float(obv_w.std()) if obv_w.std()>0 else 1)
    if obv_slope_z < -1.5: return None
    last_vol   = float(v.iloc[-1])
    recent_atr = float(atr(w,14).iloc[-20:].mean())
    prior_atr  = float(atr(w,14).iloc[-80:-20].mean()) if len(w)>=80 else recent_atr
    # MACD
    try:
        macd_h = (c.ewm(12,adjust=False).mean()-c.ewm(26,adjust=False).mean())
        sig_l  = macd_h.ewm(9,adjust=False).mean()
        hist   = macd_h-sig_l
        mac_exp= abs(float(hist.iloc[-1]))>abs(float(hist.iloc[-2])) and float(hist.iloc[-1])>0
    except: mac_exp = False
    return BacktestSignal(
        symbol=symbol, signal_date=str(w.index[-1].date()),
        signal_price=round(last_close,2), rsi=round(last_rsi,1), adx=0.0,
        vol_surge=last_vol>=avg_vol*1.5, obv_bullish=obv_slope_z>0.1,
        has_tight_base=(recent_atr/prior_atr)<0.7 if prior_atr>0 else False,
        breakout_score=round(min(max(mom/2.0,0),1.0),3),
        macd_expanding=mac_exp,
    )


def measure_returns(sig, df, nifty_df, idx):
    future  = df.iloc[idx:]
    if future.empty: return sig
    entry_p = float(future["Close"].iloc[0])
    def fwd(days):
        if idx+days<len(df):
            return round((float(df["Close"].iloc[idx+days])-entry_p)/entry_p*100,1)
        return None
    sig.return_1m = fwd(21); sig.return_3m = fwd(63); sig.return_6m = fwd(126)
    if not nifty_df.empty:
        nc = nifty_df["Close"].squeeze()
        ni = nifty_df.index.searchsorted(pd.Timestamp(sig.signal_date))
        if ni<len(nc):
            np0 = float(nc.iloc[ni])
            def nf(days):
                if ni+days<len(nc): return round((float(nc.iloc[ni+days])-np0)/np0*100,1)
                return None
            sig.nifty_1m = nf(21); sig.nifty_3m = nf(63)
            if sig.return_3m is not None and sig.nifty_3m is not None:
                sig.alpha_3m = round(sig.return_3m-sig.nifty_3m,1)
    return sig


def summarise(signals: list[BacktestSignal], label: str) -> dict:
    if not signals:
        return {"label": label, "count": 0, "hit_rate_3m": "N/A",
                "avg_return_3m": "N/A", "avg_alpha_3m": "N/A",
                "median_3m": "N/A", "best_3m": "N/A"}
    r3 = [s.return_3m for s in signals if s.return_3m is not None]
    a3 = [s.alpha_3m  for s in signals if s.alpha_3m  is not None]
    hits = [x for x in r3 if x>0]
    result = {
        "label":         label,
        "count":         len(signals),
        "hit_rate_3m":   f"{len(hits)/len(r3)*100:.1f}%" if r3 else "N/A",
        "avg_return_3m": f"{np.mean(r3):+.1f}%"          if r3 else "N/A",
        "avg_alpha_3m":  f"{np.mean(a3):+.1f}%"          if a3 else "N/A",
        "median_3m":     f"{np.median(r3):+.1f}%"        if r3 else "N/A",
        "best_3m":       f"{max(r3):+.1f}%"              if r3 else "N/A",
    }
    log.info(f"\n  [{label}] n={result['count']} "
             f"hit={result['hit_rate_3m']} avg3M={result['avg_return_3m']} "
             f"alpha={result['avg_alpha_3m']} median={result['median_3m']}")
    return result


def run_backtest_b(symbols, lookback_days=500, sample_every=5,
                   batch_size=50, scanner="compare") -> dict:
    log.info("═"*60)
    log.info(f"BACKTEST B — LAYER 1 HISTORICAL  [scanner={scanner}]")
    log.info(f"  Symbols:{len(symbols)} Lookback:{lookback_days}d Sample:every {sample_every}d")
    log.info("═"*60)

    log.info("Fetching Nifty 50 benchmark...")
    nifty_df = yf.download("^NSEI", period="4y", interval="1d",
                           progress=False, auto_adjust=True)
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)

    v3_all, v4_all = [], []
    v3_obv, v4_obv = [], []
    v4_pending     = []  # (sig, df, idx) for cross-sectional pass
    skipped = 0

    batches = [symbols[i:i+batch_size] for i in range(0,len(symbols),batch_size)]
    for bi, batch in enumerate(batches):
        log.info(f"  Batch {bi+1}/{len(batches)}: {batch[0]}…{batch[-1]}")
        try:
            raw = yf.download(batch, period="4y", interval="1d",
                              group_by="ticker", auto_adjust=True,
                              progress=False, threads=True)
        except Exception as e:
            log.warning(f"  Batch {bi+1} failed: {e}")
            time.sleep(3); continue

        for sym in batch:
            try:
                df = raw[sym].copy() if len(batch)>1 and isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(subset=["Close","Volume"], inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if len(df) < 200: skipped+=1; continue

                warmup = 312 if scanner in ("v4","compare") else 260
                start  = max(len(df)-lookback_days, warmup)
                end    = len(df)-21
                if start>=end: skipped+=1; continue

                for idx in range(start, end, sample_every):
                    if scanner in ("v3","compare"):
                        s = simulate_v3(df, sym, idx)
                        if s:
                            s = measure_returns(s, df, nifty_df, idx)
                            v3_all.append(s)
                            if s.obv_bullish: v3_obv.append(s)

                    if scanner in ("v4","compare"):
                        s = simulate_v4(df, sym, idx)
                        if s: v4_pending.append((s, df.copy(), idx))

            except Exception as e:
                log.debug(f"  {sym}: {e}"); skipped+=1

        time.sleep(1.5)

    # v4 cross-sectional percentile gate
    if v4_pending:
        from collections import defaultdict
        by_date = defaultdict(list)
        for item in v4_pending:
            by_date[item[0].signal_date].append(item)
        kept = 0
        for date_str, items in sorted(by_date.items()):
            scores = [x[0].breakout_score for x in items]
            if len(scores)<5: continue
            thresh = np.percentile(scores, 80)
            for sig, df, idx in items:
                if sig.breakout_score >= thresh:
                    sig = measure_returns(sig, df, nifty_df, idx)
                    v4_all.append(sig)
                    if sig.obv_bullish: v4_obv.append(sig)
                    kept += 1
        log.info(f"  v4 cross-sectional: {len(v4_pending)} raw → {kept} kept (top 20%/date)")

    log.info(f"\n  Skipped: {skipped}  v3 signals: {len(v3_all)}  v4 signals: {len(v4_all)}")

    out = {
        "v3_all":   summarise(v3_all,  "v3 all"),
        "v3_obv":   summarise(v3_obv,  "v3 OBV confirmed"),
        "v4_all":   summarise(v4_all,  "v4 all (top-20% momentum)"),
        "v4_obv":   summarise(v4_obv,  "v4 OBV confirmed"),
    }
    if scanner=="compare":
        log.info("\n  ══ v3 vs v4 COMPARISON ══")
        for metric in ["count","hit_rate_3m","avg_return_3m","avg_alpha_3m","median_3m"]:
            log.info(f"  {metric:<20} v3:{out['v3_all'].get(metric,'?'):<12} v4:{out['v4_all'].get(metric,'?')}")
    return out

# ─────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────

def write_report(date_str, mode, a_results, b_results, c_results, params):
    sep = "═"*68
    lines = [
        sep,
        f"  BACKTEST REPORT — {date_str}",
        f"  Mode: {mode} | Scanner: {params.get('scanner','?')} | "
        f"Universe: {'Full ~1800' if params.get('full') else '48 curated'}",
        sep, "",
    ]

    if a_results:
        lines += ["  BACKTEST A — VALUEPICK GROUND TRUTH", ""]
        for s in a_results:
            lines += [
                f"  {s.symbol}  Entry:₹{s.entry_price} on {s.entry_date}",
                f"  Thesis: {s.thesis}",
                f"  Signals: {s.signals_present}  L1score:{s.estimated_l1_score}",
                f"  Returns: 1M:{s.return_1m}% 3M:{s.return_3m}% 6M:{s.return_6m}% 1Y:{s.return_1y}%",
                f"  Nifty:   1M:{s.nifty_1m}% 3M:{s.nifty_3m}%",
                f"  {s.notes}" if s.notes else "",
                "",
            ]

    if c_results:
        lines += ["", "  BACKTEST C — DNA SCORER VALIDATION", ""]
        passed = sum(1 for r in c_results if r.get("pass"))
        lines.append(f"  Passed (DNA ≥0.50): {passed}/{len(c_results)}")
        lines.append("")
        for r in c_results:
            if "error" in r:
                lines.append(f"  {r['symbol']:<22} ERROR: {r['error']}")
            else:
                s = "✅" if r["pass"] else "❌"
                lines.append(
                    f"  {s} {r['symbol']:<20} DNA:{r['dna_score']:.3f}[{r['dna_grade'][:1]}] "
                    f"rev:{r['rev_accel']:.2f} promo:{r['promoter']:.2f} "
                    f"cat:{r['catalyst']:.2f} ({r['known_return']}x)"
                )
        lines.append("")

    if b_results:
        lines += ["", "  BACKTEST B — LAYER 1 HISTORICAL PERFORMANCE", ""]
        for label, m in b_results.items():
            lines.append(
                f"  [{m['label']}] n={m['count']} hit={m['hit_rate_3m']} "
                f"avg3M={m['avg_return_3m']} alpha={m['avg_alpha_3m']} "
                f"median={m['median_3m']}"
            )
        if "v3_all" in b_results and "v4_all" in b_results:
            lines += [
                "", "  v3 vs v4 VERDICT:",
                "  v3 = 52W breakout (legacy baseline — keep for comparison)",
                "  v4 = volatility-adjusted momentum (NSE formula)",
                "  If v4 alpha > v3 alpha: momentum filter is adding value.",
                "  If spread is small: regime or universe choice matters more.",
            ]

    lines += [
        "", sep,
        "  SURVIVORSHIP BIAS NOTE:",
        "  This backtest uses current constituents — delisted stocks are missing.",
        "  Hit rates are overstated. Use directionally, not as proof of edge.",
        "  DNA Backtest C uses only 6 ground truth picks — too few for statistics.",
        "  Its value is sanity-checking the model, not proving it works.",
        sep,
    ]
    return "\n".join(l for l in lines)


# ─────────────────────────────────────────────────────────────
# CURATED 48-STOCK UNIVERSE (unchanged — used for weekly B run)
# ─────────────────────────────────────────────────────────────

CURATED_48 = [
    # Large liquid NSE stocks spanning all major sectors
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","HINDUNILVR.NS","ITC.NS","AXISBANK.NS","BAJFINANCE.NS",
    "WIPRO.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS",
    # Mid caps
    "PIDILITIND.NS","BERGEPAINT.NS","TORNTPHARM.NS","MPHASIS.NS","COFORGE.NS",
    "LTTS.NS","POLYCAB.NS","ASTRAL.NS","PIIND.NS","AAPL.NS",
    "DEEPAKNTR.NS","NAVINFLUOR.NS","ALKYLAMINE.NS","FINECHEM.NS","GALAXYSURF.NS",
    # Small caps (confirmed multibaggers or near-misses)
    "SHANTIGEAR.NS","COSMOFERR.NS","EKINDIA.NS","PAUSHAKLTD.NS",
    # Sector representatives
    "KPIT.NS","TATAELXSI.NS","PERSISTENT.NS","CYIENT.NS",
    "AETHER.NS","APOLLOPIPE.NS","GPIL.NS","KSB.NS","SPLPETRO.NS",
    # Indices as sanity check
]
# Filter to valid NSE symbols only
CURATED_48 = [s for s in CURATED_48 if s.endswith(".NS")]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multibagger Backtest v5")
    parser.add_argument("--mode", default="both",
                        choices=["a","b","c","both","all"],
                        help="a=ground truth, b=L1 historical, c=DNA validation, "
                             "both=a+b, all=a+b+c")
    parser.add_argument("--full",    action="store_true",
                        help="Full NSE universe (~1,800 symbols) for Backtest B")
    parser.add_argument("--v4",      action="store_true",
                        help="Backtest B: v4 momentum scanner only")
    parser.add_argument("--compare", action="store_true",
                        help="Backtest B: v3 vs v4 side-by-side (default)")
    parser.add_argument("--tune",    action="store_true",
                        help="(Deprecated — no-op. DNA model tuning is done via C.)")
    args = parser.parse_args()

    if args.tune:
        log.warning("--tune is deprecated. Use --mode c to validate DNA scorer instead.")

    os.makedirs("results", exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")

    scanner = "v4" if args.v4 else "compare"  # default compare
    params  = {"full": args.full, "v4": args.v4, "compare": not args.v4, "scanner": scanner}

    a_results, b_results, c_results = [], {}, []

    if args.mode in ("a","both","all"):
        a_results = run_backtest_a()

    if args.mode in ("b","both","all"):
        if args.full:
            try:
                from universe import fetch_nse_symbols
                symbols = fetch_nse_symbols()
                log.info(f"Full universe: {len(symbols)} symbols")
            except Exception as e:
                log.warning(f"Universe fetch failed: {e}. Using curated 48.")
                symbols = CURATED_48
        else:
            symbols = CURATED_48
            log.info(f"Curated 48-stock universe")

        b_results = run_backtest_b(
            symbols,
            lookback_days=500 if not args.full else 365,
            sample_every=5   if not args.full else 10,
            scanner=scanner,
        )

    if args.mode in ("c","all"):
        c_results = run_backtest_c()

    # Write report
    report = write_report(date_str, args.mode, a_results, b_results, c_results, params)
    print("\n" + report)

    txt_path  = f"results/backtest_{date_str}.txt"
    json_path = f"results/backtest_{date_str}.json"
    with open(txt_path, "w") as f: f.write(report)

    json_data = {
        "date": date_str, "params": params,
        "backtest_a": [vars(s) for s in a_results],
        "backtest_b": b_results,
        "backtest_c": c_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    log.info(f"\n✅ Report: {txt_path}")
    log.info(f"   JSON:   {json_path}")


if __name__ == "__main__":
    main()
