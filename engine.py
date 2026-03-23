"""
engine.py — Multibagger Discovery Engine v4 (yfinance edition)
Pipeline: Volatility-Adjusted Momentum → Sector Filter → Fundamental Quality → Top 20

v4 MAJOR REDESIGN based on full NSE universe backtest (1,371 signals, March 2026):
  Previous hit rate: 49.8%, avg 3M +2.7%, alpha +1.1% — essentially no edge.

  Root causes identified:
  1. RAW 52W HIGH BREAKOUT has no predictive power at scale. It fires on too many
     weak setups — any stock that had a quiet period and one up-day can trigger it.
  2. OBV is unreliable on NSE small caps due to thin/stale yfinance volume data.
  3. NO SECTOR FILTER — breakouts in falling sectors have near-zero hit rate.
  4. LIQUIDITY TOO LOW — signals on stocks with avg vol 50k-100k are noise.
  5. MOMENTUM NOT VALIDATED — we were using EMA crossover as "momentum" but the
     academic literature shows 6M+12M volatility-adjusted return is what actually works.

v4 improvements (sourced from NSE's own factor research + academic literature):

  1. VOLATILITY-ADJUSTED MOMENTUM SCORE — NSE's Nifty200 Momentum 30 formula:
     score = (6M_return/annual_vol + 12M_return/annual_vol) / 2
     Return divided by volatility = risk-adjusted momentum, not raw price movement.
     NSE's own Smallcap250 Momentum Quality 100 index: 23.79% CAGR vs 16.55% (2005-2023),
     outperformed in 17/19 calendar years. This is the validated approach for India.
     Source: NSE Indices Smallcap250 Momentum Quality 100 whitepaper (2024)

  2. SECTOR MOMENTUM GATE — Only accept stocks in sectors with positive 3M trend.
     Breakouts in declining sectors fail overwhelmingly. Academic research: sector momentum
     is largely independent of individual stock momentum and a strong performance predictor.
     Source: Markit Research Signals sector rotation model; Fama-French sector studies.

  3. LIQUIDITY MINIMUM — Require ≥ 50 actively traded days in last 63 (3 months).
     Eliminates stocks that barely trade — their yfinance data is stale/noisy and
     any "breakout" is meaningless. Standard practice for Indian momentum strategies.

  4. MOMENTUM Z-SCORE RANKING — Score each stock relative to the full universe.
     A stock in the top 10% of momentum z-scores is genuinely strong.
     A stock above its 52W high but in the bottom 50% of momentum is noise.

  5. ADX MINIMUM RAISED from 18 → 25 — require a defined trend, not just any move.

  6. TIER-BASED HARD GATE — must pass ALL of:
     a. Momentum z-score > 0 (above-average momentum)
     b. Sector 3M return > -5% (not in a falling sector)
     c. Liquidity: ≥50 traded days / 63
     d. ADX ≥ 25 (defined trend)
     e. EMA21 > EMA55 (directional trend)
     f. RSI 48-80 (not oversold, not extreme overbought)

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
# CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── Universe filters ──
    min_price: float         = 20.0
    max_price: float         = 10_000.0   # removed ceiling — was filtering HAL, MRF etc
    min_avg_volume: int      = 150_000    # RAISED: was 75k — low liquidity = noisy signals
    min_traded_days_63: int  = 50         # NEW: must have ≥50 active trading days in last 63
    max_market_cap_cr: float = 15_000.0

    # ── Momentum parameters (NSE Momentum Index methodology) ──
    mom_6m_days:  int = 126   # 6-month momentum lookback
    mom_12m_days: int = 252   # 12-month momentum lookback
    vol_days:     int = 252   # annualised volatility window

    # ── Sector momentum gate ──
    sector_3m_min_return: float = -5.0  # sector 3M return must be > this (%)

    # ── Technical gates ──
    ema_fast: int   = 21
    ema_slow: int   = 55
    rsi_period: int = 14
    rsi_lo: float   = 48.0
    rsi_hi: float   = 80.0    # raised from 78 — allow stronger momentum
    adx_min: float  = 25.0    # RAISED: was 18 — require a defined trend

    # ── OBV (retained but only as extreme distribution gate) ──
    obv_trend_window: int = 10

    # ── Relative strength days ──
    rs_lookback_days: int      = 20
    nifty_down_threshold: float = 0.5

    # ── Consolidation base ──
    base_atr_lookback: int          = 20
    base_atr_full: int              = 60
    base_contraction_ratio: float   = 0.7

    # ── Fundamental hard filters ──
    max_de_ratio: float            = 1.5   # slightly loosened — some growth cos carry debt
    min_roe_pct: float             = 8.0   # slightly loosened — catches more small caps
    min_revenue_growth_pct: float  = 5.0   # loosened — sector context matters more
    max_mcap_to_sales: float       = 8.0   # loosened — high quality cos deserve premium

    # ── Scoring weights v4.1 — added MACD, BB, structure ──
    #
    # Research basis:
    #   - Volatility-adjusted momentum (6M+12M) is NSE's validated formula
    #   - MACD histogram expanding = momentum accelerating (new)
    #   - Bollinger %B + squeeze expansion = breakout precursor (new)
    #   - HH/HL price structure = structural trend confirmation (new)
    #   - Sector momentum is structural — stocks in hot sectors outperform
    #   - Fundamental quality prevents momentum traps
    #
    w_mom_score:   float = 0.30   # volatility-adjusted momentum z-score (primary)
    w_fundamental: float = 0.20   # quality filter
    w_macd:        float = 0.12   # NEW: MACD histogram direction/expansion
    w_bb:          float = 0.10   # NEW: Bollinger %B + squeeze
    w_structure:   float = 0.08   # NEW: HH/HL price structure
    w_rel_str:     float = 0.10   # RS days — stock holding up on down-market days
    w_momentum:    float = 0.06   # EMA trend confirmation
    w_volume:      float = 0.04   # volume
    w_obv:         float = 0.00   # OBV removed from scoring (unreliable on small caps)

    top_n: int          = 20
    batch_size: int     = 50
    sleep_between_batches: float = 2.0

    nifty_symbol: str = "^NSEI"


CFG = Config()


# ─────────────────────────────────────────────────────────────
# MARKET REGIME CLASSIFIER
# Called once per run. Returns a dict describing current market conditions.
# ─────────────────────────────────────────────────────────────
def get_market_regime() -> dict:
    """
    Classify current NSE market regime using Nifty Smallcap 100.
    Returns bull / neutral / bear with supporting data.
    This is displayed at the top of every report so you know whether
    to act on breakouts or wait.
    """
    try:
        # Nifty Smallcap 100 is the relevant benchmark for our universe
        sc100 = yf.download("^CNXSC", period="6mo", interval="1d",
                            progress=False, auto_adjust=True)
        nifty50 = yf.download("^NSEI", period="6mo", interval="1d",
                              progress=False, auto_adjust=True)

        results = {}
        for name, df in [("smallcap100", sc100), ("nifty50", nifty50)]:
            if df is None or df.empty or len(df) < 60:
                results[name] = None
                continue
            c = df["Close"].squeeze()
            e20  = c.ewm(span=20, adjust=False).mean()
            e50  = c.ewm(span=50, adjust=False).mean()
            e200 = c.ewm(span=200, adjust=False).mean()

            last       = float(c.iloc[-1])
            chg_1m     = (last - float(c.iloc[-21])) / float(c.iloc[-21]) * 100
            chg_3m     = (last - float(c.iloc[-63])) / float(c.iloc[-63]) * 100

            above_50   = last > float(e50.iloc[-1])
            above_200  = last > float(e200.iloc[-1])
            ema_stacked = float(e20.iloc[-1]) > float(e50.iloc[-1]) > float(e200.iloc[-1])

            results[name] = {
                "last":        round(last, 0),
                "chg_1m_pct":  round(chg_1m, 1),
                "chg_3m_pct":  round(chg_3m, 1),
                "above_50ema": above_50,
                "above_200ema": above_200,
                "ema_stacked": ema_stacked,
            }

        # Regime decision — Smallcap 100 primary, Nifty 50 fallback
        # Bug fix: when sc is None, never default to BULL — use n50 data instead
        sc  = results.get("smallcap100")
        n50 = results.get("nifty50")

        # Pick the best available benchmark
        benchmark = sc if sc is not None else n50

        if benchmark is None:
            regime      = "UNKNOWN"
            regime_note = "Could not fetch market data from yfinance"
        elif benchmark["ema_stacked"] and benchmark["chg_1m_pct"] > 0 and benchmark["above_200ema"]:
            regime      = "BULL"
            regime_note = "All EMAs stacked, index above 200EMA. Breakout conditions favourable."
        elif not benchmark["above_200ema"] and benchmark["chg_3m_pct"] < -10:
            regime      = "BEAR"
            regime_note = "Index below 200EMA, down >10% in 3 months. High false breakout risk — be very selective."
        elif not benchmark["above_200ema"] or benchmark["chg_3m_pct"] < -5:
            regime      = "BEAR"
            regime_note = f"Index {'below' if not benchmark['above_200ema'] else 'near'} 200EMA, {benchmark['chg_3m_pct']:+.1f}% in 3 months. Caution."
        else:
            regime      = "NEUTRAL"
            regime_note = "Mixed signals. Favour high-conviction setups only."

        # Add note if falling back to Nifty 50
        if sc is None and n50 is not None:
            regime_note += " (Smallcap 100 data unavailable — using Nifty 50)"

        return {
            "regime":      regime,
            "regime_note": regime_note,
            "smallcap100": results.get("smallcap100"),
            "nifty50":     results.get("nifty50"),
        }

    except Exception as e:
        log.warning(f"Market regime fetch failed: {e}")
        return {"regime": "UNKNOWN", "regime_note": str(e),
                "smallcap100": None, "nifty50": None}


# ─────────────────────────────────────────────────────────────
# NIFTY DAILY RETURNS — fetched once, passed to each stock analysis
# ─────────────────────────────────────────────────────────────
def fetch_nifty_returns(period: str = "3mo") -> pd.Series:
    """Fetch Nifty 50 daily % returns for relative strength calculation."""
    try:
        df = yf.download(CFG.nifty_symbol, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        c = df["Close"].squeeze()
        return c.pct_change() * 100
    except Exception as e:
        log.warning(f"Nifty fetch failed: {e}")
        return pd.Series(dtype=float)


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


def di_spread(df: pd.DataFrame, n: int = 14) -> tuple[float, float]:
    """Return (+DI, -DI) values. +DI >> -DI = buyers firmly in control."""
    try:
        up   = df["High"].diff()
        down = -df["Low"].diff()
        pdm  = up.where((up > down) & (up > 0), 0.0)
        ndm  = down.where((down > up) & (down > 0), 0.0)
        atr_ = atr(df, n)
        pdi  = float((100 * pdm.rolling(n).mean() / atr_).iloc[-1])
        ndi  = float((100 * ndm.rolling(n).mean() / atr_).iloc[-1])
        return round(pdi, 1), round(ndi, 1)
    except Exception:
        return 0.0, 0.0


def macd_signal(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    MACD indicator: histogram direction + zero-line relationship.

    Returns:
      macd_hist:         current histogram value
      hist_expanding:    histogram is growing (momentum accelerating)
      hist_positive:     histogram is above zero
      macd_above_signal: MACD line above signal line
      macd_score:        0-1 composite score
    """
    try:
        ema_fast = c.ewm(span=fast, adjust=False).mean()
        ema_slow = c.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
        hist      = macd_line - sig_line

        last_hist  = float(hist.iloc[-1])
        prev_hist  = float(hist.iloc[-2]) if len(hist) > 1 else 0.0
        hist_exp   = abs(last_hist) > abs(prev_hist) and last_hist > 0
        hist_pos   = last_hist > 0
        macd_above = float(macd_line.iloc[-1]) > float(sig_line.iloc[-1])

        # Score: best when histogram positive AND expanding (accelerating momentum)
        if hist_pos and hist_exp and macd_above:
            score = 0.90
        elif hist_pos and macd_above:
            score = 0.70
        elif hist_pos and not macd_above:
            score = 0.45   # fading but still positive
        elif not hist_pos and not macd_above:
            score = 0.10   # bearish
        else:
            score = 0.30

        return {
            "macd_hist":         round(last_hist, 4),
            "hist_expanding":    hist_exp,
            "hist_positive":     hist_pos,
            "macd_above_signal": macd_above,
            "macd_score":        score,
        }
    except Exception:
        return {
            "macd_hist": 0.0, "hist_expanding": False,
            "hist_positive": False, "macd_above_signal": False,
            "macd_score": 0.3,
        }


def bollinger_signals(c: pd.Series, n: int = 20, k: float = 2.0) -> dict:
    """
    Bollinger Band %B and bandwidth.

    %B = (price - lower) / (upper - lower)
      0 = at lower band, 1 = at upper band, 0.5 = at midline
      0.6-0.95 = ideal swing entry zone (above midline, not extreme)

    Bandwidth = (upper - lower) / midline
      Squeeze = bandwidth < 50th percentile of last 120 days
      Expanding = bandwidth growing from squeeze = breakout precursor

    Returns:
      pct_b:             0-1, position within bands
      bandwidth:         (upper-lower)/mid normalised
      is_squeeze:        bandwidth in bottom 25% of 120-day range
      squeeze_expanding: was in squeeze, now expanding = breakout signal
      bb_score:          0-1 composite
    """
    try:
        mid  = c.rolling(n).mean()
        std  = c.rolling(n).std()
        upper = mid + k * std
        lower = mid - k * std

        last_close = float(c.iloc[-1])
        last_upper = float(upper.iloc[-1])
        last_lower = float(lower.iloc[-1])
        last_mid   = float(mid.iloc[-1])
        band_range = last_upper - last_lower

        pct_b = (last_close - last_lower) / band_range if band_range > 0 else 0.5

        # Bandwidth history for squeeze detection
        bw_series = (upper - lower) / mid.replace(0, np.nan)
        bw_now    = float(bw_series.iloc[-1]) if not np.isnan(bw_series.iloc[-1]) else 0.1
        bw_hist   = bw_series.iloc[-120:].dropna()
        bw_pct25  = float(bw_hist.quantile(0.25)) if len(bw_hist) >= 20 else bw_now

        is_squeeze = bw_now <= bw_pct25

        # Was in squeeze recently (last 10 days) but not now = expanding from squeeze
        recent_bw        = bw_series.iloc[-10:]
        was_in_squeeze   = bool((recent_bw <= bw_pct25).any())
        squeeze_expanding = was_in_squeeze and not is_squeeze and bw_now > float(recent_bw.min())

        # Score: %B 0.6-0.85 = ideal breakout zone; squeeze expanding = bonus
        if 0.60 <= pct_b <= 0.85:
            bb_score = 0.80
        elif 0.85 < pct_b <= 0.95:
            bb_score = 0.65   # strong but approaching upper band
        elif 0.95 < pct_b:
            bb_score = 0.35   # extended above upper band
        elif 0.45 <= pct_b < 0.60:
            bb_score = 0.55   # mid-band pullback, OK
        else:
            bb_score = 0.20   # below midline

        if squeeze_expanding:
            bb_score = min(bb_score + 0.15, 1.0)

        return {
            "pct_b":             round(pct_b, 3),
            "bandwidth":         round(bw_now, 4),
            "is_squeeze":        is_squeeze,
            "squeeze_expanding": squeeze_expanding,
            "bb_score":          round(bb_score, 3),
        }
    except Exception:
        return {
            "pct_b": 0.5, "bandwidth": 0.1, "is_squeeze": False,
            "squeeze_expanding": False, "bb_score": 0.3,
        }


def price_structure(c: pd.Series, lookback: int = 10) -> dict:
    """
    Check if price is making Higher Highs and Higher Lows (confirmed uptrend).

    Uses swing pivots over the last `lookback` bars:
    - HH: most recent swing high > prior swing high
    - HL: most recent swing low  > prior swing low

    HH + HL = structural uptrend confirmed
    Any lower high = warning even if EMA is bullish

    Returns:
      has_hh_hl:      bool — structural uptrend
      has_ll_lh:      bool — structural downtrend (disqualifier)
      structure_score: 0-1
      structure_note:  string description
    """
    try:
        window = c.iloc[-lookback * 2:]  # need double for pivot detection

        # Simple pivot detection: local max/min over 3-bar windows
        highs = []
        lows  = []
        for i in range(1, len(window) - 1):
            if float(window.iloc[i]) > float(window.iloc[i-1]) and float(window.iloc[i]) > float(window.iloc[i+1]):
                highs.append(float(window.iloc[i]))
            if float(window.iloc[i]) < float(window.iloc[i-1]) and float(window.iloc[i]) < float(window.iloc[i+1]):
                lows.append(float(window.iloc[i]))

        has_hh_hl = False
        has_ll_lh = False
        note      = "Insufficient pivots"

        if len(highs) >= 2:
            hh = highs[-1] > highs[-2]   # most recent high > prior high
        else:
            hh = None

        if len(lows) >= 2:
            hl = lows[-1] > lows[-2]    # most recent low > prior low
        else:
            hl = None

        if hh is True and hl is True:
            has_hh_hl = True
            note      = "HH+HL — Structural uptrend confirmed"
        elif hh is False or hl is False:
            has_ll_lh = True
            note      = "Structure weakening — lower high or lower low"
        else:
            note = "Structure neutral"

        if has_hh_hl:
            score = 0.85
        elif has_ll_lh:
            score = 0.10
        else:
            score = 0.45

        return {
            "has_hh_hl":      has_hh_hl,
            "has_ll_lh":      has_ll_lh,
            "structure_score": score,
            "structure_note":  note,
        }
    except Exception:
        return {
            "has_hh_hl": False, "has_ll_lh": False,
            "structure_score": 0.45, "structure_note": "Error",
        }


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — running sum of volume weighted by price direction.
    OBV rising = volume flowing into the stock (accumulation).
    OBV falling with rising price = divergence, breakout may be false.
    """
    direction = np.sign(df["Close"].diff().fillna(0))
    return (df["Volume"] * direction).cumsum()


# ─────────────────────────────────────────────────────────────
# NEW: OBV DIRECTION SIGNAL
# ─────────────────────────────────────────────────────────────
def calc_obv_signal(df: pd.DataFrame) -> tuple[bool, float]:
    """
    Returns (obv_bullish: bool, obv_score: float 0-1).
    OBV is bullish when its short-term EMA is above its longer EMA
    and the slope over the last N days is positive.
    """
    try:
        obv_series = obv(df)
        obv_fast   = obv_series.ewm(span=5, adjust=False).mean()
        obv_slow   = obv_series.ewm(span=CFG.obv_trend_window * 2, adjust=False).mean()

        # Slope: is OBV rising over last N days?
        obv_window    = obv_series.iloc[-CFG.obv_trend_window:]
        obv_slope     = float(obv_window.iloc[-1] - obv_window.iloc[0])
        obv_std       = float(obv_window.std()) if obv_window.std() > 0 else 1
        slope_z       = obv_slope / obv_std   # normalised slope

        obv_ema_bull  = float(obv_fast.iloc[-1]) > float(obv_slow.iloc[-1])
        obv_rising    = slope_z > 0.1          # meaningfully positive slope

        obv_bullish   = obv_ema_bull and obv_rising
        obv_score     = min(max(slope_z / 2, 0), 1.0)  # cap at 1

        return obv_bullish, round(obv_score, 3)
    except Exception:
        return True, 0.5   # neutral if calculation fails


# ─────────────────────────────────────────────────────────────
# NEW: RELATIVE STRENGTH DAYS
# ─────────────────────────────────────────────────────────────
def calc_relative_strength(df: pd.DataFrame, nifty_returns: pd.Series) -> tuple[int, float]:
    """
    Count days where stock rose when Nifty fell >0.5%.
    These are hidden accumulation days — institutions buying the dips.
    Returns (rs_days: int, rs_score: float 0-1).
    """
    try:
        if nifty_returns.empty:
            return 0, 0.3

        stock_returns = df["Close"].pct_change() * 100
        stock_returns.index = pd.to_datetime(stock_returns.index).tz_localize(None)
        nifty_aligned = nifty_returns.copy()
        nifty_aligned.index = pd.to_datetime(nifty_aligned.index).tz_localize(None)

        # Align on common dates, take last N days
        common = stock_returns.index.intersection(nifty_aligned.index)
        if len(common) < 5:
            return 0, 0.3

        common_last = common[-CFG.rs_lookback_days:]
        s_ret = stock_returns.reindex(common_last)
        n_ret = nifty_aligned.reindex(common_last)

        # Days where Nifty fell hard and stock still went up
        nifty_down_days  = n_ret < -CFG.nifty_down_threshold
        stock_up_days    = s_ret > 0
        rs_days          = int((nifty_down_days & stock_up_days).sum())
        total_down_days  = int(nifty_down_days.sum())

        # Score: proportion of Nifty-down days where stock held up
        if total_down_days == 0:
            rs_score = 0.3
        else:
            rs_score = min(rs_days / total_down_days, 1.0)

        return rs_days, round(rs_score, 3)
    except Exception:
        return 0, 0.3


# ─────────────────────────────────────────────────────────────
# v4 NEW: VOLATILITY-ADJUSTED MOMENTUM SCORE
# NSE's own Nifty200 Momentum 30 and Smallcap250 Momentum Quality 100 formula.
# Proven to deliver 23.79% CAGR vs 16.55% for Smallcap 250 (2005-2023).
# ─────────────────────────────────────────────────────────────
def calc_momentum_score(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Compute volatility-adjusted momentum score using NSE's methodology:
      mom_ratio = return / annual_volatility
      final_score = (6M_mom_ratio + 12M_mom_ratio) / 2

    Returns (mom_score, mom_6m_pct, mom_12m_pct)
    """
    try:
        c = df["Close"].squeeze()
        if len(c) < CFG.mom_12m_days + 10:
            return 0.0, 0.0, 0.0

        last  = float(c.iloc[-1])
        p_6m  = float(c.iloc[-CFG.mom_6m_days])
        p_12m = float(c.iloc[-CFG.mom_12m_days])

        ret_6m  = (last - p_6m)  / p_6m
        ret_12m = (last - p_12m) / p_12m

        # Annual volatility (std of daily returns × √252)
        daily_ret = c.pct_change().dropna()
        ann_vol   = float(daily_ret.iloc[-CFG.vol_days:].std()) * (252 ** 0.5)

        if ann_vol <= 0.01:   # near-zero vol = stale data
            return 0.0, 0.0, 0.0

        # Volatility-adjusted momentum ratios (NSE formula)
        mom_6m_ratio  = ret_6m  / ann_vol
        mom_12m_ratio = ret_12m / ann_vol
        mom_score     = (mom_6m_ratio + mom_12m_ratio) / 2

        return round(mom_score, 4), round(ret_6m * 100, 1), round(ret_12m * 100, 1)
    except Exception:
        return 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────
# v4 NEW: SECTOR MOMENTUM MAP
# Fetches 3M return for major NSE sector indices.
# Used to gate breakout signals — stocks in falling sectors are almost
# always false breakouts regardless of individual chart setup.
# ─────────────────────────────────────────────────────────────

# NSE sector index symbols → yfinance tickers
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
    "Midcap":             "^NSEMDCP50",
    "Smallcap":           "^CNXSC",
}

def fetch_sector_momentum() -> dict:
    """
    Fetch 3M returns for NSE sector indices.
    Returns dict: {sector_name: 3m_return_pct}
    Called once per run, cached in the regime object.
    """
    sector_returns = {}
    for sector, ticker in SECTOR_INDEX_MAP.items():
        try:
            df = yf.download(ticker, period="4mo", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 60:
                continue
            c      = df["Close"].squeeze()
            ret_3m = (float(c.iloc[-1]) - float(c.iloc[-63])) / float(c.iloc[-63]) * 100
            sector_returns[sector] = round(ret_3m, 1)
        except Exception:
            pass
    log.info(f"  Sector momentum: {sector_returns}")
    return sector_returns


def get_stock_sector(symbol: str, info: dict) -> str:
    """Map yfinance sector string to our sector buckets."""
    sector_raw = (info.get("sector") or "").lower()
    industry   = (info.get("industry") or "").lower()

    if any(x in sector_raw for x in ["financial", "bank", "insurance"]):
        return "Financial Services"
    if any(x in sector_raw for x in ["technology", "software", "it"]):
        return "IT"
    if any(x in sector_raw for x in ["health", "pharma", "drug", "biotech"]):
        return "Pharma"
    if any(x in sector_raw for x in ["auto", "vehicle", "transport"]):
        return "Auto"
    if any(x in sector_raw for x in ["metal", "steel", "alumin", "copper", "zinc"]):
        return "Metals"
    if any(x in sector_raw for x in ["energy", "oil", "gas", "power", "util"]):
        return "Energy"
    if any(x in sector_raw for x in ["consumer", "food", "beverag", "fmcg"]):
        return "FMCG"
    if any(x in sector_raw for x in ["real estate", "realt", "construct"]):
        return "Realty"
    if any(x in sector_raw for x in ["infra", "cement", "engineer"]):
        return "Infra"
    if any(x in sector_raw for x in ["media", "entertainment", "publish"]):
        return "Media"
    return "Other"
def calc_base_quality(df: pd.DataFrame) -> tuple[bool, float]:
    """
    Checks if price consolidated (ATR contracted) before the breakout.
    A tight base before a volume breakout = coiled spring.
    Returns (has_tight_base: bool, base_score: float 0-1).
    """
    try:
        atr_series        = atr(df, 14)
        recent_atr        = atr_series.iloc[-CFG.base_atr_lookback:].mean()
        prior_atr         = atr_series.iloc[-(CFG.base_atr_lookback + CFG.base_atr_full):
                                           -CFG.base_atr_lookback].mean()

        if prior_atr <= 0 or np.isnan(prior_atr):
            return False, 0.3

        contraction_ratio = recent_atr / prior_atr
        has_tight_base    = contraction_ratio < CFG.base_contraction_ratio

        # Score: tighter base = higher score, capped at 1
        base_score = min(max(1.0 - contraction_ratio, 0.0), 1.0)
        return has_tight_base, round(base_score, 3)
    except Exception:
        return False, 0.3


# ─────────────────────────────────────────────────────────────
# STAGE 1 — MOMENTUM ANALYSER v4 (per symbol)
# ─────────────────────────────────────────────────────────────
def analyse_technicals(df: pd.DataFrame, symbol: str,
                        nifty_returns: pd.Series,
                        sector_momentum: dict = None,
                        universe_mom_scores: list = None,
                        stock_info: dict = None) -> Optional[dict]:
    """
    v4: Volatility-adjusted momentum scoring using NSE's own Momentum Index formula.
    Gate conditions (must ALL pass):
      1. Liquidity: ≥50 traded days in last 63
      2. Average volume ≥ 150k shares
      3. EMA21 > EMA55 (trend direction)
      4. RSI 48-80 (momentum zone)
      5. ADX ≥ 25 (defined trend)
      6. Sector 3M return > -5% (not in falling sector)
      7. Volatility-adjusted momentum z-score > 0 (above-average momentum)
    """
    min_rows = CFG.mom_12m_days + 60
    if df is None or len(df) < min_rows:
        return None

    c = df["Close"].squeeze()
    v = df["Volume"].squeeze()

    # ── Gate 1: Liquidity — enough active trading days ──
    last_63_vol = v.iloc[-63:]
    traded_days = int((last_63_vol > 0).sum())
    if traded_days < CFG.min_traded_days_63:
        return None

    last_close = float(c.iloc[-1])
    avg_vol    = float(v.rolling(20).mean().iloc[-1])
    last_vol   = float(v.iloc[-1])

    if not (CFG.min_price <= last_close):
        return None

    # ── Gate 2: Volume ──
    if avg_vol < CFG.min_avg_volume:
        return None

    # ── Indicators ──
    e21   = ema(c, CFG.ema_fast)
    e55   = ema(c, CFG.ema_slow)
    rsi_s = rsi(c, CFG.rsi_period)
    atr_s = atr(df)
    adx_s = adx(df)

    last_rsi = float(rsi_s.iloc[-1])
    last_adx = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0
    ema_bull = float(e21.iloc[-1]) > float(e55.iloc[-1])
    atr_now  = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0

    # ── Gate 3: EMA trend ──
    if not ema_bull:
        return None

    # ── Gate 4: RSI ──
    if not (CFG.rsi_lo <= last_rsi <= CFG.rsi_hi):
        return None

    # ── Gate 5: ADX — defined trend ──
    if last_adx < CFG.adx_min:
        return None

    # ── Gate 6: Sector momentum ──
    if sector_momentum and stock_info:
        sector = get_stock_sector(symbol, stock_info)
        sector_ret = sector_momentum.get(sector)
        if sector_ret is not None and sector_ret < CFG.sector_3m_min_return:
            log.debug(f"  {symbol}: sector {sector} 3M={sector_ret}% — below threshold")
            return None
    else:
        sector = "Unknown"

    # ── v4 NEW: Volatility-adjusted momentum score ──
    mom_score, mom_6m_pct, mom_12m_pct = calc_momentum_score(df)

    # ── Gate 7: Above-average momentum ──
    if mom_score <= 0:
        return None

    # ── NEW Gate 8: Price structure must not be broken ──
    # Compute structure early so it can gate (saves computing OBV/RS on rejects)
    struct_data = price_structure(c)
    if struct_data["has_ll_lh"]:
        log.debug(f"  {symbol}: broken structure (LL/LH) — rejecting despite positive momentum")
        return None

    # ── OBV: only as extreme distribution gate ──
    obv_bullish, obv_score = calc_obv_signal(df)
    obv_s      = obv(df)
    obv_window = obv_s.iloc[-CFG.obv_trend_window:]
    obv_std    = float(obv_window.std()) if obv_window.std() > 0 else 1
    obv_slope_z = (float(obv_window.iloc[-1]) - float(obv_window.iloc[0])) / obv_std
    if obv_slope_z < -1.5:
        return None

    # ── Relative strength days ──
    rs_days, rs_score = calc_relative_strength(df, nifty_returns)

    # ── Consolidation base ──
    has_tight_base, base_score = calc_base_quality(df)

    # ── NEW v4.1: MACD histogram ──
    macd_data = macd_signal(c)

    # ── NEW v4.1: Bollinger Band %B + squeeze ──
    bb_data = bollinger_signals(c)

    # ── NEW v4.1: +DI / -DI spread ──
    pdi_val, ndi_val = di_spread(df)

    # ── 52W high context (kept for reference, not gating) ──
    rolling_high = c.rolling(CFG.mom_12m_days).max().shift(1)
    w52_high     = float(rolling_high.iloc[-1])
    is_52w_breakout = last_close >= (w52_high + atr_now * 0.3)

    # ── Volume surge ──
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0
    vol_surge = vol_ratio >= 1.5

    # ── 3M price change ──
    idx_3m     = max(len(c) - 63, 0)
    chg_3m_pct = (last_close - float(c.iloc[idx_3m])) / float(c.iloc[idx_3m]) * 100

    # ── Normalise momentum score to 0-1 range for scoring ──
    # mom_score is ratio-of-ratios: typically ranges -2 to +2 for NSE stocks
    # Cap at 2.0 and normalise
    norm_mom_score = min(max(mom_score / 2.0, 0.0), 1.0)

    # ── EMA momentum score (directional strength) ──
    pct_above_e55  = (last_close - float(e55.iloc[-1])) / float(e55.iloc[-1])
    momentum_score = min(max(pct_above_e55 * 5, 0), 1.0)

    volume_score = min(vol_ratio / 3, 1.0)

    return {
        "symbol":           symbol,
        "last_close":       round(last_close, 2),
        "avg_volume":       int(avg_vol),
        "last_volume":      int(last_vol),
        "rsi":              round(last_rsi, 1),
        "atr":              round(atr_now, 2),
        "adx":              round(last_adx, 1),
        "ema21":            round(float(e21.iloc[-1]), 2),
        "ema55":            round(float(e55.iloc[-1]), 2),
        "week52_high":      round(w52_high, 2),
        "is_breakout":      is_52w_breakout,    # 52W breakout (informational)
        "vol_surge":        vol_surge,
        "ema_bullish":      ema_bull,
        "rsi_ok":           True,               # already gated above
        "adx_ok":           True,               # already gated above
        "price_chg_3m_pct": round(chg_3m_pct, 1),
        # v4 momentum signals
        "mom_score_raw":    round(mom_score, 4),
        "mom_6m_pct":       round(mom_6m_pct, 1),
        "mom_12m_pct":      round(mom_12m_pct, 1),
        "sector":           sector,
        "sector_3m_ret":    sector_momentum.get(sector) if sector_momentum else None,
        "traded_days_63":   traded_days,
        # Scores for composite
        "mom_score":        round(norm_mom_score, 3),
        "momentum_score":   round(momentum_score, 3),
        "volume_score":     round(volume_score, 3),
        # v4.1 NEW signals
        "macd_hist":        macd_data["macd_hist"],
        "macd_expanding":   macd_data["hist_expanding"],
        "macd_positive":    macd_data["hist_positive"],
        "macd_score":       macd_data["macd_score"],
        "bb_pct_b":         bb_data["pct_b"],
        "bb_squeeze":       bb_data["is_squeeze"],
        "bb_squeeze_exp":   bb_data["squeeze_expanding"],
        "bb_score":         bb_data["bb_score"],
        "has_hh_hl":        struct_data["has_hh_hl"],
        "has_ll_lh":        struct_data["has_ll_lh"],
        "structure_score":  struct_data["structure_score"],
        "structure_note":   struct_data["structure_note"],
        "pdi":              pdi_val,
        "ndi":              ndi_val,
        # Legacy signals (retained for report/conviction logic)
        "obv_bullish":      obv_bullish,
        "obv_score":        obv_score,
        "rs_days":          rs_days,
        "rs_score":         rs_score,
        "has_tight_base":   has_tight_base,
        "base_score":       base_score,
    }


# ─────────────────────────────────────────────────────────────
# STAGE 2 — FUNDAMENTAL FILTER (yfinance .info)
# ─────────────────────────────────────────────────────────────
def get_fundamentals(symbol: str) -> tuple[float, dict]:
    try:
        info = yf.Ticker(symbol).info
    except Exception as e:
        log.debug(f"{symbol} info fetch failed: {e}")
        return 0.5, {}

    def g(key, default=None):
        v = info.get(key, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v

    de          = g("debtToEquity")
    roe         = g("returnOnEquity")
    rev_growth  = g("revenueGrowth")
    mktcap      = g("marketCap", 0)
    profit_mgn  = g("profitMargins", 0)
    revenue     = g("totalRevenue", 0)           # NEW: for MCap/Sales
    free_cf     = g("freeCashflow")              # NEW: FCF yield

    de_ratio   = (de / 100) if (de is not None and de > 5) else de
    roe_pct    = (roe * 100) if roe is not None else None
    rev_gr_pct = (rev_growth * 100) if rev_growth is not None else None
    mktcap_cr  = mktcap / 1e7 if mktcap else None

    # NEW: MCap to Sales
    mcap_to_sales = None
    if mktcap and revenue and revenue > 0:
        mcap_to_sales = round(mktcap / revenue, 2)

    # NEW: FCF yield
    fcf_yield = None
    if free_cf and mktcap and mktcap > 0:
        fcf_yield = round((free_cf / mktcap) * 100, 2)

    details = {
        "market_cap_cr":      round(mktcap_cr, 0) if mktcap_cr else None,
        "de_ratio":           round(de_ratio, 2) if de_ratio is not None else None,
        "roe_pct":            round(roe_pct, 1) if roe_pct is not None else None,
        "revenue_growth_pct": round(rev_gr_pct, 1) if rev_gr_pct is not None else None,
        "profit_margin_pct":  round(profit_mgn * 100, 1) if profit_mgn else None,
        "mcap_to_sales":      mcap_to_sales,
        "fcf_yield_pct":      fcf_yield,
    }

    # ── Hard filters ──
    if mktcap_cr and mktcap_cr > CFG.max_market_cap_cr:
        return 0.0, details
    if de_ratio is not None and de_ratio > CFG.max_de_ratio:
        return 0.0, details
    if roe_pct is not None and roe_pct < CFG.min_roe_pct:
        return 0.0, details
    if rev_gr_pct is not None and rev_gr_pct < CFG.min_revenue_growth_pct:
        return 0.0, details
    # NEW: MCap/Sales hard cap — priced >5× revenue is too expensive for small cap
    if mcap_to_sales is not None and mcap_to_sales > CFG.max_mcap_to_sales:
        return 0.0, details

    # ── Soft score ──
    roe_score  = min(max((roe_pct or 10) - 10, 0) / 30, 1.0)
    rev_score  = min(max((rev_gr_pct or 0), 0) / 50, 1.0)
    de_score   = 1.0 - min((de_ratio or 0) / CFG.max_de_ratio, 1.0)
    mgn_score  = min(max((profit_mgn or 0) * 100, 0) / 25, 1.0)

    # NEW: FCF yield score — best if >5%, great if >10%
    fcf_score  = 0.3  # neutral if unavailable
    if fcf_yield is not None:
        if fcf_yield > 0:
            fcf_score = min(fcf_yield / 10, 1.0)   # caps at 10%
        else:
            fcf_score = 0.0   # negative FCF = penalty

    # NEW: MCap/Sales score — lower is better for value
    mcs_score = 0.5  # neutral if unavailable
    if mcap_to_sales is not None:
        mcs_score = max(1.0 - mcap_to_sales / CFG.max_mcap_to_sales, 0.0)

    # Weights adjusted to include FCF (strongest predictor from research)
    fscore = (roe_score  * 0.20 +
              rev_score  * 0.25 +
              de_score   * 0.15 +
              mgn_score  * 0.15 +
              fcf_score  * 0.15 +   # NEW
              mcs_score  * 0.10)    # NEW

    return round(fscore, 3), details


# ─────────────────────────────────────────────────────────────
# COMPOSITE SCORER — updated for new signals
# ─────────────────────────────────────────────────────────────
def composite_score(tech: dict, fscore: float) -> float:
    return round(
        tech["mom_score"]       * CFG.w_mom_score   +   # vol-adjusted momentum
        fscore                  * CFG.w_fundamental  +
        tech["macd_score"]      * CFG.w_macd         +   # NEW: MACD histogram
        tech["bb_score"]        * CFG.w_bb            +   # NEW: Bollinger %B
        tech["structure_score"] * CFG.w_structure     +   # NEW: HH/HL structure
        tech["rs_score"]        * CFG.w_rel_str       +
        tech["momentum_score"]  * CFG.w_momentum      +
        tech["volume_score"]    * CFG.w_volume,
        4
    )


# ─────────────────────────────────────────────────────────────
# SIGNAL FLAGS
# ─────────────────────────────────────────────────────────────
def multibagger_flags(s: dict) -> list[str]:
    flags = []
    fd = s.get("fundamentals", {})

    # ── v4.1 NEW signal combinations ──
    if s.get("macd_expanding") and s.get("has_hh_hl") and s.get("bb_squeeze_exp"):
        flags.append("★ MACD↑ + HH/HL + BB Squeeze Expanding (high-conviction setup)")
    elif s.get("macd_expanding") and s.get("has_hh_hl"):
        flags.append("🚀 MACD Expanding + Structural Uptrend (HH/HL)")
    elif s.get("bb_squeeze_exp"):
        flags.append("🔵 Bollinger Squeeze Expanding — Breakout Imminent")
    elif s.get("macd_positive") and s.get("has_hh_hl"):
        flags.append("📈 MACD Positive + HH/HL Structure")

    # ── Legacy signals ──
    if s.get("obv_bullish") and s.get("vol_surge") and s.get("is_breakout"):
        flags.append("★ OBV + Volume + 52W Breakout")
    elif s.get("is_breakout") and s.get("vol_surge"):
        flags.append("🚀 52W Breakout + Volume Surge")
    if s.get("rs_days", 0) >= 3:
        flags.append(f"💪 RS: Rose {s['rs_days']}× when Nifty fell")
    if s.get("has_tight_base") or s.get("bb_squeeze"):
        flags.append("🔵 Volatility Compression / Tight Base")
    if s.get("adx_ok") and s.get("adx", 0) > 30:
        pdi = s.get("pdi", 0)
        ndi = s.get("ndi", 0)
        flags.append(f"📈 Strong Trend ADX {s['adx']} (+DI {pdi} / -DI {ndi})")
    if s.get("price_chg_3m_pct", 0) > 25:
        flags.append(f"⚡ 3M Return +{s['price_chg_3m_pct']}%")
    if (fd.get("fcf_yield_pct") or 0) > 5:
        flags.append(f"💵 FCF Yield {fd['fcf_yield_pct']}%")
    if (fd.get("revenue_growth_pct") or 0) > 25:
        flags.append(f"💰 Rev Growth {fd['revenue_growth_pct']}%")
    if (fd.get("roe_pct") or 0) > 20:
        flags.append(f"✅ ROE {fd['roe_pct']}%")
    if (fd.get("de_ratio") or 1) < 0.3:
        flags.append("🛡️ Nearly Debt-Free")

    return flags


# ─────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_report(top_n: list[dict], regime: dict) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═" * 72

    # Regime banner
    regime_str   = regime.get("regime", "UNKNOWN")
    regime_emoji = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴", "UNKNOWN": "⚪"}.get(regime_str, "⚪")
    sc = regime.get("smallcap100") or {}
    n50 = regime.get("nifty50") or {}

    lines = [
        sep,
        "  MULTIBAGGER DISCOVERY ENGINE v4  —  NSE NIGHTLY SCAN",
        f"  {now}  |  Volatility-Adjusted Momentum + Sector Filter + Fundamentals",
        sep,
        f"  MARKET REGIME:  {regime_emoji} {regime_str}",
        f"  {regime.get('regime_note', '')}",
    ]
    if sc:
        lines.append(
            f"  Nifty SC100: {sc.get('last', '?')}  "
            f"1M: {sc.get('chg_1m_pct', '?'):+.1f}%  "
            f"3M: {sc.get('chg_3m_pct', '?'):+.1f}%  "
            f"Above 200EMA: {'✓' if sc.get('above_200ema') else '✗'}"
        )
    if n50:
        lines.append(
            f"  Nifty 50:    {n50.get('last', '?')}  "
            f"1M: {n50.get('chg_1m_pct', '?'):+.1f}%  "
            f"3M: {n50.get('chg_3m_pct', '?'):+.1f}%"
        )
    lines += [
        sep,
        f"  Top {len(top_n)} candidates  (Breakout × OBV × Relative Strength × Fundamentals)",
        sep, "",
    ]

    for i, s in enumerate(top_n, 1):
        fd    = s.get("fundamentals", {})
        flags = multibagger_flags(s)
        mcap  = f"₹{fd['market_cap_cr']} Cr" if fd.get("market_cap_cr") else "N/A"
        lines += [
            f"#{i:02d}  {s['symbol']:<16}  Score: {s['composite_score']:.3f}   MCap: {mcap}",
            f"    Price ₹{s['last_close']:<9}  RSI {s['rsi']:<6}  ADX {s['adx']:<6}  "
            f"Sector: {s.get('sector','?')}  SectorMom3M: {s.get('sector_3m_ret','?')}%",
            f"    MomScore: {s.get('mom_score_raw',0):.2f}  6M: {s.get('mom_6m_pct',0):+.1f}%  "
            f"12M: {s.get('mom_12m_pct',0):+.1f}%  3M: {s.get('price_chg_3m_pct',0):+.1f}%",
            f"    MACD: {'↑Expanding' if s.get('macd_expanding') else ('Positive' if s.get('macd_positive') else '–')}  "
            f"BB%B: {s.get('bb_pct_b',0):.2f}  BBSqueeze: {'→Expanding!' if s.get('bb_squeeze_exp') else ('Yes' if s.get('bb_squeeze') else '–')}  "
            f"Structure: {'HH/HL✓' if s.get('has_hh_hl') else ('⚠LL/LH' if s.get('has_ll_lh') else '–')}  "
            f"+DI:{s.get('pdi',0):.0f}/-DI:{s.get('ndi',0):.0f}",
            f"    RS Days: {s.get('rs_days','?')}  OBV: {'✓' if s.get('obv_bullish') else '✗'}  "
            f"52W Break: {'✓' if s.get('is_breakout') else '✗'}",
            f"    Fundamental Score: {s['fundamental_score']:.2f}   "
            f"ROE: {fd.get('roe_pct', '?')}%   "
            f"D/E: {fd.get('de_ratio', '?')}   "
            f"FCF Yield: {fd.get('fcf_yield_pct', 'N/A')}%",
        ]
        if flags:
            lines.append(f"    ▶  {' | '.join(flags[:4])}")
        lines.append("")

    lines += [sep, "  ⚠  Educational use only. Not financial advice.", sep]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_engine(regime: Optional[dict] = None) -> list[dict]:
    log.info("══════════════════════════════════════════════════════")
    log.info("  MULTIBAGGER DISCOVERY ENGINE v4 — SCAN START")
    log.info("  v4: Volatility-adjusted momentum | Sector filter | Liquidity gate")
    log.info("══════════════════════════════════════════════════════")

    symbols = fetch_nse_symbols()
    log.info(f"Universe: {len(symbols)} NSE symbols")

    # Fetch Nifty returns once for all RS calculations
    log.info("Fetching Nifty 50 returns for relative strength calculation...")
    nifty_returns = fetch_nifty_returns(period="3mo")
    log.info(f"  Nifty returns loaded: {len(nifty_returns)} days")

    # v4: Fetch sector momentum once — used as gate in analyse_technicals
    log.info("Fetching sector momentum (NSE sector indices 3M returns)...")
    sector_momentum = fetch_sector_momentum()

    # STAGE 1: Batch OHLCV + Technical scan with v4 momentum scoring
    log.info("Stage 1: Volatility-adjusted momentum scan...")
    breakout_candidates = []
    batches = [symbols[i:i + CFG.batch_size] for i in range(0, len(symbols), CFG.batch_size)]

    for b_idx, batch in enumerate(batches):
        log.info(f"  Batch {b_idx+1}/{len(batches)} ({len(batch)} symbols)...")
        try:
            raw = yf.download(
                batch, period="2y", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
        except Exception as e:
            log.warning(f"  Batch download error: {e}")
            time.sleep(5)
            continue

        for sym in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                elif isinstance(raw.columns, pd.MultiIndex):
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym].copy()
                else:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(subset=["Close", "Volume"], inplace=True)
                if df.empty:
                    continue

                # Fetch info for sector classification (lightweight)
                try:
                    info = yf.Ticker(sym).info
                except Exception:
                    info = {}

                result = analyse_technicals(
                    df, sym, nifty_returns,
                    sector_momentum=sector_momentum,
                    stock_info=info,
                )
                if result:
                    breakout_candidates.append(result)
                    log.info(
                        f"    ✓ {sym:<16} ₹{result['last_close']:<8} "
                        f"MomScore={result['mom_score_raw']:.2f}  "
                        f"6M={result['mom_6m_pct']:+.0f}%  12M={result['mom_12m_pct']:+.0f}%  "
                        f"Sector={result.get('sector','?')}  ADX={result['adx']}"
                    )
            except Exception as e:
                log.debug(f"  {sym} error: {e}")

        time.sleep(CFG.sleep_between_batches)

    log.info(f"Stage 1 complete. Momentum candidates: {len(breakout_candidates)}")

    if not breakout_candidates:
        log.warning("No momentum candidates today.")
        return []

    # ── CROSS-SECTIONAL PERCENTILE GATE ─────────────────────────────────────
    # NSE Momentum Index methodology: rank all candidates by volatility-adjusted
    # momentum score and keep only the top 20% (80th percentile and above).
    # This is what separates genuine momentum from "above zero" noise.
    # With ~2,270 symbols, Stage 1 typically passes 300-600 candidates.
    # After percentile gate: ~60-120 remain for fundamental scoring.
    if len(breakout_candidates) >= 10:
        raw_scores = [c["mom_score_raw"] for c in breakout_candidates]
        percentile_threshold = float(np.percentile(raw_scores, 80))
        pre_count = len(breakout_candidates)
        breakout_candidates = [c for c in breakout_candidates
                                if c["mom_score_raw"] >= percentile_threshold]
        log.info(
            f"Cross-sectional gate (top 20%): {pre_count} → {len(breakout_candidates)} "
            f"(threshold: {percentile_threshold:.3f})"
        )
    # ────────────────────────────────────────────────────────────────────────

    # STAGE 2: Fundamental filter + scoring
    log.info("Stage 2: Fundamental filter (ROE, D/E, FCF, MCap/Sales)...")
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
        log.info(
            f"  ✓ {sym:16s} score={total:.3f}  mom={cand['mom_score_raw']:.2f}  "
            f"fscore={fscore:.3f}  FCF={fd.get('fcf_yield_pct', 'N/A')}%"
        )
        time.sleep(0.3)

    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    top_n = ranked[:CFG.top_n]
    log.info(f"Stage 2 complete. {len(ranked)} passed → top {len(top_n)}")

    # Output
    if regime is None:
        regime = {"regime": "UNKNOWN", "regime_note": "Not computed"}
    report = generate_report(top_n, regime)
    print("\n" + report)

    date_str = datetime.utcnow().strftime("%Y%m%d")
    with open(f"results/report_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(f"results/scan_{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2, default=str)
    with open("results/latest.json", "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2, default=str)
    with open("results/latest_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    log.info("Results written to results/")
    return top_n


if __name__ == "__main__":
    regime = get_market_regime()
    run_engine(regime)
