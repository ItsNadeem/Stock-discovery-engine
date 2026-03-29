"""
pre_breakout_scanner.py — Layer 2: Multibagger Discovery Engine v5

KEY CHANGE FROM v4: DNA Score replaces keyword catalyst scoring.
multibagger_dna.py validates signals against 6 confirmed NSE multibaggers.
Filters relaxed — Paushak/EKI/Cosmo Ferrites would have failed v4 filters.
Concall analysis added for forward-looking signals.

See multibagger_dna.py and concall_analyser.py for full documentation.
"""

import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf

from screener_fetcher import enrich_with_screener
from public_data_fetcher import fetch_all_public_signals
from multibagger_dna import compute_dna_score
from concall_analyser import get_concall_signals

log = logging.getLogger(__name__)


@dataclass
class PreBreakoutConfig:
    # Universe — RELAXED. Thin volume = undiscovered = opportunity.
    max_market_cap_cr: float = 800.0   # tighter ceiling (was 2000)
    min_market_cap_cr: float = 20.0
    max_price:         float = 1_500.0 # wider (was 500)
    min_price:         float = 5.0
    min_avg_volume:    int   = 30_000  # was 150k — this was cutting gems

    excluded_industries: list = field(default_factory=lambda: [
        "asset management","investment holding","core investment company",
        "holding company","investment company","closed end fund",
        "venture capital","private equity","financial holding",
        "capital markets","diversified financials",
    ])
    excluded_description_keywords: list = field(default_factory=lambda: [
        "core investment company","holding company","investment in shares",
        "investment in securities","investment holding","holds equity",
        "invests in shares","holding of shares","investment activities",
    ])
    excluded_name_patterns: list = field(default_factory=lambda: [
        "investments ltd","investment ltd","holdings ltd",
        "capital & finance","finance & investment",
    ])
    max_operating_margin_pct:          float = 85.0
    max_believable_revenue_growth_pct: float = 300.0

    trusted_groups: list = field(default_factory=lambda: [
        "murugappa","jk group","jk cement","singhania","alembic","cosmo films",
        "tata","tvs","bajaj","sundaram","godrej","mahindra","birla",
        "pi industries","astral","aarti","deepak nitrite","navin fluorine",
        "galaxy surfactants","alkyl amines","vinati organics",
    ])

    announcement_lookback_days: int = 90

    # Composite weights v5
    w_dna:       float = 0.45
    w_concall:   float = 0.15
    w_promoter:  float = 0.12
    w_piotroski: float = 0.10
    w_lynch:     float = 0.08
    w_public:    float = 0.06
    w_debt:      float = 0.04

    top_n: int = 15


PCFG = PreBreakoutConfig()
BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
}


def should_exclude(info: dict) -> tuple[bool, str]:
    industry     = (info.get("industry") or "").lower()
    sector       = (info.get("sector") or "").lower()
    summary      = (info.get("longBusinessSummary") or "").lower()
    company_name = (info.get("longName") or "").lower()
    for excl in PCFG.excluded_industries:
        if excl in industry or excl in sector:
            return True, f"Excluded industry: '{excl}'"
    for kw in PCFG.excluded_description_keywords:
        if kw in summary:
            return True, f"Holding company keyword: '{kw}'"
    for pattern in PCFG.excluded_name_patterns:
        if pattern in company_name:
            return True, f"Holding company name: '{pattern}'"
    if (info.get("operatingMargins") or 0) * 100 > PCFG.max_operating_margin_pct:
        return True, "Passive income"
    if (info.get("revenueGrowth") or 0) * 100 > PCFG.max_believable_revenue_growth_pct:
        return True, "Data artefact"
    return False, ""


def fetch_bse_announcements(symbol_code: str, days_back: int = 90) -> list[dict]:
    cutoff = datetime.now() - timedelta(days=days_back)
    result, seen = [], set()
    for stype in ["P", "C"]:
        try:
            url = (
                f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
                f"?strCat=-1&strPrevDate=&strScrip={symbol_code}"
                f"&strSearch={stype}&strToDate=&strType=C&subcategory=-1"
            )
            r = requests.get(url, headers=BSE_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            for ann in r.json().get("Table", []):
                try:
                    dt = datetime.strptime(ann.get("News_submission_dt", "")[:10], "%Y-%m-%d")
                    if dt >= cutoff:
                        title = ann.get("NEWSSUB", "").strip()
                        if title and title not in seen:
                            seen.add(title)
                            result.append({
                                "date":     dt.strftime("%Y-%m-%d"),
                                "title":    title,
                                "category": ann.get("CATEGORYNAME", "").strip(),
                            })
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"BSE {stype} {symbol_code}: {e}")
    log.debug(f"BSE {symbol_code}: {len(result)} announcements")
    return result


def get_bse_code(symbol: str) -> Optional[str]:
    sym_clean = symbol.replace(".NS", "")
    try:
        url = (
            f"https://api.bseindia.com/BseIndiaAPI/api/fetchCompanyList/w"
            f"?marketcap=&industry=&status=Active&scripcode="
            f"&companyname={sym_clean}&segment=Equity"
        )
        r = requests.get(url, headers=BSE_HEADERS, timeout=8)
        if r.status_code == 200:
            items = r.json() if isinstance(r.json(), list) else r.json().get("Table", [])
            for item in items:
                if (sym_clean.lower() in str(item.get("SCRIP_CD", "")).lower() or
                        sym_clean.lower() in str(item.get("Issuer_Name", "")).lower()):
                    return str(item.get("SCRIP_CD", ""))
    except Exception:
        pass
    return None


def get_sector(info: dict) -> str:
    s = (info.get("sector") or "").lower()
    if any(x in s for x in ["financial","bank","insurance"]):   return "Financial Services"
    if any(x in s for x in ["technology","software","it"]):     return "IT"
    if any(x in s for x in ["health","pharma","drug","bio"]):   return "Pharma"
    if any(x in s for x in ["auto","vehicle"]):                 return "Auto"
    if any(x in s for x in ["metal","steel","alumin"]):         return "Metals"
    if any(x in s for x in ["energy","oil","gas","power"]):     return "Energy"
    if any(x in s for x in ["consumer","food","fmcg"]):         return "FMCG"
    if any(x in s for x in ["real estate","realt","construct"]): return "Realty"
    if any(x in s for x in ["infra","cement","engineer"]):      return "Infra"
    return "Other"


def get_trusted_group(info: dict) -> Optional[str]:
    text = " ".join([info.get(k,"") or "" for k in
                     ["longName","sector","industry","longBusinessSummary"]]).lower()
    for g in PCFG.trusted_groups:
        if g.lower() in text:
            return g.title()
    return None


def classify_price_stage(info: dict, hist: pd.DataFrame) -> str:
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    if price == 0 or hist is None or hist.empty or len(hist) < 100:
        return "UNKNOWN"
    c = hist["Close"]
    n = min(252, len(c))
    h = float(c.rolling(n).max().iloc[-1])
    l = float(c.rolling(n).min().iloc[-1])
    fl = (price - l) / l * 100 if l > 0 else 0
    fh = (h - price) / h * 100 if h > 0 else 0
    if fl < 25:             return "🟢 DEEP BASE"
    elif fl < 60 and fh>30: return "🟡 EARLY UPTREND"
    elif fh < 10:           return "🔴 NEAR 52W HIGH"
    return "🟠 MID CYCLE"


def calc_piotroski(ticker: yf.Ticker, info: dict) -> tuple[float, Optional[int]]:
    try:
        bs, fin, cf = ticker.balance_sheet, ticker.financials, ticker.cashflow
        if any(x is None or (hasattr(x,'empty') and x.empty) for x in [bs,fin,cf]):
            return 0.5, None
        def row(df, *keys):
            for k in keys:
                for idx in df.index:
                    if k.lower() in str(idx).lower():
                        v = df.loc[idx].iloc[:2]
                        return (
                            float(v.iloc[0]) if not pd.isna(v.iloc[0]) else None,
                            float(v.iloc[1]) if len(v)>1 and not pd.isna(v.iloc[1]) else None,
                        )
            return None, None
        ta_n,ta_p  = row(bs,"total assets")
        ltd_n,ltd_p= row(bs,"long term debt","longterm debt")
        ca_n,ca_p  = row(bs,"current assets","total current assets")
        cl_n,cl_p  = row(bs,"current liabilities","total current liabilities")
        sh_n,sh_p  = row(bs,"ordinary shares","common stock","share issued")
        ni_n,ni_p  = row(fin,"net income")
        gp_n,gp_p  = row(fin,"gross profit")
        rv_n,rv_p  = row(fin,"total revenue","revenue")
        oc_n,_     = row(cf,"operating cash flow","total cash from operating")
        roa_n = ni_n/ta_n if ni_n and ta_n else None
        roa_p = ni_p/ta_p if ni_p and ta_p else None
        score = sum([
            1 if roa_n and roa_n>0 else 0,
            1 if oc_n and oc_n>0 else 0,
            1 if roa_n and roa_p and roa_n>roa_p else 0,
            1 if oc_n and ta_n and roa_n and (oc_n/ta_n)>roa_n else 0,
            1 if (ltd_n/ta_n if ltd_n and ta_n else 0)<(ltd_p/ta_p if ltd_p and ta_p else 0) else 0,
            1 if ca_n and cl_n and ca_p and cl_p and (ca_n/cl_n)>(ca_p/cl_p) else 0,
            1 if sh_n and sh_p and sh_n<=sh_p*1.01 else 0,
            1 if gp_n and rv_n and gp_p and rv_p and (gp_n/rv_n)>(gp_p/rv_p) else 0,
            1 if rv_n and ta_n and rv_p and ta_p and (rv_n/ta_n)>(rv_p/ta_p) else 0,
        ])
        return round(score/9, 3), score
    except Exception:
        return 0.5, None


def calc_lynch(info: dict) -> float:
    pe  = info.get("trailingPE") or info.get("forwardPE")
    rev = info.get("revenueGrowth")
    inst= (info.get("heldPercentInstitutions") or 0)*100
    peg_s = 0.3
    if pe and 0<pe<200 and rev and rev>0.05:
        peg = pe/(rev*100)
        peg_s = 1.0 if peg<0.5 else 0.85 if peg<1.0 else 0.60 if peg<1.5 else 0.35 if peg<2.0 else 0.10
    elif pe and pe<12:
        peg_s = 0.55
    inst_s = 1.0 if inst<3 else 0.85 if inst<8 else 0.60 if inst<20 else 0.30 if inst<40 else 0.10
    return round(peg_s*0.6 + inst_s*0.4, 3)


def calc_debt(info: dict) -> float:
    debt  = info.get("totalDebt") or 0
    cash  = info.get("totalCash") or 0
    book  = info.get("bookValue") or 0
    shares= info.get("sharesOutstanding") or 0
    eq    = book*shares if shares>0 else 0
    de    = debt/eq if eq>0 else 0
    if debt<=cash: return 1.0
    if de<=0.15:   return 0.80
    if de<=0.50:   return 0.65
    if de<=1.00:   return 0.45
    if de<=2.00:   return 0.25
    return 0.10


def analyse_pre_breakout(symbol, nse_session=None, universe_deals=None):
    sym_clean = symbol.replace(".NS","")
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}
    except Exception as e:
        log.debug(f"{symbol}: {e}")
        return None
    if not info or not info.get("marketCap"):
        return None

    mktcap_cr = (info.get("marketCap") or 0) / 1e7
    price     = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    avg_vol   = info.get("averageVolume") or info.get("averageDailyVolume10Day") or 0

    if not (PCFG.min_market_cap_cr <= mktcap_cr <= PCFG.max_market_cap_cr): return None
    if not (PCFG.min_price <= price <= PCFG.max_price):                      return None
    if avg_vol < PCFG.min_avg_volume:                                         return None

    excl, reason = should_exclude(info)
    if excl:
        log.debug(f"  {sym_clean}: {reason}")
        return None

    try:    hist = ticker.history(period="2y", interval="1d")
    except: hist = pd.DataFrame()

    price_stage = classify_price_stage(info, hist)
    if "NEAR 52W HIGH" in price_stage:
        return None

    sector   = get_sector(info)
    bse_code = get_bse_code(symbol)
    anns     = fetch_bse_announcements(bse_code, PCFG.announcement_lookback_days) if bse_code else []

    public_data = None
    if universe_deals is not None:
        try:
            public_data = fetch_all_public_signals(symbol, info, universe_deals, session=nse_session)
        except Exception as e:
            log.debug(f"  {sym_clean}: public data: {e}")

    if public_data and (public_data.get("pledge_pct") or 0) > 30:
        log.debug(f"  {sym_clean}: high pledge — skipped")
        return None

    pio_01, pio_raw = calc_piotroski(ticker, info)
    lynch_s         = calc_lynch(info)
    debt_s          = calc_debt(info)
    group           = get_trusted_group(info)

    dna = compute_dna_score(
        symbol=symbol, market_cap_cr=mktcap_cr, info=info, hist=hist,
        screener_data=None, public_data=public_data,
        announcements=anns, sector=sector,
    )

    has_thesis = (
        dna["dna_score"] >= 0.45
        or dna["promoter_score"] >= 0.85
        or dna["catalyst_score"] >= 0.65
    )
    if not has_thesis:
        log.debug(f"  {sym_clean}: DNA {dna['dna_score']:.2f} — skipped")
        return None

    public_score = (public_data or {}).get("public_score", 0.3)
    composite = (
        dna["dna_score"]       * PCFG.w_dna      +
        0.3                    * PCFG.w_concall   +  # placeholder
        dna["promoter_score"]  * PCFG.w_promoter  +
        pio_01                 * PCFG.w_piotroski  +
        lynch_s                * PCFG.w_lynch      +
        public_score           * PCFG.w_public     +
        debt_s                 * PCFG.w_debt
    )

    flags = list(dna["dna_flags"])
    if group: flags.append(f"🏛️ {group} Group")
    if public_data: flags.extend(public_data.get("public_flags", []))

    return {
        "symbol": symbol, "price": round(price,2),
        "market_cap_cr": round(mktcap_cr,0), "avg_volume": int(avg_vol),
        "price_stage": price_stage, "sector": sector,
        "composite_score": round(composite,4),
        "scanned_at": datetime.utcnow().isoformat(),
        "dna": dna,
        "piotroski_score": pio_raw, "piotroski_01": pio_01,
        "lynch_score": lynch_s, "debt_score": debt_s,
        "group": group, "public_data": public_data,
        "bse_code": bse_code, "announcement_count": len(anns),
        "concall": None, "screener": None,
        "screener_flags": [], "screener_roce": None,
        "all_flags": flags,
    }


def run_pre_breakout_scanner(symbols, nse_session=None, universe_deals=None, regime=None):
    log.info(f"Layer 2 v5: scanning {len(symbols)} symbols...")
    candidates = []
    for i, symbol in enumerate(symbols):
        try:
            r = analyse_pre_breakout(symbol, nse_session=nse_session, universe_deals=universe_deals)
            if r:
                candidates.append(r)
                log.info(
                    f"  [{i+1}/{len(symbols)}] ✓ {symbol} "
                    f"DNA={r['dna']['dna_score']:.3f}({r['dna']['dna_grade'][:1]}) "
                    f"P={r['piotroski_score']}/9 anns={r['announcement_count']}"
                )
        except Exception as e:
            log.debug(f"  {symbol}: {e}")
        if (i+1) % 50 == 0:
            time.sleep(1.5)

    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    top = candidates[:PCFG.top_n * 2]

    log.info(f"\nEnriching top {len(top)} with Screener.in...")
    enriched = enrich_with_screener(top, max_candidates=len(top))

    log.info(f"Fetching concall signals for top {min(len(enriched),20)}...")
    for c in enriched[:20]:
        try:
            concall = get_concall_signals(c["symbol"], sleep=1.0)
            c["concall"] = concall
            concall_score = concall.get("concall_score", 0.3)

            sc_data = c.get("screener")
            if sc_data:
                dna_r = compute_dna_score(
                    symbol=c["symbol"], market_cap_cr=c["market_cap_cr"],
                    info={}, hist=pd.DataFrame(), screener_data=sc_data,
                    public_data=c.get("public_data"), announcements=[], sector=c["sector"],
                )
                c["dna"]["dna_score"]      = round(c["dna"]["dna_score"]*0.4 + dna_r["dna_score"]*0.6, 4)
                c["dna"]["rev_accel_score"] = dna_r["rev_accel_score"]
                c["dna"]["rev_accel_desc"]  = dna_r["rev_accel_desc"]
                c["dna"]["dna_grade"]       = dna_r["dna_grade"]
                c["dna"]["dna_flags"]       = dna_r["dna_flags"]

            ps = (c.get("public_data") or {}).get("public_score", 0.3)
            c["composite_score"] = round(
                c["dna"]["dna_score"]      * PCFG.w_dna      +
                concall_score              * PCFG.w_concall   +
                c["dna"]["promoter_score"] * PCFG.w_promoter  +
                c["piotroski_01"]          * PCFG.w_piotroski +
                c["lynch_score"]           * PCFG.w_lynch     +
                ps                         * PCFG.w_public    +
                c["debt_score"]            * PCFG.w_debt,
                4,
            )
            if concall.get("concall_flag"):
                c["all_flags"].insert(0, concall["concall_flag"])
            log.info(f"  🎙️ {c['symbol']}: concall={concall_score:.2f} final={c['composite_score']:.3f}")
        except Exception as e:
            log.debug(f"  {c['symbol']}: concall: {e}")

    enriched.sort(key=lambda x: x["composite_score"], reverse=True)
    final = enriched[:PCFG.top_n]
    log.info(f"Layer 2 v5: {len(candidates)} passed → {len(final)} final")
    return final


def generate_pre_breakout_report(candidates: list[dict]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "═" * 72
    lines = [
        sep,
        f"  LAYER 2 v5: MULTIBAGGER WATCHLIST — {now}",
        "  DNA Score validated against confirmed NSE multibaggers",
        sep, "",
    ]
    if not candidates:
        lines.append("  No candidates today.")
        return "\n".join(lines)

    for i, s in enumerate(candidates, 1):
        dna     = s["dna"]
        sc      = s.get("screener") or {}
        pub     = s.get("public_data") or {}
        concall = s.get("concall") or {}
        sf      = s.get("screener_flags") or []
        roce    = f"ROCE {sc.get('roce_pct'):.1f}%" if sc.get("roce_pct") else "ROCE N/A"

        lines += [
            f"  #{i:02d} {s['symbol']:<16} ₹{s['price']:<8} ₹{s['market_cap_cr']:.0f}Cr"
            f"  Score:{s['composite_score']:.3f}",
            f"  {s['price_stage']}  Sector:{s['sector']}",
            f"  DNA:{dna['dna_score']:.3f} [{dna['dna_grade']}]",
            f"  RevAccel:{dna['rev_accel_score']:.2f}  Promoter:{dna['promoter_score']:.2f}"
            f"  Catalyst:{dna['catalyst_score']:.2f}  MCap:{dna['mcap_score']:.2f}",
            f"  Piotroski:{s['piotroski_score']}/9  {roce}  Debt:{s['debt_score']:.2f}",
        ]
        if s.get("group"):
            lines.append(f"  Group: 🏛️ {s['group']}")
        if dna["dna_flags"]:
            for f in dna["dna_flags"][:3]:
                lines.append(f"  ★ {f}")
        if concall.get("tier1_signals"):
            lines.append(f"  🎙️ Concall: {concall['tier1_signals'][0][:80]}")
            if concall.get("manual_review"):
                lines.append("    ★ READ FULL CONCALL TRANSCRIPT")
        elif concall.get("concall_flag"):
            lines.append(f"  🎙️ {concall['concall_flag']}")
        if sf:
            lines.append(f"  Screener: {' | '.join(sf[:2])}")
        pub_p = []
        if pub.get("promoter_buying"):  pub_p.append(f"🧑‍💼 Insider ₹{pub.get('insider_value_cr',0):.1f}Cr")
        if pub.get("institutional_buying"): pub_p.append("🏦 Bulk Buy")
        pl = pub.get("pledge_pct")
        if pl is not None: pub_p.append("✅ No Pledge" if pl<=0 else f"Pledge {pl:.0f}%")
        if pub_p: lines.append(f"  Public: {' | '.join(pub_p)}")
        lines.append(f"  BSE anns:{s['announcement_count']}  Vol:{s['avg_volume']:,}")
        lines.append("")

    lines += [
        sep,
        "  DNA Grade A (≥0.75) = matches confirmed multibagger pattern",
        "  RevAccel ≥0.80 = revenue inflection (most predictive signal)",
        "  ★ READ CONCALL = specific forward guidance found — act on this",
        "  🧑‍💼 Insider buying = highest conviction, open-market purchase",
        sep,
        "  ⚠ Educational only. Not financial advice.",
        sep,
    ]
    return "\n".join(lines)
