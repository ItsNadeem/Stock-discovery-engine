"""
universe.py — Fetch & cache the NSE equity universe
Sources symbols from NSE's public CSV (no API key needed).
Filters to small/micro cap price range for multibagger focus.
"""

import requests
import pandas as pd
import logging
import os

log = logging.getLogger(__name__)

# NSE publishes this publicly — no auth needed
NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_nse_symbols(cache_path: str = "nse_symbols.csv") -> list[str]:
    """
    Returns list of NSE symbols formatted for yfinance (e.g. 'INFY.NS').
    Uses cached file if available and fresh (same-day).
    """
    # Use cache if it exists and is from today
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        import time
        if time.time() - mtime < 86400:  # 24 hours
            df = pd.read_csv(cache_path)
            symbols = df["yf_symbol"].tolist()
            log.info(f"Loaded {len(symbols)} symbols from cache")
            return symbols

    log.info("Fetching NSE equity list from NSE archives...")
    try:
        r = requests.get(NSE_EQUITY_LIST_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:
        log.warning(f"NSE fetch failed ({e}), falling back to embedded list")
        return _fallback_symbols()

    # NSE CSV has column ' SYMBOL' (with space) or 'SYMBOL'
    df.columns = [c.strip() for c in df.columns]
    sym_col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]

    df["symbol"] = df[sym_col].str.strip()
    df["yf_symbol"] = df["symbol"] + ".NS"

    # Save cache
    df[["symbol", "yf_symbol"]].to_csv(cache_path, index=False)
    symbols = df["yf_symbol"].tolist()
    log.info(f"Fetched {len(symbols)} NSE symbols")
    return symbols


def _fallback_symbols() -> list[str]:
    """
    Hardcoded fallback — ~200 liquid small/mid caps for testing.
    Extend this list or replace with your own watchlist.
    """
    base = [
        "ANGELONE", "APTUS", "ASTERDM", "ASTRAZEN", "AXISBANK",
        "BAJFINANCE", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BATAINDIA",
        "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL",
        "BIKAJI", "BLUEDART", "BSOFT", "CAMS", "CANFINHOME",
        "CAPLIPOINT", "CARBORUNIV", "CASTROLIND", "CEATLTD", "CENTRALBK",
        "CENTURYTEX", "CHOLAFIN", "CIGNITI", "CLEAN", "COFORGE",
        "CREDITACC", "CSBBANK", "CYIENT", "DATAPATTNS", "DCMSHRIRAM",
        "DEEPAKFERT", "DEEPAKNTR", "DELTACORP", "DELHIVERY", "DEVYANI",
        "DIXON", "DLINKINDIA", "DMART", "EASEMYTRIP", "ECLERX",
        "EIDPARRY", "ELGIEQUIP", "EMAMILTD", "ENGINERSIN", "EPIGRAL",
        "EQUITASBNK", "EROSMEDIA", "ESABINDIA", "ESCORTS", "EXIDEIND",
        "FINEORG", "FINPIPE", "FORCEMOT", "FORTIS", "FSL",
        "GALAXYSURF", "GLAND", "GLAXO", "GMRINFRA", "GNFC",
        "GODFRYPHLP", "GODREJCP", "GODREJIND", "GODREJPROP", "GRANULES",
        "GRAPHITE", "GRASIM", "GREAVESCOT", "GRINDWELL", "GUJGASLTD",
        "GULFOILLUB", "HAL", "HAPPSTMNDS", "HATSUN", "HAVELLS",
        "HDFCAMC", "HDFCBANK", "HEG", "HERANBA", "HEROMOTOCO",
        "HFCL", "HIKAL", "HINDCOPPER", "HINDPETRO", "HINDUNILVR",
        "HONAUT", "HUDCO", "IBREALEST", "ICICIBANK", "ICICIGI",
        "ICICIPRULI", "IDFCFIRSTB", "IEX", "IFBIND", "IGL",
        "IMFA", "INDIANB", "INDIAMART", "INDOCO", "INDUSINDBK",
        "INFIBEAM", "INFY", "INTELLECT", "IOC", "IPCALAB",
        "IRB", "IRCTC", "IRFC", "ISEC", "ITC",
        "JINDALSTEL", "JKCEMENT", "JKLAKSHMI", "JMFINANCIL", "JSWENERGY",
        "JUBLFOOD", "JUBLINGREA", "JUSTDIAL", "JYOTHYLAB", "KAJARIACER",
        "KANSAINER", "KEC", "KFINTECH", "KNRCON", "KOTAKBANK",
        "KPITTECH", "KRBL", "KSCL", "L&TFH", "LALPATHLAB",
        "LAURUSLABS", "LEMONTREE", "LICHOUSING", "LINDEINDIA", "LTIM",
        "LTTS", "LUXIND", "MAHINDCIE", "MARICO", "MARUTI",
        "MAXHEALTH", "MAZDOCK", "MCX", "MEDANTA", "METROPOLIS",
        "MINDTREE", "MIDHANI", "MMTC", "MPHASIS", "MRF",
        "MUTHOOTFIN", "NAUKRI", "NAVINFLUOR", "NESCO", "NESTLEIND",
        "NETWORK18", "NIFTYBEES", "NILKAMAL", "NLCINDIA", "NMDC",
        "NOCIL", "NUVOCO", "OBEROIRLTY", "OFSS", "OIL",
        "OLECTRA", "OMAXE", "ONGC", "PAGEIND", "PAISALO",
        "PARADEEP", "PARAS", "PCBL", "PERSISTENT", "PETRONET",
        "PFIZER", "PGEL", "PHOENIXLTD", "PIDILITIND", "PIIND",
        "PNBHOUSING", "POLYCAB", "POLYMED", "PRAJ", "PRINCEPIPE",
        "PRIVISCL", "PRSMJOHNSN", "PTC", "PVRINOX", "RAILTEL",
        "RAIN", "RAJESHEXPO", "RALLIS", "RAMCOCEM", "RAMCOIND",
        "RBLBANK", "REDINGTON", "RELAXO", "RELIANCE", "RITES",
        "ROUTE", "RPGLIFE", "SAFARI", "SAIL", "SANOFI",
        "SAPPHIRE", "SAREGAMA", "SCHAEFFLER", "SEQUENT", "SHARDACROP",
        "SHREECEM", "SHREDIGCEM", "SIEMENS", "SJVN", "SKIPPER",
        "SKFINDIA", "SOBHA", "SOLARINDS", "SONACOMS", "SPANDANA",
        "SPARC", "STLTECH", "SUDARSCHEM", "SUMICHEM", "SUNTV",
        "SUPRAJIT", "SUPREMEIND", "SUVENPHAR", "SUZLON", "TANLA",
        "TATACHEM", "TATACOMM", "TATACONSUMER", "TATAELXSI", "TATAMOTORS",
        "TATAPOWER", "TATASTEEL", "TCI", "TCNSBRANDS", "TEAMLEASE",
        "TECHNOE", "TEJASNET", "THERMAX", "TIMKEN", "TIMETECHNO",
        "TITAGARH", "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT",
        "TRIVENI", "TTKPRESTIG", "TUBE", "TVSSCS", "TVTODAY",
        "UCOBANK", "UJJIVANSFB", "ULTRACEMCO", "UNIONBANK", "UNOMINDA",
        "UPL", "UTIAMC", "VAIBHAVGBL", "VARDHANCRL", "VARROC",
        "VBL", "VEDL", "VENKEYS", "VINATIORGA", "VOLTAMP",
        "VOLTAS", "WELCORP", "WHIRLPOOL", "WIPRO", "WOCKPHARMA",
        "WONDERLA", "XCHANGING", "YATHARTH", "ZEEL", "ZENSARTECH",
        "ZENTEC", "ZOMATO", "ZYDUSLIFE",
    ]
    return [s + ".NS" for s in base]
