"""Write deterministic synthetic bar inputs for the Market Lab golden tests (numpy only).

Run: .venv\\Scripts\\python.exe scripts\\golden\\make_inputs.py tests\\fixtures\\marketlab
Outputs inputs_5m_stock.json, inputs_5m_index.json (volume 0), inputs_1d_stock.json, inputs_15m_stock.json and
inputs_1d_from5m.json. Each is a dict of equal-length lists: ts, open, high, low, close, volume
with ts in epoch seconds (UTC) at real IST session times.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

IST_OFFSET = 19800
OPEN_MIN = 9 * 60 + 15


def sessions(start: date, n: int) -> list[date]:
    """First n weekdays from `start`."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def epoch(d: date, minute: int) -> int:
    """Epoch second of IST date `d` at `minute` minutes after IST midnight."""
    return (d - date(1970, 1, 1)).days * 86400 + minute * 60 - IST_OFFSET


def make_bars(
    ts: list[int], sigma: float, base: float, seed: int, with_volume: bool
) -> dict[str, list[float]]:
    """GBM closes; high/low extend the open-close range by |N(0, .0008)|; lognormal volume."""
    rng = np.random.default_rng(seed)
    n = len(ts)
    close = base * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    open_ = np.r_[base, close[:-1]]
    wick = 0.0008 if sigma < 0.01 else 0.004
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, wick, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, wick, n)))
    vol = np.exp(rng.normal(10.0, 0.6, n)) if with_volume else np.zeros(n)
    return {
        "ts": [int(t) for t in ts],
        "open": [round(float(x), 6) for x in open_],
        "high": [round(float(x), 6) for x in high],
        "low": [round(float(x), 6) for x in low],
        "close": [round(float(x), 6) for x in close],
        "volume": [round(float(x), 2) for x in vol],
    }


def aggregate(b: dict[str, list[float]], per: int) -> dict[str, list[float]]:
    """Merge every `per` consecutive bars (bars of one day must divide evenly)."""
    n = len(b["ts"]) // per
    out: dict[str, list[float]] = {k: [] for k in b}
    for i in range(n):
        a, z = i * per, (i + 1) * per
        out["ts"].append(b["ts"][a])
        out["open"].append(b["open"][a])
        out["high"].append(max(b["high"][a:z]))
        out["low"].append(min(b["low"][a:z]))
        out["close"].append(b["close"][z - 1])
        out["volume"].append(round(sum(b["volume"][a:z]), 2))
    return out


def daily_from(b: dict[str, list[float]]) -> dict[str, list[float]]:
    """One bar per IST date from intraday bars (open of first, high, low, close of last, volume sum)."""
    days = [(t + IST_OFFSET) // 86400 for t in b["ts"]]
    out: dict[str, list[float]] = {k: [] for k in b}
    for d in sorted(set(days)):
        idx = [i for i, x in enumerate(days) if x == d]
        out["ts"].append(b["ts"][idx[0]])
        out["open"].append(b["open"][idx[0]])
        out["high"].append(max(b["high"][i] for i in idx))
        out["low"].append(min(b["low"][i] for i in idx))
        out["close"].append(b["close"][idx[-1]])
        out["volume"].append(round(sum(b["volume"][i] for i in idx), 2))
    return out


TAPE_STOCKS = [
    ("RELIANCE", "Oil Gas & Consumable Fuels"),
    ("ONGC", "Oil Gas & Consumable Fuels"),
    ("TCS", "Information Technology"),
    ("INFY", "Information Technology"),
    ("HDFCBANK", "Financial Services"),
    ("ICICIBANK", "Financial Services"),
    ("BAJFINANCE", "Financial Services"),
    ("SBIN", "Financial Services"),
    ("SUNPHARMA", "Healthcare"),
    ("APOLLOHOSP", "Healthcare"),
    ("TATASTEEL", "Metals & Mining"),
    ("MARUTI", "Automobile and Auto Components"),
    ("NOOPEN", "Realty"),
    ("NOPREV", "Realty"),
    ("NODLY", "Power"),
]
TAPE_DATE = date(2026, 11, 12)
TAPE_NOW_MINUTE = 120  # 11:15 IST


def make_tape(out_dir: Path) -> None:
    """One synthetic live moment: F&O meta, daily levels, industries and 1-minute closes of 18 symbols."""
    rng = np.random.default_rng(21)
    n = TAPE_NOW_MINUTE + 1
    ts = [epoch(TAPE_DATE, OPEN_MIN + k) for k in range(n)]
    meta, daily, industry, spark = {}, {}, {}, {}
    for sym, ind in TAPE_STOCKS:
        base = float(rng.uniform(100, 3000))
        drift = float(rng.normal(0, 0.0004))
        closes = base * np.exp(np.cumsum(rng.normal(drift, 0.0007, n)))
        prev = base * float(np.exp(rng.normal(0, 0.01)))
        opn = float(closes[0] * (1 + rng.normal(0, 0.002)))
        industry[sym] = ind
        meta[sym] = {
            "name": sym.title(),
            "open": round(opn, 2),
            "high": round(float(closes.max()) * 1.002, 2),
            "low": round(float(closes.min()) * 0.998, 2),
            "prev": round(prev, 2),
            "ltp": round(float(closes[-1]), 2),
            "value_cr": round(float(rng.uniform(5, 900)), 1),
            "ffmc": round(float(rng.uniform(1e9, 1e12)), 0),
        }
        if sym != "NODLY":
            hi = prev * 1.02
            daily[sym] = {
                "pdh": round(hi, 2),
                "pdl": round(prev * 0.985, 2),
                "pdc": round(prev, 2),
                "range": round(float(rng.uniform(1.2, 3.0)), 3),
            }
        if sym == "NOOPEN":
            meta[sym]["open"] = None
        if sym == "NOPREV":
            meta[sym]["prev"] = None
        spark[sym + ".NS"] = {
            "ts": ts,
            "close": [round(float(c), 2) for c in closes],
            "ltp": round(float(closes[-1]), 2),
            "prev": None if sym == "NOPREV" else round(prev, 2),
            "time": ts[-1],
        }
    for y, base in {
        "NIFTY": ("^NSEI", 24500.0),
        "BANKNIFTY": ("^NSEBANK", 52000.0),
        "SENSEX": ("^BSESN", 80000.0),
    }.values():
        closes = base * np.exp(np.cumsum(rng.normal(0.00002, 0.0004, n)))
        spark[y] = {
            "ts": ts,
            "close": [round(float(c), 2) for c in closes],
            "ltp": round(float(closes[-1]), 2),
            "prev": round(base, 2),
            "time": ts[-1] - 1,
        }
    (out_dir / "inputs_tape.json").write_text(
        json.dumps(
            {
                "now_epoch": ts[-1] + 30,
                "now_minute": TAPE_NOW_MINUTE,
                "meta": meta,
                "daily": daily,
                "industry": industry,
                "spark": spark,
            }
        ),
        encoding="utf-8",
    )
    print("inputs_tape", len(spark))


SECTOR_INDEX = {
    "Railways": "NIFTY INDIA RAILWAYS PSU",
    "Energy": "NIFTY ENERGY",
    "Infra": "NIFTY INFRASTRUCTURE",
    "Cons Durables": "NIFTY CONSUMER DURABLES",
    "Metal": "NIFTY METAL",
    "Tourism": "NIFTY INDIA TOURISM",
    "Defence": "NIFTY INDIA DEFENCE",
    "Auto": "NIFTY AUTO",
    "Chemicals": "NIFTY CHEMICALS",
    "PSU Bank": "NIFTY PSU BANK",
    "Cap Markets": "NIFTY CAPITAL MARKETS",
    "Realty": "NIFTY REALTY",
    "FMCG": "NIFTY FMCG",
    "Fin Services": "NIFTY FINANCIAL SERVICES",
    "Healthcare": "NIFTY HEALTHCARE INDEX",
    "IT": "NIFTY IT",
    "Bank": "NIFTY BANK",
    "Pharma": "NIFTY PHARMA",
    "Media": "NIFTY MEDIA",
    "Cement": "NIFTY CEMENT",
}
# indices Yahoo carried in Market Lab get a stored history; the other sectors use the median of members
HIST_INDEX = [
    "NIFTY 50",
    "NIFTY BANK",
    "NIFTY IT",
    "NIFTY AUTO",
    "NIFTY PHARMA",
    "NIFTY FMCG",
    "NIFTY METAL",
    "NIFTY REALTY",
    "NIFTY ENERGY",
    "NIFTY PSU BANK",
    "NIFTY MEDIA",
    "NIFTY INFRASTRUCTURE",
    "NIFTY FINANCIAL SERVICES",
    "NIFTY HEALTHCARE INDEX",
    "NIFTY CONSUMER DURABLES",
]
SECTOR_PLAN = [
    (
        "ENG",
        14,
        "Oil Gas & Consumable Fuels",
        "Oil Gas & Consumable Fuels",
        ["Refineries & Marketing", "Gas Distribution"],
    ),
    ("ITS", 14, "Information Technology", "IT - Software", ["Computers - Software & Consulting"]),
    ("PHM", 14, "Healthcare", "Pharmaceuticals & Biotechnology", ["Pharmaceuticals"]),
    ("BNK", 14, "Financial Services", "Banks", ["Private Sector Bank", "Public Sector Bank"]),
    ("NBF", 6, "Financial Services", "Finance", ["Non Banking Financial Company (NBFC)"]),
    ("HOS", 6, "Healthcare", "Healthcare Services", ["Hospital"]),
]


def make_sector(out_dir: Path) -> None:
    """Universe, daily history, live state and headlines for the Sector Map golden tests (seed 21)."""
    rng = np.random.default_rng(21)
    days = sessions(date(2025, 9, 1), 300)
    dates = [d.isoformat() for d in days]
    hist, cls, lst, stocks = {}, {}, {}, {}
    for prefix, n, sector, industry, basics in SECTOR_PLAN:
        for i in range(1, n + 1):
            sym = f"{prefix}{i:02d}"
            m = 45 if sym in ("ENG14", "ITS14") else 300
            c = rng.uniform(100, 2000) * np.exp(np.cumsum(rng.normal(0.0004, 0.015, m)))
            h = c * (1 + np.abs(rng.normal(0, 0.01, m)))
            v = np.exp(rng.normal(12.0, 0.5, m)).round()
            hist[sym] = {
                "n": m,
                "close": [round(float(x), 4) for x in c],
                "high": [round(float(x), 4) for x in h],
                "volume": [float(x) for x in v],
            }
            ltp = float(c[-1] * (1 + rng.normal(0, 0.01)))
            prev = float(c[-1])
            name = f"{sym.title()} Limited"
            lst[sym] = {"name": name, "industry": industry}
            stocks[sym] = {
                "sym": sym,
                "ltp": round(ltp, 2),
                "pct": round((ltp / prev - 1) * 100, 2),
                "prev": round(prev, 2),
                "open": round(prev, 2),
                "high": round(max(ltp, prev) * 1.004, 2),
                "low": round(min(ltp, prev) * 0.996, 2),
                "volume": int(float(v[-60:].mean()) * rng.uniform(0.5, 2.5)),
                "value_cr": round(float(rng.uniform(5, 500)), 2),
                "ffmc": float(rng.integers(5000, 90000)) * 1e7,
                "high52": round(float(h.max()), 2),
                "low52": round(float(c.min()), 2),
                "name": name,
            }
            cls[sym] = {
                "macro": "Macro",
                "sector": sector,
                "industry": industry,
                "basic": basics[i % len(basics)],
                "index": "NIFTY 500",
                "pe": round(float(rng.uniform(8, 45)), 2),
                "sector_pe": 20.5,
                "mcap": float(rng.integers(8000, 400000)) * 1e7,
                "ffmc": stocks[sym]["ffmc"],
                "delivery": round(float(rng.uniform(20, 70)), 2),
                "volume": stocks[sym]["volume"],
                "high52": round(float(h.max()), 2),
                "low52": round(float(c.min()), 2),
                "name": name,
                "ltp": stocks[sym]["ltp"],
                "pct": stocks[sym]["pct"],
                "fetched": 1.7e9,
            }
    for sym in ("OTH1", "OTH2"):  # in the Nifty 500 list but never classified
        lst[sym] = {"name": f"{sym} Limited", "industry": "Other"}
    idx_hist = {}
    for k, name in enumerate(HIST_INDEX):
        c = 10000 * (1 + k * 0.3) * np.exp(np.cumsum(rng.normal(0.0003, 0.008, 300)))
        idx_hist[name] = {"close": [round(float(x), 4) for x in c]}
    indices = {}
    for k, (_sec, name) in enumerate({**SECTOR_INDEX, "Nifty": "NIFTY 50"}.items()):
        last = 10000.0 + 500 * k
        indices[name] = {
            "name": name,
            "last": last,
            "pct": round(float(rng.normal(0.3, 1.0)), 2),
            "chg": 10.0,
            "prev": last - 10,
            "w1": round(float(rng.normal(0.5, 2)), 2) if k % 3 else None,
            "m1": round(float(rng.normal(1, 4)), 2) if k % 2 else None,
            "y1": 12.0,
            "pe": 22.5 + k,
            "adv": 30,
            "dec": 20,
            "high52": last * 1.1,
            "low52": last * 0.8,
            "open": last - 5,
        }
    members = {}
    for index, prefix, extra in (
        ("NIFTY ENERGY", "ENG", ["ITS01"]),
        ("NIFTY IT", "ITS", []),
        ("NIFTY PHARMA", "PHM", []),
        ("NIFTY BANK", "BNK", ["NBF01"]),
    ):
        syms = [s for s in stocks if s.startswith(prefix)] + extra
        members[index] = {s: stocks[s] for s in syms}
    snaps = []
    for k in range(14):
        sec = {n: round(float(rng.normal(0, 0.6)), 3) for n in SECTOR_INDEX if rng.random() > 0.15}
        snaps.append({"m": 5 * k, "nifty": round(float(rng.normal(0, 0.3)), 3), "sec": sec})

    def items(n: int, seed: int) -> list[dict]:
        r = np.random.default_rng(seed)
        tags = ["RESULTS", "POLICY", "NEWS"]
        out = []
        for j in range(n):
            d = days[-1 - int(r.integers(1, 40))]
            out.append(
                {
                    "time": f"{d.isoformat()}T0{int(r.integers(1, 9))}:30:00+00:00",
                    "title": f"Headline {seed}-{j}",
                    "source": "Wire",
                    "link": f"https://example.invalid/{seed}/{j}",
                    "tag": tags[j % 3],
                }
            )
        return sorted(out, key=lambda x: x["time"], reverse=True)

    news = {
        n: items(5 if n == "Energy" else 4 if n == "IT" else 0, 100 + k) for k, n in enumerate(SECTOR_INDEX)
    }
    quotes = {
        "ENG03": {
            "macro": "Macro",
            "sector": "Oil Gas & Consumable Fuels",
            "industry": "Oil Gas & Consumable Fuels",
            "basic": "Gas Distribution",
            "pe": 12.5,
            "delivery": 44.4,
            "volume": 1234567,
            "high52": 2100.0,
            "low52": 900.0,
            "mcap": 2.5e12,
            "name": "Eng03 Limited",
            "sector_pe": 18.0,
        },
        "BNK02": {},
        "ZZZ": {},
    }
    out = {
        "dates": dates,
        "hist": hist,
        "idx_hist": idx_hist,
        "cls": cls,
        "list": lst,
        "indices": indices,
        "stocks": stocks,
        "members": members,
        "snaps": snaps,
        "news": news,
        "mk": items(6, 7),
        "stock_news": items(3, 8),
        "quotes": quotes,
    }
    (out_dir / "inputs_sector.json").write_text(json.dumps(out), encoding="utf-8")
    print("inputs_sector", len(hist))


def make_research(out_dir: Path) -> None:
    """Raw NSE payloads, quotes, macro readings and wiki summaries for the research-note golden (seed 31)."""
    rng = np.random.default_rng(31)
    ends = [
        "30 Jun 2024",
        "30 Sep 2024",
        "31 Dec 2024",
        "31 Mar 2025",
        "30 Jun 2025",
        "30 Sep 2025",
        "31 Dec 2025",
        "31 Mar 2026",
        "30 Jun 2026",
        "30 Sep 2026",
    ]
    filings, cmp_ = {}, {}
    plan = {"ENG03": (5.0e6, 0.08, 1.04), "BNK02": (2.0e6, 0.20, 1.03), "ENG14": (4.0e5, 0.01, 1.00)}
    for sym, (base, margin, growth) in plan.items():
        rows = []
        for k, q in enumerate(ends):
            inc = base * growth**k * (1 + rng.normal(0, 0.03))
            pat = inc * margin * (1 + rng.normal(0, 0.1))
            if sym == "ENG14" and k in (2, 5, 9):
                pat = -abs(pat) * 3
            std = {
                "gfrsymbol": sym,
                "gfrQuaterEnded": q,
                "gfrConsolidated": "Standalone",
                "gfrTotalIncome": f"{inc:,.0f}",
                "gfrProBefTax": f"{pat * 1.3:.0f}",
                "gfrNetProLoss": f"{pat:.0f}",
                "gfrErnPerShare": "1.5",
            }
            rows.append(std)
            if k != 3:  # one quarter has only the standalone filing
                con = dict(
                    std,
                    gfrConsolidated="Consolidated",
                    gfrTotalIncome=f"{inc * 1.2:.0f}",
                    gfrNetProLoss=f"{pat * 1.2:.0f}",
                )
                if k == 5:
                    con["gfrNetProLoss"] = (
                        None  # consolidated without a profit figure: the standalone one stays
                    )
                rows.insert(0, con) if k % 2 else rows.append(con)
        rows.append({"gfrsymbol": sym, "gfrQuaterEnded": None})
        filings[sym] = rows
        cmp_[sym] = (
            None
            if sym == "ENG14"
            else {
                "resCmpData": [
                    {
                        "re_debt_eqt_rat": "0.25" if sym == "ENG03" else "9.0",
                        "re_int_ser_cov": "8.1",
                        "re_to_dt": "30-SEP-2026",
                        "re_int_new": "123400",
                        "re_depr_und_exp": "445900",
                    }
                ]
            }
        )
    base_q = {
        "index": "NIFTY 500",
        "ffmc": 1.0e11,
        "delivery": 44.4,
        "volume": 1234567,
        "ltp": 1500.0,
        "pct": 0.5,
    }
    quotes = {
        "ENG03": {
            **base_q,
            "macro": "Energy",
            "sector": "Oil Gas & Consumable Fuels",
            "industry": "Oil Gas & Consumable Fuels",
            "basic": "Gas Distribution",
            "pe": 12.5,
            "sector_pe": 18.0,
            "mcap": 2.5e12,
            "high52": 2100.0,
            "low52": 900.0,
            "name": "Eng03 Limited",
        },
        "BNK02": {
            **base_q,
            "macro": "Financial Services",
            "sector": "Financial Services",
            "industry": "Banks",
            "basic": "Private Sector Bank",
            "pe": None,
            "sector_pe": 14.0,
            "mcap": 8.0e11,
            "high52": 1800.0,
            "low52": 1100.0,
            "name": "Bnk02 Limited",
        },
        "ENG14": {
            **base_q,
            "macro": "Energy",
            "sector": "Oil Gas & Consumable Fuels",
            "industry": "Oil Gas & Consumable Fuels",
            "basic": "Refineries & Marketing",
            "pe": None,
            "sector_pe": 18.0,
            "mcap": 2.0e10,
            "high52": 300.0,
            "low52": 120.0,
            "name": "Eng14 Limited",
        },
    }
    macro = {
        "oil": {"now": 82.5, "chg3m": 6.2},
        "inr": {"now": 88.1, "chg3m": -2.0},
        "metals": {"now": 4.4, "chg3m": 12.0},
        "us": {"now": 6100.0, "chg3m": 1.0},
    }
    wiki = {
        "Eng03 Limited": {
            "text": "Eng03 is an energy company. It sells gas.",
            "url": "https://example.invalid/e3",
            "title": "Eng03",
        }
    }
    out = {"filings": filings, "cmp": cmp_, "quotes": quotes, "macro": macro, "wiki": wiki}
    (out_dir / "inputs_research.json").write_text(json.dumps(out), encoding="utf-8")
    print("inputs_research", len(filings))


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    s5 = sessions(date(2025, 1, 1), 40)
    ts5 = [epoch(d, OPEN_MIN + 5 * k) for d in s5 for k in range(75)]
    stock5 = make_bars(ts5, 0.0015, 1000.0, 11, True)
    index5 = make_bars(ts5, 0.0010, 22000.0, 13, False)
    s1 = sessions(date(2023, 1, 2), 600)
    stock1 = make_bars([epoch(d, OPEN_MIN) for d in s1], 0.018, 500.0, 12, True)
    files = {
        "inputs_5m_stock": stock5,
        "inputs_5m_index": index5,
        "inputs_1d_stock": stock1,
        "inputs_15m_stock": aggregate(stock5, 3),
        "inputs_1d_from5m": daily_from(stock5),
    }
    for name, b in files.items():
        (out_dir / f"{name}.json").write_text(json.dumps(b), encoding="utf-8")
        print(name, len(b["ts"]))
    make_tape(out_dir)
    make_sector(out_dir)
    make_research(out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
