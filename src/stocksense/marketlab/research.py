"""Ported from market-lab lab/research.py (MIT, ba3f771); changes: numpy instead of pandas, Sources
(NseClient, Yahoo, Wikipedia-API) instead of raw HTTP, prices and technicals from our stored adjusted daily bars, the Clock
for cache ages, a filing that cannot be fetched leaves that section empty instead of failing the whole note, and
the Wikipedia page must mention the company name.

A one-page, plain-language research note for any NSE stock: what the company does, how its results are moving,
its finances, valuation against peers, the industry and the economy around it, a SWOT and an overall score out of
100. Rules-based (no opinions are invented); cached for 12 hours per stock. Percentages are Market Lab's percent
units; amounts are rupees crore.
"""

# ruff: noqa: E501 (the note's sentences are copied from Market Lab)
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.marketlab import industries as I  # noqa: E741 (Market Lab's name for the module)
from stocksense.marketlab import npx
from stocksense.marketlab.sectors import Hist
from stocksense.marketlab.sources import SourceError, to_float
from stocksense.marketlab.store import LabStore, TTLCache

log = get_logger(__name__)

MACRO_SYMS = {"oil": "BZ=F", "inr": "USDINR=X", "metals": "HG=F", "us": "^GSPC"}
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"}
NOTE_TTL_S = 12 * 3600.0
MACRO_TTL_S = 6 * 3600.0
_SOFT = (SourceError, DataNotAvailable, httpx.HTTPError, OSError, KeyError, IndexError, TypeError, ValueError)
WEIGHTS = {
    "Business": 15,
    "Growth": 20,
    "Profitability": 15,
    "Financial health": 15,
    "Valuation": 15,
    "Industry & economy": 10,
    "Momentum": 10,
}


def plain(x: Any) -> Any:
    """Recursively turn numpy scalars into JSON-native Python values."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def clip(x: float, lo: float, hi: float) -> float:
    """x limited to [lo, hi]."""
    return max(lo, min(hi, x))


def lin(v: float | None, a: float, b: float) -> float | None:
    """Map v from [a, b] to [0, 10] (clipped, one decimal); None stays None."""
    if v is None:
        return None
    return round(clip((v - a) / (b - a) * 10, 0, 10), 1)


def grade_of(total: float | None) -> str | None:
    """Strong (75+), Good (62+), Average (48+), Weak (35+), else Poor."""
    if total is None:
        return None
    return (
        "Strong"
        if total >= 75
        else "Good"
        if total >= 62
        else "Average"
        if total >= 48
        else "Weak"
        if total >= 35
        else "Poor"
    )


def valuation_score(
    pe: float | None, spe: float | None, g_pat: float | None, ttm_pat: float | None
) -> float | None:
    """0-10 valuation: price/earnings against the sector's, a bonus for fast profit growth; 2 for a loss maker."""
    if pe and pe > 0 and spe:
        r = pe / spe
        v = (
            9
            if r <= 0.7
            else 7.5
            if r <= 0.9
            else 6
            if r <= 1.1
            else 4.5
            if r <= 1.4
            else 3
            if r <= 2
            else 1.5
        )
        if g_pat is not None and g_pat > 25 and r < 1.6:
            v = min(10, v + 1.5)
        return v
    return 2 if ttm_pat is not None and ttm_pat <= 0 else None


def _quarter_date(q: str) -> datetime | None:
    for fmt in ("%d %b %Y", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(q.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_quarters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last 8 quarters from NSE integrated filings in rupees crore (NSE reports lakh); consolidated when filed."""
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        q = r.get("gfrQuaterEnded")
        if not q:
            continue
        kind = r.get("gfrConsolidated") or ""
        cur = by.get(q)
        if cur is None or (
            kind == "Consolidated"
            and cur["kind"] != "Consolidated"
            and to_float(r.get("gfrNetProLoss")) is not None
        ):
            by[q] = {
                "q": q,
                "kind": kind,
                "income": to_float(r.get("gfrTotalIncome")),
                "pbt": to_float(r.get("gfrProBefTax")),
                "profit": to_float(r.get("gfrNetProLoss")),
                "eps": to_float(r.get("gfrErnPerShare")),
            }
    out = []
    for q, v in by.items():
        d = _quarter_date(q)
        if d is None:
            continue
        out.append(
            {
                **v,
                "date": d.date().isoformat(),
                "label": d.strftime("%b %y"),
                **{k: (v[k] / 100 if v[k] is not None else None) for k in ("income", "pbt", "profit")},
            }
        )
    return sorted(out, key=lambda x: x["date"])[-8:]


def parse_balance(d: dict[str, Any]) -> dict[str, Any]:
    """Debt/equity and interest cover from NSE's results-comparison filing."""
    r = (d.get("resCmpData") or [{}])[0]
    return {
        "de": to_float(r.get("re_debt_eqt_rat")),
        "icr": to_float(r.get("re_int_ser_cov")),
        "as_of": r.get("re_to_dt"),
        "interest": (to_float(r.get("re_int_new")) or 0) / 100,
        "dep": (to_float(r.get("re_depr_und_exp")) or 0) / 100,
    }


def ret(h: Hist | None, days: int) -> float | None:
    """Percent change over `days` sessions of the adjusted close, one decimal; None when the history is short."""
    if h is None or len(h) <= days:
        return None
    c = h.close
    return round(float((c[-1] / c[-1 - days] - 1) * 100), 1)


def tech_of(h: Hist | None, nifty: Hist | None) -> dict[str, Any]:
    """Trend, relative strength, distance from the 252-day high and volatility at the last bar (needs 261+ sessions)."""
    if h is None or nifty is None:
        return {}
    keep = h.volume > 0
    d, c, hi = h.dates[keep], h.close[keep], h.high[keep]
    lo = c if h.low is None else np.where(np.isnan(h.low[keep]), c, h.low[keep])
    if len(c) <= 260:
        return {}
    pc = npx.shift(c, 1, c[0])
    tr = np.maximum(hi - lo, np.maximum(abs(hi - pc), abs(lo - pc)))
    atr = npx.rma(tr, 14)
    sma50, sma200, ema20 = npx.roll(c, 50, "mean"), npx.roll(c, 200, "mean"), npx.ema(c, 20)
    roc63 = npx.pct_change(c, 63) * 100
    nroc = npx.pct_change(nifty.close, 63) * 100
    pos = int(np.searchsorted(nifty.dates, d[-1]))
    mkt = nroc[pos] if pos < len(nifty.dates) and nifty.dates[pos] == d[-1] else np.nan
    rs63 = roc63[-1] - mkt
    hh252 = npx.shift(npx.roll(hi, 252, "max"))[-1]
    i = len(c) - 1
    return {
        "uptrend": bool(c[i] > sma200[i] and sma50[i] > sma200[i] and ema20[i] > sma50[i]),
        "rs63": round(float(rs63), 1) if np.isfinite(rs63) else None,
        "from_high": round(float((c[i] / hh252 - 1) * 100), 1) if np.isfinite(hh252) else None,
        "atr_pct": round(float(atr[i] / c[i] * 100), 2),
        "ret1y": ret(h, 250),
        "ret3m": ret(h, 63),
    }


class ResearchEngine:
    """Builds and caches the research note of one stock."""

    def __init__(self, settings: Settings, clock: Clock, sources: Any, store: LabStore) -> None:
        self.settings, self.clock, self.src, self.store = settings, clock, sources, store
        self._cache = TTLCache(clock)

    # ---- inputs -------------------------------------------------------------------------------------
    def macro(self) -> dict[str, Any]:
        """3-month change of crude, the dollar-rupee, copper and the S&P 500 (Yahoo; cached 6 hours)."""

        def fetch() -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, y in MACRO_SYMS.items():
                try:
                    d = self.src.yahoo_chart(y, "1d", "6mo")
                    c = [x for x in d["indicators"]["quote"][0]["close"] if x]
                    out[k] = {
                        "now": round(c[-1], 2),
                        "chg3m": round((c[-1] / c[-64] - 1) * 100, 1) if len(c) > 64 else None,
                    }
                except _SOFT:
                    log.warning("research_macro_failed", series=k)
            return out

        return self._cache.get("macro", MACRO_TTL_S, fetch)

    def _quarters(self, sym: str) -> list[dict[str, Any]]:
        try:
            return parse_quarters(self.src.nse_integrated_filing(sym) or [])
        except _SOFT:
            log.warning("research_filing_failed", sym=sym)
            return []

    def _balance(self, sym: str) -> dict[str, Any]:
        try:
            return parse_balance(self.src.nse_results_comparison(sym))
        except _SOFT:
            return {}

    def _wiki(self, name: str | None) -> dict[str, Any] | None:
        try:
            return self.src.wiki_summary(name) if name else None
        except _SOFT:
            log.warning("research_wiki_failed")
            return None

    @staticmethod
    def _peers(sym: str, cls: dict[str, Any]) -> list[dict[str, Any]]:
        me = cls.get(sym) or {}
        same = [(k, v) for k, v in cls.items() if v.get("basic") == me.get("basic") and k != sym]
        if len(same) < 3:
            same = [(k, v) for k, v in cls.items() if v.get("industry") == me.get("industry") and k != sym]
        same.sort(key=lambda kv: -(kv[1].get("mcap") or 0))
        return [
            {
                "sym": k,
                "name": (v.get("name") or k)[:28],
                "mcap_cr": round((v.get("mcap") or 0) / 1e7),
                "pe": v.get("pe"),
            }
            for k, v in same[:6]
        ]

    def _sector_strength(self, sym: str, cls: dict[str, Any]) -> dict[str, Any]:
        hist = self.store.univ["hist"]
        me = cls.get(sym) or {}
        mates = [k for k, v in cls.items() if v.get("sector") == me.get("sector")]
        r = [x for x in (ret(hist.get(k), 63) for k in mates[:40]) if x is not None]
        n = ret(hist.get("IDX:NIFTY 50"), 63)
        return {
            "sector": me.get("sector"),
            "sector_3m": round(float(np.median(r)), 1) if r else None,
            "nifty_3m": n,
            "rel": round(float(np.median(r)) - n, 1) if r and n is not None else None,
            "n": len(r),
        }

    # ---- the note -----------------------------------------------------------------------------------
    def build(self, sym: str) -> dict[str, Any]:
        """The full note for one stock (raises SourceError when NSE's quote page cannot be read)."""
        q = self.src.nse_symbol_data(sym)
        cls = dict(self.store.univ["cls"])
        cls.setdefault(sym, q)
        macro_name, basic = q.get("macro") or "", q.get("basic") or ""
        what, drivers = I.lookup(basic, macro_name)
        fin, bal, wk = self._quarters(sym), self._balance(sym), self._wiki(q.get("name"))
        mac = self.macro()
        sec = self._sector_strength(sym, cls)
        pr = self._peers(sym, cls)
        mcap_cr = (q.get("mcap") or 0) / 1e7
        fs = macro_name == "Financial Services"

        # results
        inc = [x["income"] for x in fin if x["income"] is not None]
        pat = [x["profit"] for x in fin if x["profit"] is not None]
        ttm_inc = sum(inc[-4:]) if len(inc) >= 4 else None
        ttm_pat = sum(pat[-4:]) if len(pat) >= 4 else None
        # growth: each recent quarter against the same quarter a year earlier, over whatever pairs NSE has
        byd = {x["date"][:7]: x for x in fin}
        pairs = []
        for x in fin[-4:]:
            y = byd.get(f"{int(x['date'][:4]) - 1}{x['date'][4:7]}")
            if y:
                pairs.append((x, y))

        def growth(key: str) -> float | None:
            a = [(x[key], y[key]) for x, y in pairs if x[key] is not None and y[key] is not None]
            if not a:
                return None
            now, then = sum(u for u, _ in a), sum(v for _, v in a)
            return (now / then - 1) * 100 if then > 0 else None

        g_inc, g_pat = growth("income"), growth("profit")
        n_pairs = len(pairs)
        q_yoy = None
        if len(fin) >= 5 and fin[-1]["profit"] is not None and fin[-5]["profit"] and fin[-5]["profit"] > 0:
            q_yoy = (fin[-1]["profit"] / fin[-5]["profit"] - 1) * 100
        margin = ttm_pat / ttm_inc * 100 if ttm_inc and ttm_pat is not None else None
        norm = I.MARGIN_NORM.get(macro_name, 8)
        profit_q = sum(1 for p in pat if p > 0)
        pe, spe = q.get("pe"), q.get("sector_pe")
        if (not pe or pe <= 0) and ttm_pat and ttm_pat > 0 and mcap_cr:
            pe = mcap_cr / ttm_pat

        # price and trend (adjusted daily bars)
        hist = self.store.univ["hist"]
        tech = tech_of(hist.get(sym), hist.get("IDX:NIFTY 50"))

        # economy: which way each driver is pushing right now
        econ = []
        tail = 0.0
        for fac, eff, why in drivers:
            reading = mac.get(fac) if fac in mac else None
            push = None
            if reading and reading.get("chg3m") is not None and eff != 0:
                ch = reading["chg3m"]
                thr = 1.5 if fac == "inr" else 3
                if abs(ch) >= thr:
                    push = eff * (1 if ch > 0 else -1)
                    tail += push
            elif reading and reading.get("chg3m") is not None and abs(reading["chg3m"]) >= 10:
                push, tail = 0, tail - 0.5  # a two-way driver moving this much is a risk either way
            econ.append(
                {
                    "factor": fac,
                    "name": I.FACTOR_NAMES.get(fac, fac),
                    "effect": eff,
                    "why": why,
                    "reading": reading,
                    "push": push,
                }
            )

        # scores (0-10 each)
        sc: dict[str, Any] = {}
        size = lin(float(np.log10(max(mcap_cr, 1))), 2.5, 5.3)  # about Rs 300 cr .. Rs 2 lakh cr
        cons = profit_q / max(len(pat), 1) * 10 if pat else None
        sc["Business"] = (
            round(np.nanmean([x for x in (size, cons) if x is not None]), 1)
            if (size is not None or cons is not None)
            else None
        )
        gs = [lin(g_inc, -10, 25), lin(g_pat, -20, 40), lin(q_yoy, -30, 50)]
        sc["Growth"] = (
            round(np.nanmean([x for x in gs if x is not None]), 1) if any(x is not None for x in gs) else None
        )
        sc["Profitability"] = lin(margin / norm if margin is not None else None, 0, 2)
        if fs:
            sc["Financial health"] = round(cons, 1) if cons is not None else None
        else:
            de = bal.get("de")
            de_s = (
                None
                if de is None
                else (10 if de < 0.3 else 8 if de < 0.6 else 6 if de < 1 else 3.5 if de < 2 else 1)
            )
            hs = [x for x in (de_s, cons) if x is not None]  # NSE's coverage field is unreliable: not used
            sc["Financial health"] = round(np.nanmean(hs), 1) if hs else None
        sc["Valuation"] = valuation_score(pe, spe, g_pat, ttm_pat)
        ind = lin(sec.get("rel"), -10, 10)
        sc["Industry & economy"] = round(clip((ind if ind is not None else 5) + tail * 1.5, 0, 10), 1)
        if tech:
            m = (
                (4 if tech.get("uptrend") else 0)
                + (lin(tech.get("from_high"), -30, 0) or 0) * 0.3
                + (3 if (tech.get("rs63") or 0) > 0 else 0)
            )
            sc["Momentum"] = round(clip(m, 0, 10), 1)
        have = {k: v for k, v in sc.items() if v is not None}
        total = (
            round(sum(have[k] * WEIGHTS[k] for k in have) / sum(WEIGHTS[k] for k in have) * 10)
            if have
            else None
        )
        grade = grade_of(total)

        # SWOT
        s_, w_, o_, t_ = [], [], [], []
        if g_pat is not None and g_pat > 20:
            s_.append(f"Profit is growing fast: up {g_pat:.0f}% on the same quarters a year earlier.")
        if g_inc is not None and g_inc > 15:
            s_.append(f"Sales are growing: up {g_inc:.0f}% on a year earlier.")
        if margin is not None and margin >= norm:
            s_.append(
                f"Keeps a healthy share of sales as profit ({margin:.1f}%, typical for its kind is ~{norm}%)."
            )
        if pat and profit_q == len(pat) and len(pat) >= 6:
            s_.append(f"Made a profit in every one of the last {len(pat)} quarters.")
        if not fs and bal.get("de") is not None and bal["de"] < 0.3:
            s_.append(
                f"Very little debt in its last detailed filing ({bal.get('as_of') or 'date n/a'}, own business only)."
            )
        if pr and mcap_cr and all(mcap_cr >= p["mcap_cr"] for p in pr):
            s_.append("The biggest company in its industry on the market.")
        if tech.get("rs63") and tech["rs63"] > 10:
            s_.append(f"The share has beaten Nifty by {tech['rs63']:.0f} points over 3 months.")
        if margin is not None and margin < norm / 2:
            w_.append(f"Thin profit margin ({margin:.1f}% of sales), so small cost changes hurt.")
        if g_pat is not None and g_pat < -10:
            w_.append(f"Profit fell {abs(g_pat):.0f}% against a year earlier.")
        if g_inc is not None and g_inc < -5:
            w_.append(f"Sales shrank {abs(g_inc):.0f}% against a year earlier.")
        if pat and profit_q < len(pat):
            w_.append(f"Made a loss in {len(pat) - profit_q} of the last {len(pat)} quarters.")
        if not fs and bal.get("de") is not None and bal["de"] > 1.5:
            w_.append(
                f"Carries a lot of debt (debt {bal['de']:.1f}× its own capital, filing of {bal.get('as_of') or 'n/a'})."
            )
        if pe and spe and pe > 1.5 * spe:
            w_.append(f"Expensive: {pe:.0f}× its yearly profit against ~{spe:.0f}× for its sector.")
        if sec.get("rel") is not None and sec["rel"] > 3:
            o_.append(
                f"Its sector ({sec['sector']}) has beaten Nifty by {sec['rel']:.0f} points over 3 months: money is flowing in."
            )
        for e in econ:
            if e["push"] == 1:
                o_.append(f"{e['name']} is moving in its favour ({e['why']}).")
            if e["push"] == -1:
                t_.append(f"{e['name']} is moving against it ({e['why']}).")
            if e["push"] == 0:
                t_.append(
                    f"{e['name']} moved {e['reading']['chg3m']:+.0f}% in 3 months; big swings hurt it ({e['why']})."
                )
        if pe and spe and pe < 0.85 * spe and (g_pat or 0) > 0:
            o_.append(
                f"Cheaper than its sector ({pe:.0f}× vs ~{spe:.0f}× profit) while profit is still growing."
            )
        if tech.get("uptrend") and tech.get("from_high") is not None and tech["from_high"] > -8:
            o_.append("Share is in an uptrend and close to its 52-week high.")
        if any(e["factor"] == "reg" for e in econ):
            t_.append(
                "Depends on government or regulator decisions ("
                + next(e["why"] for e in econ if e["factor"] == "reg")
                + ")."
            )
        if any(e["factor"] == "china" for e in econ):
            t_.append("Cheap Chinese imports can cut its selling prices.")
        if sec.get("rel") is not None and sec["rel"] < -3:
            t_.append(f"Its sector has lagged Nifty by {abs(sec['rel']):.0f} points over 3 months.")
        if tech.get("atr_pct") and tech["atr_pct"] > 3.5:
            t_.append(f"The share swings a lot (about {tech['atr_pct']:.1f}% on a normal day).")
        if tech.get("from_high") is not None and tech["from_high"] < -30:
            t_.append(f"The share is {abs(tech['from_high']):.0f}% below its 52-week high.")

        # plain-language summary
        size_word = (
            "very large"
            if mcap_cr >= 100000
            else "large"
            if mcap_cr >= 50000
            else "mid-sized"
            if mcap_cr >= 15000
            else "small"
            if mcap_cr >= 3000
            else "very small"
        )
        lines = [
            f"{q.get('name') or sym}: {what}",
            f"It is a {size_word} company: the market values the whole business at about ₹{mcap_cr:,.0f} crore.",
        ]
        if ttm_inc:
            lines.append(
                f"In the last 12 months it brought in ₹{ttm_inc:,.0f} crore and kept ₹{ttm_pat:,.0f} crore as profit"
                + (f", about ₹{margin:.1f} of profit on every ₹100 of sales." if margin is not None else ".")
            )
        if g_inc is not None or g_pat is not None:
            parts = (
                f"income is {'up' if g_inc >= 0 else 'down'} {abs(g_inc):.0f}%"
                if g_inc is not None
                else None,
                f"profit is {'up' if g_pat >= 0 else 'down'} {abs(g_pat):.0f}%"
                if g_pat is not None
                else None,
            )
            lines.append(
                (
                    "Compared with the same quarters a year earlier, "
                    if n_pairs < 4
                    else "Compared with the 12 months before, "
                )
                + ", ".join(x for x in parts if x)
                + "."
            )
        if pe and pe > 0:
            cmp_ = (
                ""
                if not spe
                else (
                    " — cheaper than"
                    if pe < 0.9 * spe
                    else " — more expensive than"
                    if pe > 1.1 * spe
                    else " — about the same as"
                )
                + f" its sector (~{spe:.0f}×)"
            )
            lines.append(f"At today's price you pay about {pe:.0f}× one year's profit{cmp_}.")
        out = {
            "sym": sym,
            "name": q.get("name"),
            "built": int(self.clock.now().timestamp()),
            "macro": macro_name,
            "sector": q.get("sector"),
            "industry": q.get("industry"),
            "basic": basic,
            "index": q.get("index"),
            "what": what,
            "summary": lines,
            "wiki": wk,
            "mcap_cr": round(mcap_cr),
            "pe": round(pe, 1) if pe else None,
            "sector_pe": spe,
            "delivery": q.get("delivery"),
            "high52": q.get("high52"),
            "low52": q.get("low52"),
            "ltp": q.get("ltp"),
            "results": fin,
            "ttm": {
                "income": ttm_inc,
                "profit": ttm_pat,
                "margin": round(margin, 1) if margin is not None else None,
                "g_inc": round(g_inc, 1) if g_inc is not None else None,
                "g_pat": round(g_pat, 1) if g_pat is not None else None,
                "q_yoy": round(q_yoy, 1) if q_yoy is not None else None,
                "norm": norm,
                "pairs": n_pairs,
                "profit_q": profit_q,
                "n_q": len(pat),
            },
            "balance": bal,
            "financial": fs,
            "peers": pr,
            "sector_strength": sec,
            "economy": econ,
            "tech": tech,
            "scores": sc,
            "weights": WEIGHTS,
            "total": total,
            "grade": grade,
            "swot": {
                "Strengths": s_ or ["Nothing stands out."],
                "Weaknesses": w_ or ["No obvious weak spot in the numbers."],
                "Opportunities": o_ or ["No clear tailwind right now."],
                "Threats": t_ or ["No clear headwind right now."],
            },
            "sources": [
                "NSE quarterly filings & quote page",
                "Wikipedia",
                "Yahoo (crude, dollar-rupee, copper, S&P 500)",
            ],
        }
        return plain(out)

    def _path(self, sym: str) -> Path:
        d = Paths(self.settings).lab / "research"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{sym}.json"

    def report(self, sym: str, fresh: bool = False) -> dict[str, Any]:
        """The note, from the 12-hour disk cache unless `fresh` or older than 12 hours."""
        p = self._path(sym)
        now = self.clock.now().timestamp()
        if not fresh and p.exists():
            cached = json.loads(p.read_text(encoding="utf-8"))
            if now - cached.get("built", 0) < NOTE_TTL_S:
                return cached
        r = self.build(sym)
        p.write_text(json.dumps(r), encoding="utf-8")
        return r


def build_research_engine(settings: Settings, clock: Clock, store: LabStore, sources: Any) -> ResearchEngine:
    """The research engine sharing the sector engine's history and classification."""
    return ResearchEngine(settings, clock, sources, store)
