"""Q0.1a / Q0.6 -- network reality: public IP, and which feeds this ISP can reach.

Why the public IP matters more than anything else in this build: since
2026-04-01 Angel One accepts API ORDER EXECUTION only from a registered primary
static IP, for self-coded algos whose logic runs on the client machine -- which
is exactly what this is. A dynamic home IP means live auto-execution is blocked
no matter how good the strategy is. Paper trading and all research are
unaffected, so this gates Q7's live path only.
"""

from __future__ import annotations

import ipaddress
import time
from datetime import date, timedelta

import requests

from stocksense.probes.base import ProbeResult

# Deliberately a mix of v4-only and dual-stack endpoints. A dual-stack service
# answers over whichever family the OS prefers, so the SAME machine legitimately
# reports a v4 address to one and a v6 address to another. Comparing across
# services therefore proves nothing; comparing each service to ITSELF over time
# is the only sound test, and the families must be separated before judging.
_IP_SERVICES = {
    "ipify_v4": "https://api.ipify.org",       # v4-only endpoint
    "ifconfig_me": "https://ifconfig.me/ip",   # dual-stack
    "icanhazip": "https://icanhazip.com",      # dual-stack
}

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _classify(value: str) -> str | None:
    try:
        return f"v{ipaddress.ip_address(value.strip()).version}"
    except ValueError:
        return None


def probe_public_ip(result: ProbeResult) -> None:
    """Resolve the public IP per service, twice, 8s apart, separating v4 from v6.

    Two samples cannot PROVE an IP is static -- only the ISP can, and only a
    router reboot really tests a DHCP lease. But a change between samples proves
    it is NOT, which is the cheap decisive case.
    """
    rounds: list[dict[str, str]] = []
    for round_no in (1, 2):
        seen: dict[str, str] = {}
        for label, url in _IP_SERVICES.items():
            try:
                r = requests.get(url, timeout=10, headers=_UA)
                seen[label] = r.text.strip() if r.ok else f"HTTP {r.status_code}"
            except Exception as exc:
                seen[label] = f"error: {type(exc).__name__}"
        rounds.append(seen)
        result.note(f"round {round_no}: {seen}")
        if round_no == 1:
            time.sleep(8)

    result.findings["rounds"] = rounds

    # Per-service stability -- the only comparison that means anything.
    unstable = [k for k in _IP_SERVICES if rounds[0].get(k) != rounds[1].get(k)]
    result.findings["services_that_changed"] = unstable

    v4 = sorted({v for rd in rounds for v in rd.values() if _classify(v) == "v4"})
    v6 = sorted({v for rd in rounds for v in rd.values() if _classify(v) == "v6"})
    result.findings["ipv4_seen"] = v4
    result.findings["ipv6_seen"] = v6

    if unstable:
        result.verdict = "BLOCKED"
        result.note(f"these services reported a DIFFERENT address 8s apart: {unstable} -- dynamic IP")
        return

    if not v4:
        result.verdict = "BLOCKED"
        result.note("no IPv4 address resolved. Angel One's static-IP registration is IPv4.")
        return

    result.findings["ipv4"] = v4[0]
    result.verdict = "PASS" if len(v4) == 1 else "BLOCKED"
    result.note(f"IPv4 stable across both samples: {v4[0]}")
    if v6:
        result.note(f"also reachable over IPv6 ({v6[0]}) -- dual-stack ISP, expected, not a problem")
    result.note(
        "STABILITY != STATIC. This only shows the address did not change in 8 seconds. "
        "Before relying on it: reboot the router and re-run, and confirm with the ISP "
        "that the lease is static. Angel One rejects API orders from any other IP."
    )


# Real, specific URLs. A bare domain legitimately 404s or 403s and proves nothing
# about whether the resource we actually fetch is reachable.
def _recent_weekday(days_back: int = 3) -> date:
    d = date.today() - timedelta(days=days_back)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def probe_feed_reachability(result: ProbeResult) -> None:
    """GET a REAL resource from every source this build depends on."""
    d = _recent_weekday()
    ymd = d.strftime("%Y%m%d")

    feeds = {
        # The actual UDiFF bhavcopy file, not the domain root.
        "nse_bhavcopy": (
            f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
        ),
        "yahoo_nse_equity": (
            "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS?range=5d&interval=1d"
        ),
        "yahoo_nifty": "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=5d&interval=1d",
        "yahoo_indiavix": "https://query1.finance.yahoo.com/v8/finance/chart/%5EINDIAVIX?range=5d&interval=1d",
        "finviz_screener": "https://finviz.com/screener.ashx?v=111",
        "moneycontrol_rss": "https://www.moneycontrol.com/rss/latestnews.xml",
        "et_markets_rss": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "angel_smartapi": "https://apiconnect.angelone.in",
        "ollama_local": "http://127.0.0.1:11434/api/tags",
    }

    out: dict[str, object] = {}
    for name, url in feeds.items():
        try:
            r = requests.get(url, timeout=20, headers=_UA)
            out[name] = {"status": r.status_code, "bytes": len(r.content), "ok": r.ok, "url": url}
            result.note(f"{name:20s} HTTP {r.status_code}  {len(r.content):>9,}B")
        except Exception as exc:
            out[name] = {"status": None, "error": f"{type(exc).__name__}: {exc}", "url": url}
            result.note(f"{name:20s} ERROR {type(exc).__name__}")

    result.findings["bhavcopy_probe_date"] = d.isoformat()
    result.findings["feeds"] = out
    reachable = {k for k, v in out.items() if isinstance(v, dict) and v.get("ok")}
    result.findings["reachable"] = sorted(reachable)

    # Ollama down is fine by design -- the pipeline runs without it. finviz and
    # Angel are not critical to RESEARCH. These four are.
    critical = {"nse_bhavcopy", "yahoo_nse_equity", "yahoo_nifty", "moneycontrol_rss"}
    missing = sorted(critical - reachable)
    result.findings["critical_missing"] = missing
    result.verdict = "PASS" if not missing else "FAIL"
    if missing:
        result.note(f"CRITICAL feeds unreachable: {missing}")
