# Probe: q0_6_feed_reachability

- **Question:** Which market-data and news feeds are reachable from this ISP?
- **Verdict:** **PASS**
- **Run at:** 2026-09-02T19:42:22.200917+05:30
- **Machine:** Windows-11-10.0.26200-SP0

## Findings

```json
{
  "bhavcopy_probe_date": "2026-08-28",
  "feeds": {
    "nse_bhavcopy": {
      "status": 200,
      "bytes": 202201,
      "ok": true,
      "url": "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20260828_F_0000.csv.zip"
    },
    "yahoo_nse_equity": {
      "status": 200,
      "bytes": 1563,
      "ok": true,
      "url": "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS?range=5d&interval=1d"
    },
    "yahoo_nifty": {
      "status": 200,
      "bytes": 1512,
      "ok": true,
      "url": "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=5d&interval=1d"
    },
    "yahoo_indiavix": {
      "status": 200,
      "bytes": 1545,
      "ok": true,
      "url": "https://query1.finance.yahoo.com/v8/finance/chart/%5EINDIAVIX?range=5d&interval=1d"
    },
    "finviz_screener": {
      "status": 200,
      "bytes": 219776,
      "ok": true,
      "url": "https://finviz.com/screener.ashx?v=111"
    },
    "moneycontrol_rss": {
      "status": 200,
      "bytes": 15347,
      "ok": true,
      "url": "https://www.moneycontrol.com/rss/latestnews.xml"
    },
    "et_markets_rss": {
      "status": 200,
      "bytes": 51732,
      "ok": true,
      "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    },
    "angel_smartapi": {
      "status": 200,
      "bytes": 247,
      "ok": true,
      "url": "https://apiconnect.angelone.in"
    },
    "ollama_local": {
      "status": 200,
      "bytes": 422,
      "ok": true,
      "url": "http://127.0.0.1:11434/api/tags"
    }
  },
  "reachable": [
    "angel_smartapi",
    "et_markets_rss",
    "finviz_screener",
    "moneycontrol_rss",
    "nse_bhavcopy",
    "ollama_local",
    "yahoo_indiavix",
    "yahoo_nifty",
    "yahoo_nse_equity"
  ],
  "critical_missing": [],
  "elapsed_s": 5.98
}
```

## Log

- nse_bhavcopy         HTTP 200    202,201B
- yahoo_nse_equity     HTTP 200      1,563B
- yahoo_nifty          HTTP 200      1,512B
- yahoo_indiavix       HTTP 200      1,545B
- finviz_screener      HTTP 200    219,776B
- moneycontrol_rss     HTTP 200     15,347B
- et_markets_rss       HTTP 200     51,732B
- angel_smartapi       HTTP 200        247B
- ollama_local         HTTP 200        422B
