# Probe: q0_3_angel_readonly

- **Question:** Can we log in to Angel One and read holdings/positions/tradebook from this ISP?
- **Verdict:** **PASS**
- **Run at:** 2026-09-02T19:42:53.640006+05:30
- **Machine:** Windows-11-10.0.26200-SP0

## Findings

```json
{
  "credentials_present": true,
  "login_latency_ms": 831.5,
  "login_ok": true,
  "holdings_count": 0,
  "holdings_status": true,
  "positions_count": 2,
  "positions_status": true,
  "tradebook_count": 41,
  "tradebook_status": true,
  "rms_count": 1,
  "rms_status": true,
  "elapsed_s": 4.05
}
```

## Log

- generated TOTP, logging in fresh
- login OK in 831.5ms
- holdings   status=True rows=0
- positions  status=True rows=2
- tradebook  status=True rows=41
- rms        status=True rows=1
