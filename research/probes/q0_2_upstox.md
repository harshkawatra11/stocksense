# Probe: q0_2_upstox

- **Question:** Does the Upstox market-data feed work from here today?
- **Verdict:** **BLOCKED**
- **Run at:** 2026-09-02T19:42:34.201617+05:30
- **Machine:** Windows-11-10.0.26200-SP0

## Findings

```json
{
  "token_present": true,
  "ltp_status": 401,
  "ltp_latency_ms": 410.2,
  "body": "{\"status\":\"error\",\"errors\":[{\"errorCode\":\"UDAPI100050\",\"message\":\"Invalid token used to access API\",\"propertyPath\":null,\"invalidValue\":null,\"error_code\":\"UDAPI100050\",\"property_path\":null,\"invalid_value\":null}]}",
  "elapsed_s": 0.41
}
```

## Log

- LTP  HTTP 401 in 410.2ms
- token expired -- Upstox tokens die daily around 03:30 IST; re-run the OAuth flow
