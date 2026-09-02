# Probe: q0_1_public_ip

- **Question:** Is this machine's public IP stable enough to register as Angel One's static IP?
- **Verdict:** **PASS**
- **Run at:** 2026-09-02T19:42:16.051885+05:30
- **Machine:** Windows-11-10.0.26200-SP0

## Findings

```json
{
  "rounds": [
    {
      "ipify_v4": "49.36.185.50",
      "ifconfig_me": "2405:201:401f:4096:4559:cb31:3e63:deb2",
      "icanhazip": "2405:201:401f:4096:4559:cb31:3e63:deb2"
    },
    {
      "ipify_v4": "49.36.185.50",
      "ifconfig_me": "2405:201:401f:4096:4559:cb31:3e63:deb2",
      "icanhazip": "2405:201:401f:4096:4559:cb31:3e63:deb2"
    }
  ],
  "services_that_changed": [],
  "ipv4_seen": [
    "49.36.185.50"
  ],
  "ipv6_seen": [
    "2405:201:401f:4096:4559:cb31:3e63:deb2"
  ],
  "ipv4": "49.36.185.50",
  "elapsed_s": 10.98
}
```

## Log

- round 1: {'ipify_v4': '49.36.185.50', 'ifconfig_me': '2405:201:401f:4096:4559:cb31:3e63:deb2', 'icanhazip': '2405:201:401f:4096:4559:cb31:3e63:deb2'}
- round 2: {'ipify_v4': '49.36.185.50', 'ifconfig_me': '2405:201:401f:4096:4559:cb31:3e63:deb2', 'icanhazip': '2405:201:401f:4096:4559:cb31:3e63:deb2'}
- IPv4 stable across both samples: 49.36.185.50
- also reachable over IPv6 (2405:201:401f:4096:4559:cb31:3e63:deb2) -- dual-stack ISP, expected, not a problem
- STABILITY != STATIC. This only shows the address did not change in 8 seconds. Before relying on it: reboot the router and re-run, and confirm with the ISP that the lease is static. Angel One rejects API orders from any other IP.
