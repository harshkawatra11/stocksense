# Q0 probe results — 2026-09-02

Every probe answers one factual question about this machine or this ISP in a few
minutes. They run **before** the code that depends on them, because each previous
build of this project lost days to an assumption a ten-minute probe would have killed.

Re-run any of them with `stocksense probe <network|compute|upstox|angel|all>`.

| Probe | Question | Verdict |
|---|---|---|
| `q0_1_public_ip` | Is the public IP stable enough to register as Angel One's static IP? | **PASS (provisional)** |
| `q0_2_upstox` | Does the Upstox market-data feed authenticate today? | **BLOCKED** — token expired |
| `q0_3_angel_readonly` | Can we read the Angel One account from this ISP? | **PASS** |
| `q0_4_gpu` | How many Monte Carlo paths fit in the VRAM budget? | **PASS** |
| `q0_5_torch_cuda` | Is there a CUDA torch wheel for this Python? | **BLOCKED** — none exists |
| `q0_6_feed_reachability` | Which data feeds are reachable? | **PASS** — all of them |

## What each result actually means for the build

**Q0.1 — IPv4 `49.36.185.50`, stable across samples, dual-stack ISP.**
This is *stability*, not proof of a *static lease*. Angel One rejects API orders
from any unregistered IP since 2026-04-01, so before the live path is armed:
reboot the router, re-run this probe, and confirm the lease with the ISP. Until
then the execution layer is paper-only — which is the plan for week 1 regardless.

**Q0.2 — Upstox returns HTTP 401 in 410 ms.**
Expected, not broken: Upstox access tokens expire daily around 03:30 IST and the
one in `.env` is from a previous session. The integration itself is untested
until a fresh token exists. The OAuth flow has to be re-run before the intraday
spine can backfill.

**Q0.3 — Angel One works from this ISP.** Fresh TOTP login in 831 ms; holdings 0,
positions 2, tradebook 41 rows, RMS readable. Notable because a previous build
recorded SmartAPI login timing out on this same home network — that blocker is
gone. Read-only by construction; this module contains no order-placement call.

**Q0.4 — GPU is genuinely fast enough that Monte Carlo is not a bottleneck.**
RTX 3050, 3,965 MB free. At the 2,500 MB budget: **2.6 M float32 paths × 250 steps**.
Measured: 1 M paths × 250 steps in **0.183 s** on GPU vs ~3.5 s on CPU — about
19× — after a **1.9 s one-time CUDA context warm-up** that the live engine must
pay at startup, never on the decision path. The CPU fallback (100 k paths in
0.35–0.52 s) is usable, so a busy GPU degrades the system rather than stopping it.

**Q0.5 — No CUDA PyTorch for Python 3.14.** `pip install torch --index-url
.../cu124` returns *"no matching distribution"*. Any sequence/path-forecast model
is therefore CPU-only, deferred, or run under the Python 3.12 interpreter already
on this machine. Not a blocker: LightGBM is the primary model and CuPy covers
simulation.

**Q0.6 — Every feed reachable**, including the real 202 KB UDiFF bhavcopy zip,
Yahoo for NSE equities / `^NSEI` / `^INDIAVIX`, finviz, Moneycontrol and ET RSS,
and a local Ollama.

## One environment blocker found and solved

**Windows Smart App Control is ON** (`VerifiedAndReputablePolicyState = 1`) and
blocks unsigned binaries lacking download reputation. It blocked **scipy 1.18.x**:

```
ImportError: DLL load failed while importing cython_lapack:
An Application Control policy has blocked this file.
```

Measured across both Python 3.12 and 3.14: **1.16.0, 1.16.3 and 1.17.1 load fine;
1.18.1 does not.** `pyproject.toml` therefore pins `scipy>=1.11,<1.18`, and that
ceiling is load-bearing — raise it only after verifying the new version actually
imports on this machine. Turning Smart App Control off was rejected as a fix: on
Windows it is a one-way switch that can only be re-enabled by reinstalling.

`smartapi-python` also imports `logzero` without declaring it; added explicitly.
