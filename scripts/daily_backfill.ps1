<#
Phase H (found live, 2026-08-24): reconcile/retrain-weekly were
scheduled (Phase H3) but nothing was ever scheduled to keep bhavcopy_eq
itself fresh. Without this, "the most recent available cross-section"
reconcile scores stays pinned to whatever date backfill-nse-archive was
last run manually for -- confirmed live: a reconcile run on 08-24
still scored 08-20's data, five calendar days stale, because nothing
had pulled 08-21 onward.

backfill-nse-archive needs explicit --start/--end dates -- a static
Windows Scheduled Task command line can't compute "today" at fire time,
only a script can. This wrapper does that: pulls a rolling 7-calendar-
day window ending today, every time it runs. Deliberately a WIDE
window, not just "yesterday to today": backfill-nse-archive is content-
hash-cached and re-fetching an already-known day is cheap
(research/bhavcopy_rerun_sweep.py's own docstring on this command:
"re-running the same range replays quickly through the cached prefix"),
so a 7-day window makes this self-healing -- if the laptop is off for
a few days, or the source is briefly unreachable, the next run catches
up automatically instead of leaving a permanent gap that only a human
re-running the command by hand would notice.

Usage: powershell -ExecutionPolicy Bypass -File scripts\daily_backfill.ps1
#>

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = (Get-Command python -ErrorAction Stop).Source

$Start = (Get-Date).AddDays(-7).ToString("yyyy-MM-dd")
$End = (Get-Date).ToString("yyyy-MM-dd")

Push-Location $RepoRoot
try {
    # Phase J0.1: backfill-nse-archive itself now retries a DuckDB lock
    # collision internally (connect_with_retry, 5 attempts / 60s apart)
    # -- this outer loop is defense in depth for a collision so
    # persistent it outlasts even that (e.g. a multi-minute UI-triggered
    # job holding the write lock), so the scheduled task exits non-zero
    # only after genuinely giving up, not on the first bad second.
    $maxAttempts = 3
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        & $PythonPath -m stocksense.cli.main backfill-nse-archive --start $Start --end $End --kind cm
        if ($LASTEXITCODE -eq 0) { break }
        Write-Host "backfill-nse-archive exited $LASTEXITCODE (attempt $attempt/$maxAttempts)"
        if ($attempt -lt $maxAttempts) { Start-Sleep -Seconds 120 }
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
