<#
Phase H3 (+ a fix found running this live on 2026-08-24): registers
three Windows Scheduled Tasks so the daily brief actually populates
every morning without anyone clicking a button.

  StockSense-DailyBackfill  daily  07:30  -- pulls fresh NSE bhavcopy
                                             data (rolling 7-day window,
                                             self-healing -- see
                                             daily_backfill.ps1). MUST
                                             run before reconcile: found
                                             live that without this,
                                             reconcile silently kept
                                             scoring a stale data date
                                             every day, which then
                                             duplicated every prediction
                                             row (see the commit that
                                             made record_predictions
                                             idempotent per DATA date).
  StockSense-Reconcile      daily  08:00  -- grade matured predictions,
                                             record today's, for the
                                             LIVE model. Runs before
                                             NSE's 09:15 open.
  StockSense-PaperRun       daily  08:05  -- Phase J2: steps every
                                             ACTIVE paper account
                                             forward. Runs AFTER
                                             reconcile so a paper account
                                             always acts on a ledger that
                                             already reflects today's
                                             predictions, never a stale
                                             one from before this
                                             morning's run.
  StockSense-RetrainWeekly  weekly Sun 06:00 -- walk-forward retrain,
                                             gate, register a fresh
                                             candidate. Never touches
                                             which model is 'live' --
                                             promote-model stays a
                                             manual, human step.

Registered under the CURRENT user's own logon session (LogonType
Interactive) -- no password is stored or prompted for by this script.
That means each task only runs while this Windows account is logged
in; if you need it to run unattended even when logged out, re-register
the task yourself with your own credentials via Task Scheduler's GUI
(Properties -> General -> "Run whether user is logged on or not") --
that is a deliberate choice this script does not make for you.

This registers PERSISTENT, UNATTENDED AUTOMATION on this machine. To
inspect: Get-ScheduledTask -TaskName "StockSense-*"
To remove:  Unregister-ScheduledTask -TaskName "StockSense-DailyBackfill","StockSense-Reconcile","StockSense-PaperRun","StockSense-RetrainWeekly","StockSense-BrokerSync" -Confirm:$false

Usage: powershell -ExecutionPolicy Bypass -File scripts\register_scheduled_tasks.ps1
#>

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot "data_store\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$PythonPath = (Get-Command python -ErrorAction Stop).Source
Write-Host "Using python: $PythonPath"
Write-Host "Repo root:    $RepoRoot"

function Register-StockSenseTask {
    param(
        [string]$TaskName,
        [string]$Command,        # the full command line to run, e.g. "`"$PythonPath`" -m stocksense.cli.main ..."
        [Microsoft.Management.Infrastructure.CimInstance]$Trigger,
        [string]$LogFileStem
    )

    # Route through cmd.exe so >> redirection works -- Register-
    # ScheduledTask's action has no built-in stdout/stderr capture, and
    # a task that fails silently with no log is much harder to debug
    # than the small extra indirection of a cmd /c wrapper. One rolling
    # log per task (not per-day) -- simple, and each run's own output is
    # still timestamped by the CLI's own structlog lines.
    $cmdArgs = "/c `"$Command >> `"$LogDir\$LogFileStem.log`" 2>&1`""

    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmdArgs -WorkingDirectory $RepoRoot
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
    # DisallowStartIfOnBatteries/StopIfGoingOnBatteries default to TRUE
    # on New-ScheduledTaskSettingsSet -- found live: the first real run
    # of this task sat in "Queued" state indefinitely and never actually
    # executed, on a LAPTOP, because it wasn't plugged in. Explicitly
    # disabled -- this machine is a laptop, not a server, and the whole
    # point of this automation is that it runs unattended regardless of
    # whether it happens to be on AC power at 7:30/8:00 AM.
    #
    # -WakeToRun: found live a second time, 2026-08-25 -- neither task
    # fired at all that morning (no log entry, LastTaskResult showed
    # "has not run") even though registration itself looked healthy.
    # Root cause: Windows' default scheduler will not start a task on a
    # machine that's asleep, only queue it for next time the machine
    # happens to be awake -- StartWhenAvailable covers "missed while the
    # machine was briefly busy," not "machine was asleep at trigger
    # time." WakeToRun asks the machine to actually wake from sleep for
    # this trigger (requires wake-timer support in the hardware/BIOS;
    # does nothing for a fully powered-off machine, which is a separate,
    # unaddressed gap -- the user still needs to leave the laptop
    # sleeping, not shut down, overnight for this to help).
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun

    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $Trigger `
        -Principal $principal -Settings $settings -Force | Out-Null

    Write-Host "Registered: $TaskName"
}

# Must run BEFORE reconcile -- see daily_backfill.ps1's own docstring
# for why this was missing and what it broke.
$backfillTrigger = New-ScheduledTaskTrigger -Daily -At 7:30AM
Register-StockSenseTask -TaskName "StockSense-DailyBackfill" `
    -Command "powershell.exe -ExecutionPolicy Bypass -File `"$RepoRoot\scripts\daily_backfill.ps1`"" `
    -Trigger $backfillTrigger -LogFileStem "daily_backfill"

# --cap-band full_pit MUST match the cap band the live model was
# actually trained with (Phase H1: cross_sectional_ranker_h10_n10_...,
# trained via `train-candidate --cap-band full_pit`) -- reconcile.py's
# own --cap-band docstring warns explicitly that a mismatch here
# silently scores/grades the ledger against a different universe than
# the model was trained and gated on. If a different cap band is ever
# promoted to live instead, update BOTH lines below to match it.
$reconcileTrigger = New-ScheduledTaskTrigger -Daily -At 8:00AM
Register-StockSenseTask -TaskName "StockSense-Reconcile" `
    -Command "`"$PythonPath`" -m stocksense.cli.main reconcile --horizon 10 --lifecycle live --cap-band full_pit" `
    -Trigger $reconcileTrigger -LogFileStem "reconcile"

# Phase J2: paper-run-all steps every ACTIVE paper account forward, so
# opening a new account later never requires touching this script again.
$paperRunTrigger = New-ScheduledTaskTrigger -Daily -At 8:05AM
Register-StockSenseTask -TaskName "StockSense-PaperRun" `
    -Command "`"$PythonPath`" -m stocksense.cli.main paper-run-all" `
    -Trigger $paperRunTrigger -LogFileStem "paper_run"

$retrainTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 6:00AM
Register-StockSenseTask -TaskName "StockSense-RetrainWeekly" `
    -Command "`"$PythonPath`" -m stocksense.cli.main retrain-weekly --horizon 10 --top-n 10 --cost-bps 25.0 --cap-band full_pit" `
    -Trigger $retrainTrigger -LogFileStem "retrain_weekly"

# Phase J1: post-market-close, holdings + positions only (this pass does
# not ingest trades/orders into the canonical `trades` table -- see
# brokers/angel_sync.py's module docstring). 16:30 gives NSE's 15:30
# close time to settle before syncing.
$brokerSyncTrigger = New-ScheduledTaskTrigger -Daily -At 4:30PM
Register-StockSenseTask -TaskName "StockSense-BrokerSync" `
    -Command "`"$PythonPath`" -m stocksense.cli.main broker-sync" `
    -Trigger $brokerSyncTrigger -LogFileStem "broker_sync"

Write-Host ""
Write-Host "Done. Verify with:  Get-ScheduledTask -TaskName 'StockSense-*'"
Write-Host "Logs will appear under: $LogDir"
