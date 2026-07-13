# ============================================================
# StockSense — one-command local startup (Windows)
#   Brings up: DB (Docker) -> checks Ollama -> backend -> scheduler -> frontend
#   Backend & scheduler run on the HOST (they need host Ollama + the claude CLI),
#   only the database runs in Docker.
#
#   Usage:  ./start.ps1            (start everything)
#           ./start.ps1 -NoFrontend
#           ./start.ps1 -NoScheduler
# ============================================================
param(
    [switch]$NoScheduler,
    [switch]$NoFrontend
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root "venv\Scripts\python.exe"

function Section($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

# 0. Kill any already-running backend/scheduler instances -------------------
# This script now also runs automatically at Windows logon (Task Scheduler
# "StockSense-AutoStart"). Without this guard, a logon that happens while a
# manually-started backend/scheduler is still alive (e.g. after sleep/wake,
# or a second logon) would spawn a duplicate process pair — the exact
# duplicate-python-process bug that caused repeated confusion during manual
# restarts on 2026-07-13. Always start from a clean slate.
Section "Clearing any existing StockSense processes"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match [regex]::Escape($root) -and
                   ($_.CommandLine -match "uvicorn backend.main" -or $_.CommandLine -match "scheduler.market_runner") } |
    ForEach-Object {
        Write-Host "Stopping stale process PID $($_.ProcessId): $($_.CommandLine)" -ForegroundColor Yellow
        taskkill /F /T /PID $_.ProcessId *> $null
    }

# 1. Docker engine + DB ------------------------------------------------------
Section "Docker / database"
docker info *> $null
if (-not $?) {
    Write-Host "Docker engine not running — launching Docker Desktop..." -ForegroundColor Yellow
    $dd = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd }
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 5
        docker info *> $null
        if ($?) { break }
    }
    docker info *> $null
    if (-not $?) { Write-Host "Docker still not ready. Start Docker Desktop manually and re-run." -ForegroundColor Red; exit 1 }
}
Push-Location $root
docker compose up -d db
Pop-Location

# Wait for Postgres to report healthy (compose db service has a healthcheck)
Write-Host "Waiting for the database to be ready..."
$dbName = (docker compose ps -q db 2>$null)
for ($i = 0; $i -lt 30; $i++) {
    $health = ""
    if ($dbName) { $health = (docker inspect --format '{{.State.Health.Status}}' $dbName 2>$null) }
    if ($health -eq "healthy") { Write-Host "Database is up." -ForegroundColor Green; break }
    Start-Sleep -Seconds 2
}

# 2. Ollama (SLM macro layer) ------------------------------------------------
Section "Ollama (SLM macro layer)"
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $model = ($tags.models | Select-Object -First 1).name
    Write-Host "Ollama up. Model: $model" -ForegroundColor Green
} catch {
    Write-Host "Ollama not reachable on :11434 — the macro/news layer will fall back to NEUTRAL." -ForegroundColor Yellow
    Write-Host "Start it with:  ollama serve   (and: ollama pull qwen2.5:7b)" -ForegroundColor Yellow
}

# 3. Backend (host, port 8000) ----------------------------------------------
Section "Backend API"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$root'; & '$py' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Write-Host "Backend starting on http://localhost:8000  (new window)" -ForegroundColor Green

# 4. Scheduler (host) --------------------------------------------------------
if (-not $NoScheduler) {
    Section "Market scheduler"
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "cd '$root'; & '$py' -m scheduler.market_runner"
    Write-Host "Scheduler started (new window). Auto-runs pipeline every 30 min, 9:15-15:45 IST Mon-Fri." -ForegroundColor Green
}

# 5. Frontend ----------------------------------------------------------------
if (-not $NoFrontend) {
    Section "Frontend"
    $fe = Join-Path $root "frontend"
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "cd '$fe'; npm run dev"
    Write-Host "Frontend starting on http://localhost:5173  (new window)" -ForegroundColor Green
}

Section "StockSense is up"
Write-Host "  App:      http://localhost:5173   (Live Signals tab)" -ForegroundColor White
Write-Host "  API:      http://localhost:8000/api/health" -ForegroundColor White
Write-Host "  Trigger a run now:  POST http://localhost:8000/api/live/run" -ForegroundColor White
Write-Host ""
Write-Host "Note: signals use the latest OHLCV in the DB. If the daily data isn't current," -ForegroundColor Yellow
Write-Host "      update it first:  & '$py' -m data.pipeline.fetch_historical incremental" -ForegroundColor Yellow
