# StockSense - one-shot local deployment launcher.
# Idempotent: safe to run repeatedly; kills stale instances before starting fresh.
# Registered to run at Windows logon (see deploy notes). App: http://localhost:8000

$root = "c:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense\stocksense"
$py = "$root\venv\Scripts\python.exe"
$logDir = "$root\logs\$(Get-Date -Format 'yyyy-MM-dd')"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $line
    Add-Content "$logDir\launcher.log" $line
}

Log "--- StockSense launcher start ---"

# 1. Docker containers (DB + Redis). Wait for Docker engine if it's still booting.
$dockerReady = $false
for ($i = 0; $i -lt 30; $i++) {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $dockerReady = $true; break }
    Start-Sleep -Seconds 10
}
if (-not $dockerReady) { Log "ERROR: Docker engine never came up"; exit 1 }
docker start stocksense-db-1 redis-stock 2>$null | Out-Null
Log "Docker containers up"

# Wait for Postgres to accept connections
for ($i = 0; $i -lt 30; $i++) {
    docker exec stocksense-db-1 pg_isready -U stocksense 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 2
}
Log "Postgres ready"

# 2. Kill any stale StockSense python processes (uvicorn / scheduler)
$stale = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'uvicorn backend\.main|scheduler\.market_runner' }
foreach ($p in $stale) {
    taskkill /F /T /PID $p.ProcessId 2>$null | Out-Null
    Log "Killed stale PID $($p.ProcessId)"
}
Start-Sleep -Seconds 2

# 3. Backend (also serves the built frontend at /)
Start-Process -FilePath $py `
    -ArgumentList "-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$logDir\backend.out.log" `
    -RedirectStandardError "$logDir\backend.err.log" `
    -WindowStyle Hidden
Log "Backend started (port 8000)"

# 4. Scheduler (the autonomous brain)
Start-Process -FilePath $py `
    -ArgumentList "-m","scheduler.market_runner" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$logDir\scheduler.out.log" `
    -RedirectStandardError "$logDir\scheduler.err.log" `
    -WindowStyle Hidden
Log "Scheduler started"

# 5. Verify backend answers
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        $h = Invoke-RestMethod "http://localhost:8000/api/health" -TimeoutSec 3
        if ($h.status -eq "ok") { $ok = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if ($ok) { Log "HEALTH OK - app live at http://localhost:8000" }
else { Log "WARN: backend health check did not pass in time" }

Log "--- launcher done ---"
