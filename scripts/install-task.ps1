# Registers and starts the 'StockSense' scheduled task (daemon at logon and daily at 07:25).
# Run once from a normal PowerShell in the repo. Re-running replaces the task.
$repo = Split-Path -Parent $PSScriptRoot
$action   = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$repo\scripts\run-daemon.ps1`"" -WorkingDirectory $repo
$triggers = @((New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME), (New-ScheduledTaskTrigger -Daily -At 07:25))
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -WakeToRun `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'StockSense' -Action $action -Trigger $triggers -Settings $settings -Description 'StockSense paper-trading daemon' -Force
Start-ScheduledTask -TaskName 'StockSense'
