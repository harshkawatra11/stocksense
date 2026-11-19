# Stops and removes the 'StockSense' scheduled task.
Stop-ScheduledTask -TaskName 'StockSense' -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'StockSense' -Confirm:$false
