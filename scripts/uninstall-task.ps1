# NOT USED: the user starts StockSense manually
# Stops and removes the 'StockSense' scheduled task.
Stop-ScheduledTask -TaskName 'StockSense' -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'StockSense' -Confirm:$false
