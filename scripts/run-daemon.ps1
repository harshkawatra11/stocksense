$env:PYTHONUTF8 = "1"
$env:POLARS_MAX_THREADS = "8"
$env:OMP_NUM_THREADS = "8"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
Set-Location (Split-Path -Parent $PSScriptRoot)
& .venv\Scripts\python.exe -m stocksense run
