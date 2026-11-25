@echo off
cd /d %~dp0
if not exist web\node_modules ( call npm --prefix web ci || goto :err )
for /f %%i in ('powershell -NoProfile -Command "if (-not (Test-Path web\out\index.html) -or ((Get-ChildItem web\src -Recurse -File | Measure-Object LastWriteTime -Maximum).Maximum -gt (Get-Item web\out\index.html).LastWriteTime)) {'build'} else {'ok'}"') do set NEED=%%i
if "%NEED%"=="build" ( call npm --prefix web run build || goto :err )
start "" http://127.0.0.1:8765
.venv\Scripts\python.exe -m stocksense serve
goto :eof
:err
echo Build failed. See the messages above.
pause