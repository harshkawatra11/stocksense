@echo off
REM StockSense desktop launcher (Phase G/E1). Double-clickable entry
REM point for the shortcut created by desktop/create_shortcut.ps1.
REM
REM Recommendation over an electron-builder installer, stated once here:
REM a packaged .exe would need a bundled Python interpreter AND the
REM src/ tree, while the actual data (a multi-GB DuckDB file plus a
REM data_store/ cache) lives on this machine and cannot ship inside an
REM installer anyway. For a single-user tool whose backend and data are
REM both already local, this batch file running the existing `npm
REM start` gives one-click access with none of that packaging fragility.
REM
REM %~dp0 is this .bat file's own directory (desktop/), regardless of
REM where the shortcut invoking it lives or what the current directory
REM was -- so this works whether launched from the Desktop, the Start
REM Menu, or anywhere else.
cd /d "%~dp0"
call npm start
if errorlevel 1 (
    echo.
    echo StockSense exited with an error -- see above.
    pause
)
