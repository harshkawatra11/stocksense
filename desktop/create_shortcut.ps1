# StockSense desktop shortcut creator (Phase G/E1). Creates a Windows
# .lnk on the user's real Desktop pointing at launch.bat -- re-runnable
# safely (WScript.Shell's CreateShortcut overwrites an existing .lnk at
# the same path rather than erroring or duplicating it).
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot "launch.bat"
if (-not (Test-Path $launcher)) {
    throw "launch.bat not found at $launcher -- run this script from within the repo's desktop/ folder."
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "StockSense.lnk"

$iconPath = Join-Path $PSScriptRoot "build\icon.ico"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.WindowStyle = 7  # minimized -- the batch window is just launch plumbing, Electron opens its own window
$shortcut.Description = "StockSense control center"
if (Test-Path $iconPath) {
    $shortcut.IconLocation = $iconPath
}
$shortcut.Save()

Write-Output "Shortcut created at $shortcutPath -> $launcher"
