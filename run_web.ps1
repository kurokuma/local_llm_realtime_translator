$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error ".venv was not found. Run: python -m venv .venv"
}

Set-Location $ProjectRoot
& $Python web_translator.py @args
