# Run the full pipeline in one command (Windows PowerShell)
# Usage:
#   .\scripts\run_pipeline.ps1
#   .\scripts\run_pipeline.ps1 -SkipWeather
#   .\scripts\run_pipeline.ps1 -SkipSQL
#   .\scripts\run_pipeline.ps1 -SkipML

param(
  [switch]$SkipWeather,
  [switch]$SkipSQL,
  [switch]$SkipML
)

if (-not (Test-Path ".\.venv")) {
  Write-Host "No .venv found. Creating..." -ForegroundColor Yellow
  python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$cmd = "python -m src.run_pipeline"
if ($SkipWeather) { $cmd += " --skip-weather" }
if ($SkipSQL) { $cmd += " --skip-sql" }
if ($SkipML) { $cmd += " --skip-ml" }

Write-Host "Running: $cmd" -ForegroundColor Cyan
Invoke-Expression $cmd
