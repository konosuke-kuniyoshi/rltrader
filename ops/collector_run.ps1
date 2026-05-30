<#
.SYNOPSIS
  Start the collector service and stop it after 6 hours.

.DESCRIPTION
  Requires Docker Desktop and Docker Compose. Copy `ops/.env.example` to `.env` and
  fill in `BINANCE_API_KEY` and `BINANCE_API_SECRET` before running.

.EXAMPLE
  .\ops\collector_run.ps1
#>

Set-StrictMode -Version Latest

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Push-Location $root

Write-Host "Starting collector..."
docker-compose up -d collector

Write-Host "Tailing collector logs (press Ctrl+C to detach)"
Start-Process -NoNewWindow -FilePath docker-compose -ArgumentList 'logs -f collector'

Write-Host "Collector will be stopped in 6 hours (21600 seconds)."
Start-Sleep -Seconds 21600

Write-Host "Stopping collector..."
docker-compose stop collector

Write-Host "Collector stopped after 6 hours."

Pop-Location
