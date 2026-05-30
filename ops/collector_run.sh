#!/usr/bin/env bash
# Start the collector service and stop it after 6 hours (21600 seconds).
# Usage: copy ops/.env.example to .env and edit secrets first, then run this script.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting collector..."
docker-compose up -d collector

echo "Collector is running. Tailing logs (Ctrl-C to detach)"
docker-compose logs -f collector &
LOG_PID=$!

echo "Will stop collector in 6 hours (21600 seconds)."
sleep 21600

echo "Stopping collector..."
docker-compose stop collector

echo "Stopping log tail (PID: $LOG_PID)"
kill $LOG_PID 2>/dev/null || true

echo "Collector stopped after 6 hours."
