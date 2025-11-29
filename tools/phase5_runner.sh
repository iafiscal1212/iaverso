#!/usr/bin/env bash
set -euo pipefail

LOG="/root/NEO_EVA/logs/phase5_runner.log"
echo "== Phase5 start: $(date -Is)" | tee -a "$LOG"

cd /root/NEO_EVA/tools

# --- A) Phase 4 integration test ---
echo "[A] Phase 4 integration test..." | tee -a "$LOG"
if [ -f phase4_integration.py ]; then
  timeout 60 python3 phase4_integration.py 2>&1 | tee -a "$LOG" || true
fi

# --- B) Run EVA cycles ---
echo "[B] Running EVA cycles..." | tee -a "$LOG"
if [ -f /root/EVASYNT/core/evasynt.py ]; then
  timeout 120 python3 /root/EVASYNT/core/evasynt.py --cycles 500 2>&1 | tee -a "$LOG" || true
fi

# --- C) Analysis suite ---
echo "[C] Running analysis..." | tee -a "$LOG"
if [ -f analysis.py ]; then
  python3 analysis.py jacobian --world neo 2>&1 | tee -a "$LOG" || true
  python3 analysis.py susceptibility --world neo 2>&1 | tee -a "$LOG" || true
  python3 analysis.py nulls --world neo 2>&1 | tee -a "$LOG" || true
fi

# --- D) Quick checks ---
echo "[D] Quick checks..." | tee -a "$LOG"
if [ -f quick_checks.py ]; then
  timeout 120 python3 quick_checks.py 2>&1 | tee -a "$LOG" || true
fi

# --- E) Status report ---
echo "[E] Status report..." | tee -a "$LOG"
if [ -f status_report.py ]; then
  python3 status_report.py 2>&1 | tee -a "$LOG" || true
fi

# --- F) Phase 5 report ---
echo "[F] Phase 5 report..." | tee -a "$LOG"
python3 phase5_report.py 2>&1 | tee -a "$LOG" || true

echo "== Phase5 end: $(date -Is)" | tee -a "$LOG"
echo "[OK] Results in /root/NEO_EVA/results/"
