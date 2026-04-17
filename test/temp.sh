#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time a new ad-hoc
# probe is needed. Do NOT rely on the contents across sessions.
#
# Usage:  bash test/temp.sh
#
# Current probe: "why do EXTRACTS find no DBG lines?" -- locate where
# the python _dbg_write output actually landed (stderr vs file), and
# how the server logs look.

set -u

cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"

echo "==== baseline_latest ===="
ls -la logs/baseline_latest/ 2>/dev/null || echo "(missing)"

echo ""
echo "==== diag_bypass_latest ===="
ls -la logs/diag_bypass_latest/ 2>/dev/null || echo "(missing)"

echo ""
echo "==== sizes ===="
wc -l logs/baseline_latest/* logs/diag_bypass_latest/* 2>/dev/null

echo ""
echo "==== TURBOQUANT_DBG mentions in server logs ===="
grep -c "TURBOQUANT_DBG" logs/baseline_latest/*.log 2>/dev/null
grep -c "TURBOQUANT_DBG" logs/diag_bypass_latest/*.log 2>/dev/null

echo ""
echo "==== python-side debug files (any with content) ===="
for f in \
    logs/baseline_latest/turboquant_debug_tq8.log \
    logs/baseline_latest/turboquant_debug_tq4.log \
    logs/diag_bypass_latest/turboquant_debug.log \
    logs/turboquant_debug.log \
    /tmp/turboquant_debug.log \
    /mnt/nvme3n1/g00872988/turboquant/turboquant_debug.log; do
    if [ -s "$f" ]; then
        echo "FOUND: $f  ($(wc -l < "$f") lines)"
    fi
done

echo ""
echo "==== head of tq8_server.log (first 30 lines) ===="
head -30 logs/baseline_latest/tq8_server.log 2>/dev/null

echo ""
echo "==== first TURBOQUANT_DBG line in tq8_server.log ===="
grep -m1 "TURBOQUANT_DBG" logs/baseline_latest/tq8_server.log 2>/dev/null || echo "(no match)"
