#!/usr/bin/env bash
# Kill GPU processes and free memory.
# Usage:
#   bash test/clean_gpu.sh          # show GPU status
#   bash test/clean_gpu.sh kill     # kill your python GPU processes
#   sudo bash test/clean_gpu.sh kill  # kill all python GPU processes

echo "=== GPU status ==="
nvidia-smi

if [ "$1" = "kill" ]; then
    echo ""
    echo "=== killing python GPU processes ==="
    # Find python processes using GPU
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u)
    if [ -z "$PIDS" ]; then
        echo "no GPU processes found"
    else
        for pid in $PIDS; do
            CMD=$(ps -p "$pid" -o comm= 2>/dev/null)
            echo "killing PID $pid ($CMD)"
            kill -9 "$pid" 2>/dev/null || echo "  failed (try: sudo bash test/clean_gpu.sh kill)"
        done
        sleep 2
        echo ""
        echo "=== after cleanup ==="
        nvidia-smi
    fi
fi
