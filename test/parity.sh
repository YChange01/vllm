#!/usr/bin/env bash
# Run TurboQuant Triton kernel parity tests.
# Usage: bash test/parity.sh [CASE]
#   CASE: tiny | d64 | gqa | batch | d128 | (empty = all)

CASE="${1:-}"

if [ -z "$CASE" ]; then
    python scripts/test_turboquant_parity.py --verbose
else
    python scripts/test_turboquant_parity.py --case "$CASE" --verbose
fi
