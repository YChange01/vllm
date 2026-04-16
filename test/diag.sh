#!/usr/bin/env bash
# TurboQuant diagnostic script.
#
# 1. Patches turboquant_attn.py in-place with 4 print probes (guarded by
#    env var TQ_DEBUG so it is a no-op when TQ_DEBUG is unset or 0).
# 2. Starts serve in background, captures serve.log.
# 3. Waits for health, sends 1 query (short single-token prompt).
# 4. Prints filtered debug lines (first 3 layers only) and stops server.
# 5. Restores the original file.
#
# Usage:
#   bash test/diag.sh                # default model / port / card
#   bash test/diag.sh /path model 8009 1
#
# Output artifacts: serve.log, diag.log (both in repo root).

set -u

MODEL="${1:-/mnt/nvme3n1/g00872988/models/Qwen3-0.6B}"
PORT="${2:-8009}"
GPU="${3:-1}"
MAX_LEN="${4:-4096}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT_DIR/vllm/v1/attention/backends/turboquant_attn.py"
BACKUP="$TARGET.bak.diag"
SERVE_LOG="$ROOT_DIR/serve.log"
DIAG_LOG="$ROOT_DIR/diag.log"

cleanup() {
    if [ -n "${SERVE_PID:-}" ]; then
        echo "[diag] stopping server pid=$SERVE_PID" >&2
        kill "$SERVE_PID" 2>/dev/null || true
        wait "$SERVE_PID" 2>/dev/null || true
    fi
    if [ -f "$BACKUP" ]; then
        echo "[diag] restoring $TARGET" >&2
        mv "$BACKUP" "$TARGET"
    fi
}
trap cleanup EXIT INT TERM

if [ ! -f "$TARGET" ]; then
    echo "[diag] $TARGET not found" >&2
    exit 1
fi

echo "[diag] backing up $TARGET -> $BACKUP"
cp "$TARGET" "$BACKUP"

echo "[diag] injecting probes (guarded by TQ_DEBUG)"
python3 - "$TARGET" <<'PYEOF'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()

store_probe = '''
        if int(os.environ.get("TQ_DEBUG", "0")):
            _slot0 = int(slot_mapping[0].item())
            _cv = cache_v.view(-1, self.num_kv_heads, self.head_size)
            _cv_std = _cv[_slot0].float().std().item() if _slot0 >= 0 else -1.0
            print(
                f"[STORE L{self._layer_seed}] nt={num_tokens} "
                f"slot={slot_mapping[:4].tolist()} "
                f"k_std={k.float().std().item():.3f} "
                f"v_std={v.float().std().item():.3f} "
                f"cache_v_at_slot_std={_cv_std:.3f} "
                f"k_idx_at_slot_max={int(self._k_idx[_slot0].max().item()) if _slot0 >= 0 else -1} "
                f"k_norm_at_slot={self._k_norms[_slot0].mean().item() if _slot0 >= 0 else -1:.3f}",
                flush=True,
            )
'''

attn_probe = '''
        if int(os.environ.get("TQ_DEBUG", "0")):
            print(
                f"[ATTN  L{self._layer_seed}] nt={num_tokens} "
                f"seq_lens={attn_metadata.seq_lens[:2].tolist()} "
                f"bt0={attn_metadata.block_table[0, :2].tolist()} "
                f"slot={attn_metadata.slot_mapping[:2].tolist()} "
                f"q_std={q.float().std().item():.3f} "
                f"out_std={attn_out.float().std().item():.3f} "
                f"out_absmean={attn_out.float().abs().mean().item():.4f}",
                flush=True,
            )
'''

# Insert STORE probe before the ``turboquant_store_kv(`` call returns.
store_anchor = "            block_size=kv_cache.shape[2],\n        )\n"
if store_anchor not in src:
    sys.exit("STORE anchor not found; script needs update")
src = src.replace(store_anchor, store_anchor + store_probe, 1)

# Insert ATTN probe after attn_out assignment, before output.copy_.
attn_anchor = "        output.copy_(attn_out.reshape_as(output))\n"
if attn_anchor not in src:
    sys.exit("ATTN anchor not found; script needs update")
src = src.replace(attn_anchor, attn_probe + attn_anchor, 1)

path.write_text(src)
print("[diag] probes injected OK")
PYEOF

export CUDA_VISIBLE_DEVICES="$GPU"
export TQ_DEBUG=1

echo "[diag] starting serve (log -> $SERVE_LOG)"
: > "$SERVE_LOG"
(
    cd "$ROOT_DIR"
    vllm serve "$MODEL" \
        --port "$PORT" \
        --enforce-eager \
        --attention-backend TURBOQUANT \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization 0.7 \
        >>"$SERVE_LOG" 2>&1
) &
SERVE_PID=$!
echo "[diag] serve pid=$SERVE_PID"

echo "[diag] waiting for health (up to 300s)"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[diag] server healthy after ${i} x5s"
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[diag] serve exited early; tail of log:" >&2
        tail -n 50 "$SERVE_LOG" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[diag] server never became healthy" >&2
    tail -n 80 "$SERVE_LOG" >&2
    exit 1
fi

echo "[diag] sending one query"
QUERY_OUT=$(curl -s "http://localhost:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":4}")
echo "[diag] query response:"
echo "$QUERY_OUT"

sleep 1

echo "[diag] writing filtered diag log to $DIAG_LOG"
grep -E '^\[STORE L[0-2] |^\[ATTN  L[0-2] ' "$SERVE_LOG" > "$DIAG_LOG" || true

echo ""
echo "================= first 3 layers, STORE + ATTN ================="
cat "$DIAG_LOG"
echo "==============================================================="
echo ""
echo "[diag] full serve log: $SERVE_LOG"
echo "[diag] filtered diag:  $DIAG_LOG"
echo "[diag] done."
