#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Two checks while user runs the v_vec-store-first fix:
#   (A) Standalone PyTorch test: do the three torch.zeros() calls in
#       _ensure_buffers actually return non-overlapping CUDA memory?
#       If they overlap, the K-write to cache_k_idx would corrupt
#       cache_v_fp regardless of any kernel reordering.
#   (B) Once `bash test/temp.sh` is re-run after the fix, [verify L*]
#       lines below will show stored_v|.|max ≈ input_v|.|max if the
#       fix worked. Also pulls [ptr L*] lines from the new debug.

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"

echo "=========================================================="
echo "(A) Standalone test: torch.zeros buffers don't alias"
echo "=========================================================="
CUDA_VISIBLE_DEVICES="$GPU" python3 - <<'PYEOF'
import torch, itertools

device = "cuda"
num_blocks = 18645
block_size = 16
num_kv_heads = 8
head_size = 128
shape_dim = (num_blocks, block_size, num_kv_heads, head_size)
shape_meta = (num_blocks, block_size, num_kv_heads)

# Mimic _ensure_buffers allocation order.
v_fp   = torch.zeros(shape_dim,  dtype=torch.bfloat16, device=device)
k_idx  = torch.zeros(shape_dim,  dtype=torch.uint8,    device=device)
k_norm = torch.zeros(shape_meta, dtype=torch.float32,  device=device)

bufs = {"v_fp": v_fp, "k_idx": k_idx, "k_norm": k_norm}
for n, t in bufs.items():
    p = t.data_ptr()
    nb = t.numel() * t.element_size()
    print(f"  {n:<8} ptr=0x{p:x}  nbytes={nb:>13,}  end=0x{p+nb:x}")

print()
print("  pairwise overlap check (any True = aliased):")
for (n1, t1), (n2, t2) in itertools.combinations(bufs.items(), 2):
    p1, nb1 = t1.data_ptr(), t1.numel() * t1.element_size()
    p2, nb2 = t2.data_ptr(), t2.numel() * t2.element_size()
    overlap = max(p1, p2) < min(p1 + nb1, p2 + nb2)
    print(f"    {n1} vs {n2}: overlap = {overlap}")

# Sanity: also write to k_idx and read back v_fp -- if they alias,
# v_fp would now contain non-zero data. Allocate fresh and test.
v_fp.zero_()
k_idx.zero_()
k_idx.fill_(0xff)  # all 255s in k_idx
print()
print(f"  after k_idx.fill_(0xff): v_fp.abs().max() = {v_fp.abs().max().item()}")
print(f"  (should be 0.0 if buffers are independent)")
PYEOF

echo ""
echo "=========================================================="
echo "(B) Latest [verify L*] from baseline_latest"
echo "    (re-run baseline.sh after pulling the v_vec-first fix"
echo "     to populate this; otherwise it's stale)"
echo "=========================================================="
DBG="logs/baseline_latest/turboquant_debug_tq8.log"
if [ -f "$DBG" ]; then
    echo "----- [ptr L*] (allocation pointers) -----"
    grep "^\[ptr L" "$DBG" | head -20
    echo ""
    echo "----- [verify L0..L4] -----"
    grep "^\[verify L[0-4]\]" "$DBG"
else
    echo "(missing $DBG -- run baseline.sh first)"
fi
