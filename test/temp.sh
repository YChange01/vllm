#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: PYREF and Triton produce identical broken text for mse b=8
# on long prompt. Bug is in the algorithm, NOT the Triton implementation.
# Three quick checks (no model launch needed) to localize it:
#   (1) PYREF [out L0..L31] |attn|.mean / |attn|.max -- check if PYREF
#       respects the soft-max bound  out <= max|v|. If yes, math bound
#       holds and Triton has an additional impl bug. If no, my bound
#       reasoning was wrong and we look elsewhere.
#   (2) Hadamard sanity:  H @ H.T == I, ||H||^2 per row == 1, H[0,0]
#       == 1/sqrt(d). Catches normalization or construction error.
#   (3) Codebook range vs rotated coord range -- catches "codebook
#       trained on N(0,1) but rotated values live elsewhere" mismatch.

set -u

cd "$(dirname "$0")/.."

PYREF_LOG="logs/pyref_latest/turboquant_debug.log"

echo "=========================================================="
echo "(1) PYREF per-layer |attn|.mean / |attn|.max  (call=0 only)"
echo "    rows where MSE means equal max|v| upper bound = 0.482"
echo "    are the math-correct case; bigger means something off"
echo "=========================================================="
if [ -f "$PYREF_LOG" ]; then
    grep "^\[out " "$PYREF_LOG" | head -40
else
    echo "(missing $PYREF_LOG)"
fi

echo ""
echo "=========================================================="
echo "(2) Hadamard sanity check"
echo "    ||H @ H.T - I||_max should be ~0  (orthonormal)"
echo "    H[0,0] should equal 1/sqrt(128) = 0.0883883..."
echo "    each row's L2 squared should be 1.0"
echo "=========================================================="
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from vllm.turboquant.codebook import _hadamard_matrix
H = _hadamard_matrix(128, torch.float32)
HHt = H @ H.T
print('||H @ H.T - I||_max =', (HHt - torch.eye(128)).abs().max().item())
print('H[0,0]              =', H[0,0].item(), ' expected', 1/128**0.5)
print('row L2^2 mean       =', (H*H).sum(dim=1).mean().item(), ' expected 1.0')
print('H symmetric?        =', torch.equal(H, H.T))
"

echo ""
echo "=========================================================="
echo "(3) Codebook range vs expected rotated-coord range"
echo "    rotated coords ~ N(0,1) so should fit in [-3, 3]."
echo "    For mse b=8, codebook should span roughly that range."
echo "=========================================================="
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from vllm.turboquant.codebook import QuantState
state = QuantState(algo='mse', bits=8, head_dim=128, seed=0,
                   dtype=torch.float32, device='cpu')
cb = state.codebook
print('codebook K_CB        =', cb.shape[0])
print('codebook[0]          =', float(cb[0]),  '(min centroid)')
print('codebook[-1]         =', float(cb[-1]), '(max centroid)')
print('codebook spacing min =', float((cb[1:] - cb[:-1]).min()))
print('codebook spacing max =', float((cb[1:] - cb[:-1]).max()))
print('boundaries.shape     =', state.boundaries.shape)
"

echo ""
echo "=========================================================="
echo "(4) Pure-python store+attend on REAL Llama L0 K (single token)"
echo "    Take token 1 from the long-prompt L0 store log:"
echo "    k = [3.5, 3.03, 3.44, -1.88, ...] (we know first 4 dims),"
echo "    fake the other 124 dims as zeros, run through reconstruction,"
echo "    show what main_dot * k_norm * inv_d would be vs the true <q,k>/sqrt(d)."
echo "    If reconstructed logit is MUCH bigger / smaller than expected,"
echo "    the algorithm scaling is wrong."
echo "=========================================================="
python3 -c "
import torch, math, sys
sys.path.insert(0, '.')
from vllm.turboquant.codebook import QuantState

torch.manual_seed(0)
d = 128
state = QuantState(algo='mse', bits=8, head_dim=d, seed=0,
                   dtype=torch.float32, device='cpu')
H = state.H
signs = state.signs
codebook = state.codebook
boundaries = state.boundaries

# A representative Llama L0 K: per-coord magnitude ~3, sparse sample.
k = torch.zeros(d, dtype=torch.float32)
k[0], k[1], k[2], k[3] = 3.5, 3.03, 3.44, -1.88
k[10:30] = torch.randn(20) * 1.0
k[60:80] = torch.randn(20) * 1.0
k_norm = float(k.norm())
print(f'k_norm                 = {k_norm:.4f}')

k_normed = k * (math.sqrt(d) / k_norm)
print(f'||k_normed||           = {k_normed.norm().item():.4f}  (expect sqrt(d)={math.sqrt(d):.4f})')

rotated = H @ (signs * k_normed)
print(f'||rotated||            = {rotated.norm().item():.4f}  (expect sqrt(d))')
print(f'rotated min,max        = {float(rotated.min()):.3f}, {float(rotated.max()):.3f}')
print(f'rotated per-coord std  = {float(rotated.std()):.3f}  (expect ~1.0 for codebook fit)')

# Quantize
idx = torch.zeros(d, dtype=torch.int32)
for i in range(codebook.shape[0] - 1):
    idx += (rotated > boundaries[i]).int()
rk = codebook[idx.long()]
print(f'rk min,max             = {float(rk.min()):.3f}, {float(rk.max()):.3f}')
print(f'reconstruction error   = ||rotated - rk|| = {(rotated-rk).norm().item():.4f}')
print(f'  rel err              = {(rotated-rk).norm().item() / rotated.norm().item():.4%}')

# Pretend a query q = e_0 (unit basis), expect <q,k> = k[0] = 3.5
q = torch.zeros(d, dtype=torch.float32)
q[0] = 1.0
true_logit = (q * k).sum().item() / math.sqrt(d)
print(f'true   <q,k>/sqrt(d)   = {true_logit:.4f}  (using q=e_0, so equals k[0]/sqrt(d) = {3.5/math.sqrt(d):.4f})')

q_rot = H @ (signs * q)
main_dot = (q_rot * rk).sum().item()
recon_logit = main_dot * k_norm / d
print(f'recon logit            = main_dot * k_norm / d = {recon_logit:.4f}')
print(f'logit relative error   = {abs(recon_logit - true_logit) / max(abs(true_logit), 1e-9):.4%}')
"
