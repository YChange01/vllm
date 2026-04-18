// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboQuant FA3-style attend kernel (Hopper wgmma + TMA + warp-spec).
//
// Stage 3A scaffold: compile-time proof that CUTLASS 3.x headers resolve
// and sm_90a codegen is wired. Real kernel follows.

#include <torch/extension.h>

// Only compile CUTLASS-dependent code when the build driver discovered the
// headers. Without this guard, building on machines without CUTLASS would
// break the whole extension.
#ifdef TURBOQUANT_HAS_FA3

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cute/tensor.hpp>

namespace turboquant {
namespace fa3 {

// Probe: returns the CUTLASS version as a uint64 so the Python side can
// assert the extension actually compiled against CUTLASS.
int64_t cutlass_version_probe() {
  return static_cast<int64_t>(cutlass::getVersionMajor()) * 10000 +
         static_cast<int64_t>(cutlass::getVersionMinor()) * 100 +
         static_cast<int64_t>(cutlass::getVersionPatch());
}

void attend_mse_fa3_launch(
    at::Tensor q_rot,
    at::Tensor cache_k_idx,
    at::Tensor cache_k_norm,
    at::Tensor cache_v_idx,
    at::Tensor cache_v_norm,
    at::Tensor block_table,
    at::Tensor seq_id_per_query,
    at::Tensor kv_end_per_query,
    at::Tensor codebook,
    at::Tensor out,
    int64_t block_size,
    int64_t gqa_group) {
  TORCH_CHECK(false,
              "turboquant FA3 kernel not implemented yet (Stage 3A scaffold). "
              "Use TURBOQUANT_USE_CUDA=1 without FA3 for WMMA baseline.");
}

}  // namespace fa3
}  // namespace turboquant

#else  // !TURBOQUANT_HAS_FA3

namespace turboquant {
namespace fa3 {

int64_t cutlass_version_probe() { return -1; }

void attend_mse_fa3_launch(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, int64_t, int64_t) {
  TORCH_CHECK(false, "FA3 path not built. Set TURBOQUANT_BUILD_FA3=1.");
}

}  // namespace fa3
}  // namespace turboquant

#endif
