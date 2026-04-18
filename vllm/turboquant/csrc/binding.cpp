// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// pybind11 entry point for the turboquant_cuda extension.

#include <torch/extension.h>

namespace turboquant {
namespace cuda {

void attend_mse_launch(
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
    int block_size,
    int gqa_group,
    int split_len,
    int num_splits
);

} // namespace cuda

namespace fa3 {

int64_t cutlass_version_probe();

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
    int64_t gqa_group
);

} // namespace fa3
} // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "attend_mse",
        &turboquant::cuda::attend_mse_launch,
        "TurboQuant mse attend kernel (CUDA, WMMA + cp.async + split-KV)",
        py::arg("q_rot"),
        py::arg("cache_k_idx"),
        py::arg("cache_k_norm"),
        py::arg("cache_v_idx"),
        py::arg("cache_v_norm"),
        py::arg("block_table"),
        py::arg("seq_id_per_query"),
        py::arg("kv_end_per_query"),
        py::arg("codebook"),
        py::arg("out"),
        py::arg("block_size"),
        py::arg("gqa_group"),
        py::arg("split_len"),
        py::arg("num_splits")
    );
    m.def(
        "cutlass_version_probe",
        &turboquant::fa3::cutlass_version_probe,
        "Return CUTLASS version major*10000+minor*100+patch, or -1 if FA3 "
        "path was not built."
    );
    m.def(
        "attend_mse_fa3",
        &turboquant::fa3::attend_mse_fa3_launch,
        "TurboQuant mse attend kernel (FA3: Hopper wgmma+TMA+warp-spec) "
        "— stub until Stage 3A lands.",
        py::arg("q_rot"),
        py::arg("cache_k_idx"),
        py::arg("cache_k_norm"),
        py::arg("cache_v_idx"),
        py::arg("cache_v_norm"),
        py::arg("block_table"),
        py::arg("seq_id_per_query"),
        py::arg("kv_end_per_query"),
        py::arg("codebook"),
        py::arg("out"),
        py::arg("block_size"),
        py::arg("gqa_group")
    );
}
