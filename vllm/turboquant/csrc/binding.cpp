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
    int gqa_group
);

} // namespace cuda
} // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "attend_mse",
        &turboquant::cuda::attend_mse_launch,
        "TurboQuant mse attend kernel (CUDA, WMMA tensor cores)",
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
