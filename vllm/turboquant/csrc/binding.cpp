// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// pybind11 entry point for the turboquant_cuda extension.

#include <torch/extension.h>

namespace turboquant {
namespace cuda {

void attend_launch(
    at::Tensor q_rot,
    c10::optional<at::Tensor> Sq_opt,
    at::Tensor cache_k_idx,
    at::Tensor cache_k_norm,
    at::Tensor cache_v_idx,
    at::Tensor cache_v_norm,
    c10::optional<at::Tensor> cache_k_qjl_sign_opt,
    c10::optional<at::Tensor> cache_k_rnorm_opt,
    at::Tensor block_table,
    at::Tensor seq_id_per_query,
    at::Tensor kv_end_per_query,
    at::Tensor codebook,
    at::Tensor out,
    int block_size,
    int gqa_group,
    int algo
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
        "attend",
        &turboquant::cuda::attend_launch,
        "TurboQuant attend kernel (CUDA, WMMA + cp.async; mse/prod via algo)",
        py::arg("q_rot"),
        py::arg("Sq") = py::none(),
        py::arg("cache_k_idx"),
        py::arg("cache_k_norm"),
        py::arg("cache_v_idx"),
        py::arg("cache_v_norm"),
        py::arg("cache_k_qjl_sign") = py::none(),
        py::arg("cache_k_rnorm") = py::none(),
        py::arg("block_table"),
        py::arg("seq_id_per_query"),
        py::arg("kv_end_per_query"),
        py::arg("codebook"),
        py::arg("out"),
        py::arg("block_size"),
        py::arg("gqa_group"),
        py::arg("algo")
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
