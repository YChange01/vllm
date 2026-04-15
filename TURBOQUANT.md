# TurboQuant backend for vLLM

Real 4-bit KV cache quantization (K 侧) 的 vLLM attention backend 实现.
算法基于 [TurboQuant paper](https://arxiv.org/abs/2504.19874), 算法原型在独立
仓库 [tqlite](https://github.com/YChange01/tqlite) 已验证数学正确性.

## 当前状态

**骨架 + Triton kernel 草稿 + PyTorch reference**. 需要在 B200 上迭代调试.

- [x] 核心算法: Hadamard 旋转 + Lloyd-Max 码本 + 4-bit 量化
- [x] vLLM V1 attention backend 骨架 (`TurboQuantAttentionBackend`, `TurboQuantAttentionImpl`)
- [x] Triton kernel 草稿 (`vllm/turboquant/triton_kernels.py`)
- [x] PyTorch reference 实现 (`vllm/turboquant/reference.py`), 可以在 Mac/CPU 验证算法
- [x] Registry 注册 (`VLLM_ATTENTION_BACKEND=TURBOQUANT`)
- [ ] **真正的 4-bit nibble packing** (当前 MVP 用 uint8 存 4-bit, 实际 1 byte/坐标)
- [ ] **Triton kernel 的 B200 正确性 + 性能调优**
- [ ] Cascade attention / chunked prefill
- [ ] FP8 V cache 配合
- [ ] LUT 融合优化 (当前是 dequantize-then-attend, 有空间换成 query-side LUT)

## 代码结构

```
vllm/
├── turboquant/                         # 算法层 (和 vLLM 其他部分解耦)
│   ├── __init__.py
│   ├── codebook.py                     # GaussianCodebook (rotation + Lloyd-Max)
│   ├── packing.py                      # int4 <-> uint8 nibble 打包
│   ├── reference.py                    # PyTorch 朴素实现 (算法 ground truth)
│   └── triton_kernels.py               # Triton 内核 (GPU 快速路径)
├── v1/attention/backends/
│   ├── registry.py                     # 注册 TURBOQUANT 枚举值 (已改)
│   └── turboquant_attn.py              # Backend + Impl + MetadataBuilder
└── ...
tests/v1/attention/
└── test_turboquant_reference.py        # PyTorch reference 单元测试
```

## 在 B200 上验证 (推荐顺序)

### 0. 环境准备
```bash
# PyTorch with CUDA 12.4+, Triton, vLLM deps
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .   # 在本仓库根目录
```

### 1. Reference 单元测试 (不需要 CUDA)
```bash
pytest tests/v1/attention/test_turboquant_reference.py -v
```
这会验证 `vllm/turboquant/reference.py` 的算法正确性. 如果这里挂了,
说明算法层有 bug, 不用管 Triton.

### 2. Triton kernel 正确性 (需要 CUDA)
```bash
python3 -c "
from vllm.turboquant.codebook import GaussianCodebook
from vllm.turboquant import reference, triton_kernels
import torch

# 合成数据对拍 reference vs triton
torch.manual_seed(0)
num_blocks, block_size, num_kv_heads, head_dim = 4, 16, 2, 64
num_seqs, num_q_heads = 2, 4
device = 'cuda'

cb = GaussianCodebook(head_dim, bits=4, seed=0, dtype=torch.float16, device=device)
# ... (构造测试数据, 调 reference 和 triton_kernels, 对比输出)
"
```
(完整对拍脚本见 `scripts/test_triton_parity.py`, 待补.)

### 3. 端到端最小 serve 验证
```bash
# 小模型快速验证
VLLM_ATTENTION_BACKEND=TURBOQUANT \
  vllm serve Qwen/Qwen3-0.6B --port 8000

# 另一终端
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":20}'
```

### 4. 性能 benchmark
```bash
# baseline (fp16 attention)
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  python benchmarks/benchmark_serving.py --model Qwen/Qwen2.5-7B-Instruct

# TurboQuant
VLLM_ATTENTION_BACKEND=TURBOQUANT \
  python benchmarks/benchmark_serving.py --model Qwen/Qwen2.5-7B-Instruct
```

## 预期问题 (B200 上可能遇到)

### A. Triton kernel 编译错误
`triton_kernels.py` 中 `tl.gather` 可能在当前 Triton 版本不存在.
**fallback**: 用 `tl.load(codebook_ptr + k_idx_u8 * element_size)` 替代.

### B. head_dim != 128
MVP 的 `BLOCK_D` 硬编码为 head_dim 的 next_power_of_2. 对 head_dim=64 OK,
head_dim=256 需要调优.

### C. V cache 和 K cache shape 不匹配
vLLM 分配 kv_cache 假设 K/V 同 dtype. 我们的 K 实际用 uint8, V 用 fp16,
在 `TurboQuantAttentionImpl.forward` 里用 `.view(torch.uint8)` 切换 dtype
视图, 但 block 大小分配是按 fp16 算的, 浪费一半. MVP 能跑, 但要真正 4×
省显存需要改 `KVCacheSpec.page_size_bytes` 分开计算 K/V.

### D. GQA
Kernel 里有 `gqa_group = num_q_heads // num_kv_heads`, 但未测试. Qwen3-0.6B
用 GQA (16 q head / 8 kv head), 是第一个测试目标.

## 算法参数

| 参数 | 当前值 | 备注 |
|---|---|---|
| bits per coord | 4 | 写死, 未来参数化 |
| codebook 类型 | Lloyd-Max on N(0,1) | 离线生成, per-layer 相同 |
| 旋转 | Hadamard × 随机符号 | per-layer seed 不同 |
| K/V 量化 | 只量化 K, V 保持 fp16 | 标准做法 |
| per-row normalization | **当前 kernel 未做!** | 真实 K 非 N(0,1), 需要前处理 |

### 关键注意: per-row normalization

我们在 `tqlite` 仓库发现 Qwen3 的 K vectors std 从 2 到 20+ 变化, 如果不
按 L2 norm 归一化就量化, b=4 会完全崩掉. vLLM 这边 kernel 还没加这一步,
**第一次在 B200 上跑预期会 garbage output**.

修复: 在 `_quantize_and_store_kernel` 里先 `norm = rsqrt(sum(k*k))`,
`k *= norm * sqrt(d)`, 存量化后 idx; 在 `_dequant_and_attend_kernel` 里
把 norm 作为 metadata 存一份 (per-key 1 个 fp16 标量), 反量化时恢复.

具体实现: 改 cache_k shape 为 `(num_blocks, block_size, num_kv_heads, head_dim + 1)`,
最后一维多 1 个 fp16 存 norm (需要 cast 成 uint8*2 占位).

## 联系 / 贡献

算法 prototype 在 https://github.com/YChange01/tqlite 有详细讲解和数学推导.
vLLM backend 迭代请在本分支 `turboquant-backend` 提交 PR.
