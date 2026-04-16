#!/usr/bin/env bash
# Run TurboQuant PyTorch reference tests (no GPU needed).
# Usage: bash test/reference.sh

python -c "
from tests.v1.attention.test_turboquant_reference import (
    test_quantize_dequantize_roundtrip_gaussian,
    test_paged_attention_matches_baseline_at_b8,
    test_paged_attention_b4_coherent,
)
test_quantize_dequantize_roundtrip_gaussian(); print('[1/3] quantize roundtrip OK')
test_paged_attention_matches_baseline_at_b8();  print('[2/3] b=8 attention OK')
test_paged_attention_b4_coherent();             print('[3/3] b=4 coherent OK')
print('all reference tests passed')
"
