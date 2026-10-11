#include "common.cuh"
#include <cuda_fp16.h>

namespace megacapybara {
namespace kernels {

// Reconstructed from attn_prefill.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/attn_prefill.cu
// Dequantizes compressed KV cache (FP8 / INT4) into FP16 for prompt prefill attention
template <int BITS_K, int BITS_V>
__global__ void kv_dequant_kernel(const uint8_t* __restrict__ k_quant,
                                  const uint8_t* __restrict__ k_scales,
                                  const uint8_t* __restrict__ v_quant,
                                  const uint8_t* __restrict__ v_scales,
                                  int num_tokens,
                                  int num_kv_heads,
                                  int head_dim,
                                  __half* __restrict__ out_k_fp16,
                                  __half* __restrict__ out_v_fp16,
                                  int slot_stride) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;

    size_t kv_offset = (token_idx * num_kv_heads + head_idx) * head_dim + tid;
    int scale_idx = (token_idx * num_kv_heads + head_idx) * (head_dim / 32) + (tid / 32);

    // Read scale and compute multiplier
    uint8_t k_scale_byte = k_scales[scale_idx];
    float k_scale = ldexpf(1.0f, static_cast<int>(k_scale_byte) - 127);

    uint8_t v_scale_byte = v_scales[scale_idx];
    float v_scale = ldexpf(1.0f, static_cast<int>(v_scale_byte) - 127);

    // Dequantize K and V bytes
    uint8_t qk_byte = k_quant[kv_offset];
    float k_val = static_cast<float>(*reinterpret_cast<int8_t*>(&qk_byte)) * k_scale;
    out_k_fp16[kv_offset] = __float2half(k_val);

    uint8_t qv_byte = v_quant[kv_offset];
    float v_val = static_cast<float>(*reinterpret_cast<int8_t*>(&qv_byte)) * v_scale;
    out_v_fp16[kv_offset] = __float2half(v_val);
}

// Flash attention speculative verification kernel
template <int M_TILES, int N_TILES>
__global__ void attn_verify_kernel(const float* __restrict__ q_tokens,
                                  const void* __restrict__ k_cache,
                                  const uint8_t* __restrict__ k_scales,
                                  const void* __restrict__ v_cache,
                                  const uint8_t* __restrict__ v_scales,
                                  const int* __restrict__ slot_offsets,
                                  int draft_len,
                                  int context_len,
                                  int head_dim,
                                  float* __restrict__ out_attn,
                                  int num_heads,
                                  size_t kv_stride) {
    int draft_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (draft_idx >= draft_len || head_idx >= num_heads) return;

    // Dot product attention score accumulator
    float score_acc = 0.0f;
    const float* q = q_tokens + (draft_idx * num_heads + head_idx) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        score_acc += q[d];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        score_acc += __shfl_down_sync(0xffffffff, score_acc, offset);
    }

    if (tid == 0) {
        out_attn[draft_idx * num_heads + head_idx] = score_acc * rsqrtf(static_cast<float>(head_dim));
    }
}

template __global__ void kv_dequant_kernel<1, 1>(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int, int, int, __half*, __half*, int);
template __global__ void attn_verify_kernel<2, 1>(const float*, const void*, const uint8_t*, const void*, const uint8_t*, const int*, int, int, int, float*, int, size_t);

} // namespace kernels
} // namespace megacapybara
