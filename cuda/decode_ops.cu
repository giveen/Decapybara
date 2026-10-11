#include "common.cuh"
#include <cuda_fp8.h>

namespace megacapybara {
namespace kernels {

// Reconstructed from decode_ops.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/decode_ops.cu
// Applies Rotary Position Embedding (RoPE) and RMSNorm to Q/K vectors before attention
__global__ void attn_prep_mx_kernel(const float* __restrict__ q_in,
                                    const float* __restrict__ k_in,
                                    const float* __restrict__ v_in,
                                    float* __restrict__ q_out,
                                    uint8_t* __restrict__ k_quant,
                                    uint8_t* __restrict__ k_scales,
                                    uint8_t* __restrict__ v_quant,
                                    uint8_t* __restrict__ v_scales,
                                    const int* __restrict__ slot_positions,
                                    float rope_theta,
                                    float rope_factor,
                                    int head_dim,
                                    int num_q_heads,
                                    int num_kv_heads,
                                    size_t slot_stride,
                                    const float2* __restrict__ rope_cache) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int pos = slot_positions ? slot_positions[token_idx] : token_idx;
    float2 rope_sincos = rope_cache ? rope_cache[pos * (head_dim / 2) + tid] : make_float2(1.0f, 0.0f);

    // Apply RoPE rotation to 2 consecutive dimensions:
    // (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
    if (head_idx < num_q_heads && tid < (head_dim / 2)) {
        size_t q_offset = (token_idx * num_q_heads + head_idx) * head_dim + (tid * 2);
        float x0 = q_in[q_offset];
        float x1 = q_in[q_offset + 1];

        float cos_val = rope_sincos.x;
        float sin_val = rope_sincos.y;

        q_out[q_offset]     = x0 * cos_val - x1 * sin_val;
        q_out[q_offset + 1] = x0 * sin_val + x1 * cos_val;
    }

    // Microscaling quantization for Key & Value caches into FP8 with block scales
    if (head_idx < num_kv_heads && tid < (head_dim / 32)) {
        size_t kv_offset = (token_idx * num_kv_heads + head_idx) * head_dim + (tid * 32);
        const float* k_block = k_in + kv_offset;

        float k_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            k_amax = fmaxf(k_amax, fabsf(k_block[i]));
        }

        int exp_k = 0;
        if (k_amax > 0.0f) frexpf(k_amax, &exp_k);
        uint8_t scale_byte = static_cast<uint8_t>(exp_k + 127);
        k_scales[token_idx * slot_stride + (head_idx * (head_dim / 32) + tid)] = scale_byte;

        float inv_scale = (k_amax > 0.0f) ? ldexpf(1.0f, -exp_k) : 1.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            __nv_fp8_e4m3 qk = __nv_fp8_e4m3(k_block[i] * inv_scale);
            k_quant[kv_offset + i] = *reinterpret_cast<uint8_t*>(&qk);
        }
    }
}

} // namespace kernels
} // namespace megacapybara
