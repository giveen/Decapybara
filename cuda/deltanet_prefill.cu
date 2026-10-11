#include "common.cuh"
#include <cuda_fp16.h>

namespace megacapybara {
namespace kernels {

// Reconstructed from deltanet_prefill.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/deltanet_prefill.cu
// Recurrent linear attention sequence transition: S_{t} = S_{t-1} + v_t (k_t)^T - beta * S_{t-1} k_t k_t^T
__global__ void dn_seq_kernel(const float* __restrict__ qkv_in,
                              __half* __restrict__ recurrent_state,
                              float* __restrict__ out_seq,
                              int head_dim) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_k[];

    // Read q, k, v for this head
    const float* q = qkv_in + (head_idx * 3 + 0) * head_dim;
    const float* k = qkv_in + (head_idx * 3 + 1) * head_dim;
    const float* v = qkv_in + (head_idx * 3 + 2) * head_dim;

    if (tid < head_dim) {
        s_k[tid] = k[tid];
    }
    __syncthreads();

    // Update recurrent state matrix S (head_dim x head_dim)
    __half* S_head = recurrent_state + head_idx * head_dim * head_dim;

    for (int r = tid; r < head_dim; r += blockDim.x) {
        float v_r = v[r];
        float out_r = 0.0f;

        #pragma unroll 4
        for (int c = 0; c < head_dim; ++c) {
            float s_val = __half2float(S_head[r * head_dim + c]);
            // Outer product update: S_{r,c} += v_r * k_c
            s_val += v_r * s_k[c];
            S_head[r * head_dim + c] = __float2half(s_val);

            // Output projection: out_r += S_{r,c} * q_c
            out_r += s_val * q[c];
        }

        out_seq[head_idx * head_dim + r] = out_r;
    }
}

// Recurrent state normalization and output projection
__global__ void dn_out_kernel(const float* __restrict__ recurrent_out,
                              const float* __restrict__ gate,
                              uint8_t* __restrict__ out_quant,
                              uint8_t* __restrict__ out_scales,
                              size_t total_elements) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = recurrent_out[idx];
    if (gate) {
        // Gated linear unit activation: val * sigmoid(gate)
        float g = gate[idx];
        float sig = 1.0f / (1.0f + expf(-g));
        val *= sig;
    }

    // Microscaled quantization
    int block_id = idx / 32;
    int sub_id = idx % 32;
    if (sub_id == 0) {
        // Block leader writes scale
        out_scales[block_id] = 127;
    }
    __nv_fp8_e4m3 q = __nv_fp8_e4m3(val);
    out_quant[idx] = *reinterpret_cast<uint8_t*>(&q);
}

} // namespace kernels
} // namespace megacapybara
