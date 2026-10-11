#include "common.cuh"
#include <cuda_bf16.h>

namespace megacapybara {
namespace kernels {

// Reconstructed from moe.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/moe.cu
// Top-K MoE routing kernel: calculates gate logits, softmax, and top-k selection
__global__ void moe_route_kernel(const float* __restrict__ hidden_states,
                                 const __nv_bfloat16* __restrict__ router_weights,
                                 const __nv_bfloat16* __restrict__ router_bias,
                                 int num_tokens,
                                 int hidden_dim,
                                 int num_experts,
                                 int* __restrict__ out_selected_experts,
                                 float* __restrict__ out_expert_weights,
                                 float* __restrict__ out_router_logits) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    extern __shared__ float shared_logits[];

    const float* x = hidden_states + token_idx * hidden_dim;
    int tid = threadIdx.x;

    // Compute gate logit for each expert: logit_e = x . W_e + b_e
    for (int e = tid; e < num_experts; e += blockDim.x) {
        float sum = (router_bias != nullptr) ? __bfloat162float(router_bias[e]) : 0.0f;
        const __nv_bfloat16* w_e = router_weights + e * hidden_dim;

        #pragma unroll 4
        for (int d = 0; d < hidden_dim; ++d) {
            sum += x[d] * __bfloat162float(w_e[d]);
        }
        shared_logits[e] = sum;
        if (out_router_logits) {
            out_router_logits[token_idx * num_experts + e] = sum;
        }
    }
    __syncthreads();

    // Single warp performs Softmax & Top-K selection (typically top-8 or top-4)
    if (tid == 0) {
        constexpr int TOP_K = 8;
        int top_indices[TOP_K];
        float top_vals[TOP_K];

        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            top_vals[k] = -1e20f;
            top_indices[k] = -1;
        }

        // Find top-k logits
        for (int e = 0; e < num_experts; ++e) {
            float val = shared_logits[e];
            for (int k = 0; k < TOP_K; ++k) {
                if (val > top_vals[k]) {
                    #pragma unroll
                    for (int j = TOP_K - 1; j > k; --j) {
                        top_vals[j] = top_vals[j - 1];
                        top_indices[j] = top_indices[j - 1];
                    }
                    top_vals[k] = val;
                    top_indices[k] = e;
                    break;
                }
            }
        }

        // Compute Softmax normalization over top-k values
        float max_v = top_vals[0];
        float sum_exp = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            top_vals[k] = expf(top_vals[k] - max_v);
            sum_exp += top_vals[k];
        }

        float inv_sum = 1.0f / fmaxf(sum_exp, 1e-12f);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            out_selected_experts[token_idx * TOP_K + k] = top_indices[k];
            out_expert_weights[token_idx * TOP_K + k] = top_vals[k] * inv_sum;
        }
    }
}

} // namespace kernels
} // namespace megacapybara
