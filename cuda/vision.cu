#include "common.cuh"
#include <cuda_bf16.h>

namespace megacapybara {
namespace kernels {
namespace vision {

// Reconstructed from vision.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/vision.cu
// LayerNorm for visual patch embeddings
__global__ void layer_norm_kernel(const float* __restrict__ input,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  __nv_bfloat16* __restrict__ out_bf16,
                                  float* __restrict__ out_fp32,
                                  int hidden_dim) {
    int patch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* x = input + patch_idx * hidden_dim;

    // Compute mean and variance
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += x[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float s_mean;
    if (tid == 0) s_mean = sum / static_cast<float>(hidden_dim);
    __syncthreads();

    float mean = s_mean;
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __shared__ float s_inv_std;
    if (tid == 0) s_inv_std = rsqrtf(var_sum / static_cast<float>(hidden_dim) + 1e-6f);
    __syncthreads();

    float inv_std = s_inv_std;

    // Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float norm = (x[i] - mean) * inv_std;
        if (gamma) norm *= gamma[i];
        if (beta) norm += beta[i];

        if (out_fp32) out_fp32[patch_idx * hidden_dim + i] = norm;
        if (out_bf16) out_bf16[patch_idx * hidden_dim + i] = __float2bfloat16(norm);
    }
}

// 2D Spatial Positional Embedding addition for image grid
__global__ void add_position_kernel(float* __restrict__ patch_embeddings,
                                    const float* __restrict__ pos_table,
                                    int num_patches,
                                    int hidden_dim) {
    int patch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (patch_idx >= num_patches) return;

    for (int d = tid; d < hidden_dim; d += blockDim.x) {
        patch_embeddings[patch_idx * hidden_dim + d] += pos_table[patch_idx * hidden_dim + d];
    }
}

} // namespace vision
} // namespace kernels
} // namespace megacapybara
