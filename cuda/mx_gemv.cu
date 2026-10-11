#include "common.cuh"
#include <cuda_fp8.h>

namespace megacapybara {
namespace kernels {

enum class MxFormat {
    NVFP4 = 1,
    MXFP8_E4M3 = 2,
    MXFP8_E5M2 = 3,
    BF16 = 4
};

// Reconstructed from mx_gemv.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/mx_gemv.cu
// Blackwell Microscaling Matrix-Vector GEMV Kernel
template <MxFormat FORMAT, int TILE_M>
__global__ void mx_gemv_kernel(const uint4* __restrict__ weight_data,
                              const uint32_t* __restrict__ weight_scales,
                              const uint8_t* __restrict__ act_data,
                              const uint8_t* __restrict__ act_scales,
                              float* __restrict__ out_vector,
                              int M,
                              int K,
                              int stride_weight,
                              int stride_act,
                              int lda,
                              int ldc) {
    int row = blockIdx.x * TILE_M + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row >= M) return;

    float acc = 0.0f;
    int k_blocks = K / 32;

    // Iterate across 32-element microscaling blocks
    for (int k_b = lane; k_b < k_blocks; k_b += 32) {
        // Fetch 8-bit block exponents (E8M0) for weight and activation
        uint32_t w_scale_byte = (weight_scales[row * (k_blocks / 4) + (k_b / 4)] >> ((k_b % 4) * 8)) & 0xFF;
        uint32_t a_scale_byte = act_scales[k_b];

        int w_exp = static_cast<int>(w_scale_byte) - 127;
        int a_exp = static_cast<int>(a_scale_byte) - 127;
        float combined_scale = ldexpf(1.0f, w_exp + a_exp);

        // Fetch 32-element dot product tile
        const uint4* w_vec = weight_data + (row * k_blocks + k_b);
        uint4 w = *w_vec;
        const uint4* a_vec = reinterpret_cast<const uint4*>(act_data + k_b * 32);
        uint4 a = *a_vec;

        // Perform 32-element FP8 dot product
        const int8_t* w_bytes = reinterpret_cast<const int8_t*>(&w);
        const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            dot += static_cast<float>(w_bytes[i]) * static_cast<float>(a_bytes[i]);
        }
        acc += dot * combined_scale;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        out_vector[row] = acc;
    }
}

// Explicit template instantiations matching the binary
template __global__ void mx_gemv_kernel<MxFormat::NVFP4, 1>(const uint4*, const uint32_t*, const uint8_t*, const uint8_t*, float*, int, int, int, int, int, int);
template __global__ void mx_gemv_kernel<MxFormat::NVFP4, 2>(const uint4*, const uint32_t*, const uint8_t*, const uint8_t*, float*, int, int, int, int, int, int);
template __global__ void mx_gemv_kernel<MxFormat::MXFP8_E4M3, 1>(const uint4*, const uint32_t*, const uint8_t*, const uint8_t*, float*, int, int, int, int, int, int);
template __global__ void mx_gemv_kernel<MxFormat::MXFP8_E4M3, 2>(const uint4*, const uint32_t*, const uint8_t*, const uint8_t*, float*, int, int, int, int, int, int);

} // namespace kernels
} // namespace megacapybara
