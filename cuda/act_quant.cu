#include "common.cuh"
#include <cuda_fp8.h>

namespace megacapybara {
namespace kernels {

// Reconstructed from act_quant.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/act_quant.cu
// Quantizes FP32 activations to Microscaling FP8 format (32-element blocks)
__global__ void quantize_mxfp8_kernel(const float* __restrict__ input,
                                      uint8_t* __restrict__ out_scales,
                                      uint8_t* __restrict__ out_quant,
                                      int rows,
                                      int cols) {
    int blocks_per_row = cols / 32;
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = rows * blocks_per_row;

    if (block_id >= total_blocks) {
        return;
    }

    const float* block_input = input + (block_id * 32);

    // Compute absolute maximum value over 32 elements
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float val = fabsf(block_input[i]);
        if (val > amax) {
            amax = val;
        }
    }

    // Compute E8M0 scale factor exponent
    int exp_val = 0;
    if (amax > 0.0f) {
        frexpf(amax, &exp_val);
        exp_val = max(-126, min(127, exp_val));
    }
    uint8_t scale_e8m0 = static_cast<uint8_t>(exp_val + 127);
    out_scales[block_id] = scale_e8m0;

    float inv_scale = (amax > 0.0f) ? ldexpf(1.0f, -exp_val) : 1.0f;

    // Quantize 32 float elements to FP8 (E4M3)
    uint8_t* block_output = out_quant + (block_id * 32);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float scaled = block_input[i] * inv_scale;
        __nv_fp8_e4m3 q = __nv_fp8_e4m3(scaled);
        block_output[i] = *reinterpret_cast<uint8_t*>(&q);
    }
}

} // namespace kernels
} // namespace megacapybara
