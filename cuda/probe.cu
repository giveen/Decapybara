#include "common.cuh"

namespace megacapybara {
namespace kernels {

// Reconstructed from probe.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/probe.cu
__global__ void read_kernel(const uint4* __restrict__ ptr, size_t n, unsigned int* __restrict__ result) {
    uint32_t acc = 0;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        // Cache-streaming load: ld.global.cs.v4.u32
        uint4 val = __ldcs(ptr + idx);
        acc ^= val.x ^ val.y ^ val.z ^ val.w;
    }

    if (acc == PROBE_GOLDEN_CONSTANT) {
        *result = PROBE_GOLDEN_CONSTANT;
    }
}

} // namespace kernels
} // namespace megacapybara
