#include "common.cuh"

namespace megacapybara {
namespace kernels {

// Reconstructed from checksum.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/checksum.cu
__global__ void chunk_sums_kernel(const uint4* __restrict__ data,
                                  size_t total_elements,
                                  size_t chunk_elements,
                                  unsigned long long* __restrict__ out_chunk_sums) {
    __shared__ unsigned long long part[64];

    size_t chunk_start = static_cast<size_t>(blockIdx.x) * chunk_elements;
    size_t chunk_end = chunk_start + chunk_elements;
    if (chunk_end > total_elements) {
        chunk_end = total_elements;
    }

    uint64_t acc = 0;
    for (size_t idx = chunk_start + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
        // 128-bit streaming vector load
        uint4 v = __ldcs(data + idx);

        uint64_t w0 = (static_cast<uint64_t>(v.y) << 32) | v.x;
        uint64_t w1 = (static_cast<uint64_t>(v.w) << 32) | v.z;

        uint64_t mult_k = idx * CHECKSUM_MULT;
        acc += w1 * (mult_k + CHECKSUM_ADD1);
        acc += w0 * (mult_k + CHECKSUM_ADD0);
    }

    // Warp reduction using butterfly shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        part[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        uint64_t warp_sum = (lane < (blockDim.x / 32)) ? part[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
        }
        if (lane == 0) {
            out_chunk_sums[blockIdx.x] = warp_sum;
        }
    }
}

} // namespace kernels
} // namespace megacapybara
