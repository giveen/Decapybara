#include "common.cuh"

namespace megacapybara {
namespace kernels {
namespace mtp {

struct SampleSpec {
    float temperature;
    float top_p;
    int top_k;
    int draft_vocab_size;
};

struct CatchUp {
    int main_step;
    int mtp_depth;
    float weight;
};

// Reconstructed from mtp.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/mtp.cu
// Finds top-k draft candidate slices across vocabulary chunks
__global__ void topk_slices_kernel(const float* __restrict__ logits,
                                   int vocab_chunk_size,
                                   float* __restrict__ out_slice_scores,
                                   int* __restrict__ out_slice_indices) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const float* chunk = logits + bid * vocab_chunk_size;
    float max_val = -1e20f;
    int max_idx = -1;

    for (int i = tid; i < vocab_chunk_size; i += blockDim.x) {
        float val = chunk[i];
        if (val > max_val) {
            max_val = val;
            max_idx = bid * vocab_chunk_size + i;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    if (tid == 0) {
        out_slice_scores[bid] = max_val;
        out_slice_indices[bid] = max_idx;
    }
}

// Gathers hidden states for draft tokens across depth heads
template <int HIDDEN_DIM>
__global__ void gather_rows_kernel(const float* __restrict__ src_hidden_states,
                                   int num_rows,
                                   const int* __restrict__ row_indices,
                                   float* __restrict__ dst_hidden_states) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) return;

    int src_row = row_indices[row];
    const float* src = src_hidden_states + src_row * HIDDEN_DIM;
    float* dst = dst_hidden_states + row * HIDDEN_DIM;

    for (int d = tid; d < HIDDEN_DIM; d += blockDim.x) {
        dst[d] = src[d];
    }
}

template __global__ void gather_rows_kernel<2048>(const float*, int, const int*, float*);
template __global__ void gather_rows_kernel<5120>(const float*, int, const int*, float*);

} // namespace mtp
} // namespace kernels
} // namespace megacapybara
