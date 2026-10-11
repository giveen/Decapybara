#include "common.cuh"
#include <curand_kernel.h>

namespace megacapybara {
namespace kernels {

// Reconstructed from sample.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/sample.cu
__global__ void cand_max_kernel(const float* __restrict__ logits,
                                uint32_t* __restrict__ candidate_indices,
                                const PenaltyParams* __restrict__ penalties,
                                int num_candidates,
                                float* __restrict__ out_max_logit,
                                const int* __restrict__ presence_counts,
                                int presence_stride,
                                size_t total_vocab_size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    if (idx >= num_candidates) return;

    float logit = logits[idx];

    // Apply repetition and frequency penalties
    if (penalties && presence_counts) {
        int count = presence_counts[idx];
        if (count > 0) {
            logit -= penalties->presence_penalty + (penalties->frequency_penalty * count);
        }
    }

    // Warp-level maximum reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        logit = fmaxf(logit, __shfl_down_sync(0xffffffff, logit, offset));
    }

    if (tid == 0) {
        atomicMax(reinterpret_cast<int*>(out_max_logit), __float_as_int(logit));
    }
}

__global__ void sample_next_kernel(const float* __restrict__ probabilities,
                                   const int* __restrict__ candidate_tokens,
                                   const SampleParams* __restrict__ params,
                                   const int* __restrict__ num_candidates,
                                   uint32_t* __restrict__ rng_states,
                                   int* __restrict__ out_token,
                                   int* __restrict__ out_accepted_count,
                                   uint32_t* __restrict__ out_status,
                                   RingSlot* __restrict__ ring_slots,
                                   size_t ring_head) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // RNG generation
        curandStatePhilox4_32_10_t state;
        curand_init(params->seed, 0, 0, &state);
        float rand_val = curand_uniform(&state);

        // Cumulative probability sampling
        float cumsum = 0.0f;
        int selected_token = candidate_tokens[0];
        int count = *num_candidates;

        for (int i = 0; i < count; ++i) {
            cumsum += probabilities[i];
            if (rand_val <= cumsum) {
                selected_token = candidate_tokens[i];
                break;
            }
        }

        *out_token = selected_token;
        if (ring_slots) {
            ring_slots[ring_head].token_id = selected_token;
            ring_slots[ring_head].logprob = logf(fmaxf(probabilities[0], 1e-12f));
            ring_slots[ring_head].status = 1;
        }
    }
}

} // namespace kernels
} // namespace megacapybara
