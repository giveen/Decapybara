#include "common.cuh"

namespace megacapybara {
namespace kernels {
namespace dflash_common {

struct DraftTable {
    int max_nodes;
    int num_candidates;
    int vocab_size;
    int hidden_dim;
};

struct WalkParams {
    float threshold;
    int max_depth;
    int tree_size;
};

} // namespace dflash_common

namespace dflash {

// Reconstructed from drafter.sm_120a.ptx
// Originally: /mnt/d/Projects/IntBrr/engine/src/kernels/drafter.cu
// Embeds drafted candidate tokens for speculative draft verification
__global__ void embed_block_kernel(const uint16_t* __restrict__ token_ids,
                                   const dflash_common::DraftTable* __restrict__ table,
                                   const float* __restrict__ embedding_table,
                                   float* __restrict__ out_hidden_states) {
    int node_idx = blockIdx.x;
    int tid = threadIdx.x;
    int dim = blockDim.x;

    uint32_t token = token_ids[node_idx];
    const float* emb = embedding_table + token * dim;

    out_hidden_states[node_idx * dim + tid] = emb[tid];
}

// Selects best speculative tree path via probability walking
__global__ void selector_walk_kernel(const float* __restrict__ node_probs,
                                     const int* __restrict__ parent_indices,
                                     const float* __restrict__ parent_probs,
                                     const uint16_t* __restrict__ candidate_tokens,
                                     const uint16_t* __restrict__ draft_tree_tokens,
                                     const int* __restrict__ node_depths,
                                     const dflash_common::DraftTable* __restrict__ table,
                                     const dflash_common::WalkParams* __restrict__ params,
                                     int* __restrict__ out_selected_path,
                                     float* __restrict__ out_path_prob,
                                     int* __restrict__ out_path_length,
                                     float* __restrict__ out_scores) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int best_leaf = 0;
        float best_score = -1e20f;
        int total_nodes = params->tree_size;

        for (int i = 0; i < total_nodes; ++i) {
            float prob = node_probs[i];
            int depth = node_depths[i];
            float score = prob / (static_cast<float>(depth) + 1.0f);

            if (score > best_score && prob >= params->threshold) {
                best_score = score;
                best_leaf = i;
            }
        }

        // Trace path back from best leaf to root
        int curr = best_leaf;
        int path_len = 0;
        int temp_path[16];

        while (curr >= 0 && path_len < 16) {
            temp_path[path_len++] = candidate_tokens[curr];
            curr = parent_indices[curr];
        }

        // Reverse into forward sequence order
        for (int i = 0; i < path_len; ++i) {
            out_selected_path[i] = temp_path[path_len - 1 - i];
        }

        *out_path_length = path_len;
        *out_path_prob = best_score;
    }
}

} // namespace dflash
} // namespace kernels
} // namespace megacapybara
