#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace megacapybara {
namespace kernels {

// Golden ratio constant used for memory integrity and probe validation
constexpr uint32_t PROBE_GOLDEN_CONSTANT = 0x9e3779b9;

// Linear congruential constants for VRAM bit-flip checksums
constexpr uint64_t CHECKSUM_MULT = 8709371129873690708ULL;
constexpr uint64_t CHECKSUM_ADD0 = static_cast<uint64_t>(-7046029254386353131LL);
constexpr uint64_t CHECKSUM_ADD1 = static_cast<uint64_t>(-2691343689449507777LL);

struct PenaltyParams {
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty;
    float thinking_penalty;
};

struct SampleParams {
    float temperature;
    float top_p;
    int32_t top_k;
    uint32_t seed;
};

struct RingSlot {
    int32_t token_id;
    float logprob;
    uint32_t status;
};

} // namespace kernels
} // namespace megacapybara
