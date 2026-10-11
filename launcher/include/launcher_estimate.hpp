#pragma once

#include <string>
#include <cstdint>
#include <vector>

namespace megacapybara {
namespace launcher {

struct VramBreakdown {
    double weights_gib = 0.0;
    double kv_cache_gib = 0.0;
    double activations_gib = 0.0;
    double vision_gib = 0.0;
    double reserve_gib = 1.0;
    double total_gib = 0.0;
    double available_gib = 32.0; // RTX 5090 = 32 GB GDDR7
};

struct RamBreakdown {
    double pinned_pool_gib = 0.0;
    double total_available_gib = 64.0;
};

struct Estimate {
    VramBreakdown vram;
    RamBreakdown ram;
    double disk_cache_gib = 0.0;
    uint32_t recommended_context = 32768;
    bool fits_in_vram = true;
    std::string report;
};

// Calculates bytes per KV token based on precision mode
double kv_bytes_per_token(const std::string& kv_mode, uint32_t num_layers = 64, uint32_t num_kv_heads = 8, uint32_t head_dim = 128);

// Fits maximum context window into remaining free VRAM
uint32_t fit_context(double free_vram_gib, const std::string& kv_mode, uint32_t max_concurrency = 8);

// Full capacity plan estimate
Estimate calculate_estimate(const std::string& model_size,
                            const std::string& kv_mode,
                            uint32_t context_length,
                            uint32_t concurrency,
                            bool use_kv_paging,
                            double available_vram_gib = 32.0);

} // namespace launcher
} // namespace megacapybara
