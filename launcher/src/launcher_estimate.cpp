#include "launcher_estimate.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace megacapybara {
namespace launcher {

double kv_bytes_per_token(const std::string& kv_mode, uint32_t num_layers, uint32_t num_kv_heads, uint32_t head_dim) {
    // Total KV dimensions: 2 (K and V) * num_layers * num_kv_heads * head_dim
    double total_elements = 2.0 * num_layers * num_kv_heads * head_dim;

    // Precision scaling
    double bits_per_element = 16.0;
    if (kv_mode == "6/6") bits_per_element = 6.0;
    else if (kv_mode == "8/8") bits_per_element = 8.0;
    else if (kv_mode == "4/4") bits_per_element = 4.0;
    else if (kv_mode == "6/4") bits_per_element = 5.0;

    return (total_elements * bits_per_element) / 8.0;
}

uint32_t fit_context(double free_vram_gib, const std::string& kv_mode, uint32_t max_concurrency) {
    if (free_vram_gib <= 0.0) return 0;

    double free_bytes = free_vram_gib * 1024.0 * 1024.0 * 1024.0;
    double bytes_per_tok = kv_bytes_per_token(kv_mode);

    double total_tokens = free_bytes / bytes_per_tok;
    uint32_t per_conversation = static_cast<uint32_t>(total_tokens / std::max(1u, max_concurrency));

    // Snap to standard 4096 boundaries
    return (per_conversation / 4096) * 4096;
}

Estimate calculate_estimate(const std::string& model_size,
                            const std::string& kv_mode,
                            uint32_t context_length,
                            uint32_t concurrency,
                            bool use_kv_paging,
                            double available_vram_gib) {
    Estimate est;
    est.vram.available_gib = available_vram_gib;

    // Weight sizes for Qwen3.8-27B variants
    if (model_size == "tiny") est.vram.weights_gib = 13.5;
    else if (model_size == "small") est.vram.weights_gib = 15.2;
    else if (model_size == "large") est.vram.weights_gib = 20.8;
    else est.vram.weights_gib = 17.6; // medium default

    // Vision tower and speculator weights
    est.vram.vision_gib = 0.8;
    est.vram.activations_gib = 1.2;
    est.vram.reserve_gib = 1.0;

    // KV Cache calculation
    double bpt = kv_bytes_per_token(kv_mode);
    uint32_t effective_slots = use_kv_paging ? std::max(1u, concurrency / 2) : concurrency;
    double total_kv_bytes = bpt * context_length * effective_slots;
    est.vram.kv_cache_gib = total_kv_bytes / (1024.0 * 1024.0 * 1024.0);

    est.vram.total_gib = est.vram.weights_gib + est.vram.kv_cache_gib +
                         est.vram.activations_gib + est.vram.vision_gib + est.vram.reserve_gib;

    est.fits_in_vram = (est.vram.total_gib <= est.vram.available_gib);

    // RAM tier recommendations
    est.ram.pinned_pool_gib = use_kv_paging ? 24.0 : 8.0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "Plan: Weights=" << est.vram.weights_gib << " GiB, KV Cache=" << est.vram.kv_cache_gib
       << " GiB (" << context_length << " tokens x " << concurrency << " slots), Total VRAM="
       << est.vram.total_gib << " / " << est.vram.available_gib << " GiB ["
       << (est.fits_in_vram ? "OK" : "EXCEEDED") << "]";
    est.report = ss.str();

    return est;
}

} // namespace launcher
} // namespace megacapybara
