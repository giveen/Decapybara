#pragma once

#include <string>
#include <cstdint>
#include <optional>
#include <vector>

namespace megacapybara {

enum class ContextMode {
    Fixed,
    Pool
};

enum class PrefillPolicy {
    Balanced,
    DecodePriority,
    PrefillPriority
};

enum class LoopGuardAction {
    Off,
    Stop,
    EndThinking,
    Escape
};

enum class SpeculatorType {
    DFlash2,
    MTP,
    Off
};

struct Settings {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    std::string api_key;

    std::string model_path;
    std::string drafter_path;
    std::string vision_path;

    uint32_t max_context = 32768;
    uint32_t max_concurrency = 8;
    std::string kv_mode = "6/6";
    std::string activations = "fp8";
    std::string rope_scaling = "auto";

    bool kv_paging = false;
    uint32_t ram_tier_gb = 24;
    uint32_t disk_tier_gb = 0;
    double vram_reserve_gb = 1.0;

    SpeculatorType speculator = SpeculatorType::DFlash2;
    uint32_t draft_len = 0;
    uint32_t draft_tree = 7;
    bool draft_tree_sampled = true;

    PrefillPolicy prefill_policy = PrefillPolicy::Balanced;
    LoopGuardAction loop_guard = LoopGuardAction::Stop;
    double thinking_penalty = 0.0;

    bool vision_enabled = true;
    uint32_t image_max_pixels = 1048576;
    uint32_t max_image_tokens = 8192;

    uint32_t vram_check_minutes = 10;
    bool prompt_reuse = true;
    bool verify_checksums = true;
    bool tui_enabled = true;
    std::string log_level = "info";

    static Settings load_from_args_and_json(int argc, char** argv, const std::string& config_file = "settings.json");
    static void print_help();
};

} // namespace megacapybara
