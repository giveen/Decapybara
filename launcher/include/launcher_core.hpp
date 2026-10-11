#pragma once

#include "launcher_estimate.hpp"
#include "launcher_hub.hpp"
#include <string>
#include <vector>

namespace megacapybara {
namespace launcher {

enum class Preset {
    MaxEfficiency,
    Medium,
    MaxQuality,
    Custom
};

struct LauncherConfig {
    Preset preset = Preset::Medium;
    std::string model_file;
    std::string drafter_file;
    std::string vision_file;
    std::string kv_mode = "6/6";
    uint32_t context_length = 32768;
    uint32_t concurrency = 8;
    bool use_kv_paging = false;
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    std::string api_key;
};

// Reconstructed from megacapybara::launcher::command_line
std::string build_command_line(const LauncherConfig& config, const std::string& binary_path = "./megacapybara");

// Applies preset configurations
void apply_preset(LauncherConfig& config, Preset preset);

} // namespace launcher
} // namespace megacapybara
