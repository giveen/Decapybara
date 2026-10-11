#include "launcher_core.hpp"
#include <sstream>

namespace megacapybara {
namespace launcher {

void apply_preset(LauncherConfig& config, Preset preset) {
    config.preset = preset;
    switch (preset) {
        case Preset::MaxEfficiency:
            config.kv_mode = "4/4";
            config.use_kv_paging = true;
            config.context_length = 65536;
            config.concurrency = 8;
            break;
        case Preset::Medium:
            config.kv_mode = "6/6";
            config.use_kv_paging = true;
            config.context_length = 32768;
            config.concurrency = 8;
            break;
        case Preset::MaxQuality:
            config.kv_mode = "8/8";
            config.use_kv_paging = false;
            config.context_length = 32768;
            config.concurrency = 4;
            break;
        case Preset::Custom:
            break;
    }
}

std::string build_command_line(const LauncherConfig& config, const std::string& binary_path) {
    std::ostringstream ss;
    ss << binary_path;

    if (!config.model_file.empty()) {
        ss << " --model " << config.model_file;
    }
    if (!config.drafter_file.empty()) {
        ss << " --drafter " << config.drafter_file;
    }
    if (!config.vision_file.empty()) {
        ss << " --vision " << config.vision_file;
    }

    ss << " --kv-mode " << config.kv_mode;
    ss << " --max-context " << config.context_length;
    ss << " --max-concurrency " << config.concurrency;

    if (config.use_kv_paging) {
        ss << " --kv-paging";
    }

    ss << " --host " << config.host;
    ss << " --port " << config.port;

    if (!config.api_key.empty()) {
        ss << " --api-key " << config.api_key;
    }

    return ss.str();
}

} // namespace launcher
} // namespace megacapybara
