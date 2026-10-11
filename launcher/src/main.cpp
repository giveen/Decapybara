#include "launcher_core.hpp"
#include "launcher_estimate.hpp"
#include "launcher_hub.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "=== MegaCapybara 1.35 Companion Launcher ===\n\n";

    megacapybara::launcher::Files files("model");
    std::cout << "Local Model Files Detected:\n";
    auto models = files.available_models();
    if (models.empty()) {
        std::cout << "  (none found in model/)\n";
    } else {
        for (const auto& m : models) {
            std::cout << "  - " << m << "\n";
        }
    }

    megacapybara::launcher::LauncherConfig cfg;
    megacapybara::launcher::apply_preset(cfg, megacapybara::launcher::Preset::Medium);
    cfg.model_file = files.default_model();
    cfg.drafter_file = files.dflash2_for(cfg.model_file);
    cfg.vision_file = files.vision_for(cfg.model_file);

    std::cout << "\nSelected Preset: Medium (Balanced)\n";
    auto est = megacapybara::launcher::calculate_estimate("medium", cfg.kv_mode, cfg.context_length, cfg.concurrency, cfg.use_kv_paging);
    std::cout << est.report << "\n\n";

    std::string cmd = megacapybara::launcher::build_command_line(cfg);
    std::cout << "Launch Command Generated:\n  " << cmd << "\n";

    return 0;
}
