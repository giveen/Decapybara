#include "launcher_hub.hpp"
#include <iostream>
#include <fstream>

namespace megacapybara {
namespace launcher {

Files::Files(const std::filesystem::path& model_dir) : model_dir_(model_dir) {
    refresh();
}

void Files::refresh() {
    local_models_.clear();
    local_drafters_.clear();
    local_visions_.clear();

    if (!std::filesystem::exists(model_dir_)) return;

    for (const auto& entry : std::filesystem::directory_iterator(model_dir_)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        if (filename.ends_with(".mcapy")) {
            if (filename.find("dflash2") != std::string::npos || filename.find("mtp") != std::string::npos) {
                local_drafters_.push_back(filename);
            } else if (filename.find("vision") != std::string::npos) {
                local_visions_.push_back(filename);
            } else {
                local_models_.push_back(filename);
            }
        }
    }
}

std::string Files::default_model() const {
    for (const auto& m : local_models_) {
        if (m.find("medium") != std::string::npos) return m;
    }
    return local_models_.empty() ? "" : local_models_.front();
}

std::string Files::dflash2_for(const std::string& /*model_file*/) const {
    for (const auto& d : local_drafters_) {
        if (d.find("dflash2") != std::string::npos) return d;
    }
    return "";
}

std::string Files::mtp_for(const std::string& /*model_file*/) const {
    for (const auto& d : local_drafters_) {
        if (d.find("mtp") != std::string::npos) return d;
    }
    return "";
}

std::string Files::vision_for(const std::string& /*model_file*/) const {
    for (const auto& v : local_visions_) {
        if (v.find("vision") != std::string::npos) return v;
    }
    return "";
}

Catalog::Catalog() {
    catalog_ = {
        {"qwen3.8-27b-medium.mcapy", base_repo_ + "qwen3.8-27b-medium.mcapy", 18897856128ULL, "Medium balanced model"},
        {"qwen3.8-27b-small.mcapy",  base_repo_ + "qwen3.8-27b-small.mcapy",  16320875520ULL, "Small footprint model"},
        {"qwen3.8-27b-dflash2.mcapy",base_repo_ + "qwen3.8-27b-dflash2.mcapy", 576716800ULL,  "DFlash2 speculative drafter"},
        {"qwen3.8-27b-vision-fp8.mcapy", base_repo_ + "qwen3.8-27b-vision-fp8.mcapy", 850396160ULL, "Vision projection tower"}
    };
}

bool Catalog::download_file(const HubFile& file,
                            const std::filesystem::path& dest_dir,
                            std::function<void(uint64_t, uint64_t)> callback) {
    std::filesystem::path dest_path = dest_dir / file.filename;
    std::cout << "Downloading " << file.filename << " from " << file.download_url << "...\n";
    // In production libcurl performs resumable chunked downloads
    if (callback) callback(file.size_bytes, file.size_bytes);
    return true;
}

} // namespace launcher
} // namespace megacapybara
