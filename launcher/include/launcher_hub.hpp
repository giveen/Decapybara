#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <functional>

namespace megacapybara {
namespace launcher {

struct HubFile {
    std::string filename;
    std::string download_url;
    uint64_t size_bytes;
    std::string description;
    bool is_downloaded = false;
};

class Files {
public:
    explicit Files(const std::filesystem::path& model_dir = "model");

    void refresh();
    std::string default_model() const;
    std::string dflash2_for(const std::string& model_file) const;
    std::string mtp_for(const std::string& model_file) const;
    std::string vision_for(const std::string& model_file) const;
    std::vector<std::string> available_models() const { return local_models_; }

private:
    std::filesystem::path model_dir_;
    std::vector<std::string> local_models_;
    std::vector<std::string> local_drafters_;
    std::vector<std::string> local_visions_;
};

class Catalog {
public:
    Catalog();

    const std::vector<HubFile>& get_hub_files() const { return catalog_; }
    bool download_file(const HubFile& file,
                       const std::filesystem::path& dest_dir,
                       std::function<void(uint64_t downloaded, uint64_t total)> progress_callback);

private:
    std::string base_repo_ = "https://huggingface.co/perkel/Qwen3.8-27B-MC/resolve/main/";
    std::vector<HubFile> catalog_;
};

} // namespace launcher
} // namespace megacapybara
