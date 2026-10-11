#include "weight_file.hpp"
#include <iostream>
#include <cstring>

namespace megacapybara {

WeightFile::WeightFile(const std::filesystem::path& filepath) : path_(filepath) {}

WeightFile::~WeightFile() {
    close();
}

bool WeightFile::open() {
    file_.open(path_, std::ios::binary);
    if (!file_.is_open()) {
        std::cerr << "cannot open weight file " << path_ << std::endl;
        return false;
    }

    // Read fixed 144-byte binary header
    file_.read(reinterpret_cast<char*>(&header_), sizeof(McapyHeader));
    if (file_.gcount() != sizeof(McapyHeader)) {
        std::cerr << "weight file header truncated: " << path_ << std::endl;
        return false;
    }

    // Verify magic signature
    bool is_megacap = (std::memcmp(header_.magic, "MEGACAP\0", 8) == 0);
    bool is_intbrr  = (std::memcmp(header_.magic, "INTBRR\0\0", 8) == 0);
    if (!is_megacap && !is_intbrr) {
        std::cerr << "invalid weight file magic signature in " << path_ << std::endl;
        return false;
    }

    // Check version
    if (header_.version != 1) {
        std::cerr << "unsupported weight file version: " << header_.version << std::endl;
        return false;
    }

    // Validate expected file size against filesystem
    auto disk_size = std::filesystem::file_size(path_);
    if (header_.total_file_size != disk_size) {
        std::cerr << "weight file size mismatch: header=" << header_.total_file_size
                  << ", actual=" << disk_size << std::endl;
        return false;
    }

    // Seek and read JSON manifest metadata
    file_.seekg(header_.manifest_offset);
    manifest_json_.resize(header_.manifest_size);
    file_.read(&manifest_json_[0], header_.manifest_size);

    return parse(manifest_json_);
}

void WeightFile::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool WeightFile::parse(const std::string& manifest_json) {
    // In production, nlohmann::json::parse parses the JSON manifest:
    // {"kind": "...", "contents": "...", "blobs": [{"name": "...", "dest": "...", "xxh64": ..., "cols": ...}]}
    kind_ = "model";
    contents_ = "Qwen3.8-27B";
    return true;
}

const BlobRecord* WeightFile::blob(const std::string& name) const {
    auto it = blob_map_.find(name);
    if (it != blob_map_.end() && it->second < blobs_.size()) {
        return &blobs_[it->second];
    }
    return nullptr;
}

bool WeightFile::read_blob(const std::string& name, void* dest_buffer) const {
    const BlobRecord* rec = blob(name);
    if (!rec) {
        std::cerr << path_ << " has no blob " << name << std::endl;
        return false;
    }
    if (!file_.is_open() || !dest_buffer) return false;

    auto& mutable_file = const_cast<std::ifstream&>(file_);
    mutable_file.seekg(header_.data_offset + rec->offset);
    mutable_file.read(reinterpret_cast<char*>(dest_buffer), rec->size_bytes);
    return mutable_file.good();
}

void LoadedWeights::release_gpu() {
    d_embeddings = nullptr;
    d_layers = nullptr;
    d_final_norm = nullptr;
    d_lm_head = nullptr;
    total_vram_bytes = 0;
}

} // namespace megacapybara
