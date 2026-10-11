#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <fstream>
#include <filesystem>

namespace megacapybara {

// Recovered binary header structure (144 bytes total, checked in WeightFile::open)
#pragma pack(push, 1)
struct McapyHeader {
    char magic[8];           // "MEGACAP\0" or "INTBRR\0\0"
    uint32_t version;        // 1
    uint32_t flags;          // Header flags
    uint64_t manifest_offset;// Offset to JSON metadata manifest in file
    uint64_t manifest_size;  // Size of JSON metadata manifest in bytes
    uint64_t total_file_size;// Total file size (compared against std::filesystem::file_size)
    uint64_t data_offset;    // Start of binary tensor data payload
    uint64_t reserved[12];   // Remaining padding up to 144 bytes
};
#pragma pack(pop)
static_assert(sizeof(McapyHeader) == 144, "McapyHeader must be exactly 144 bytes");

struct BlobRecord {
    std::string name;        // e.g. "head.w", "layers.0.attn.q.w"
    std::string dest;        // Target GPU buffer identifier
    uint64_t offset;         // Offset in payload
    uint64_t size_bytes;     // Byte length
    uint64_t xxh64;          // 64-bit xxHash checksum
    int32_t cols;            // Column dimension
    int32_t rows;            // Row dimension
};

class WeightFile {
public:
    explicit WeightFile(const std::filesystem::path& filepath);
    ~WeightFile();

    bool open();
    void close();

    bool is_model() const { return kind_ == "model"; }
    const std::string& kind() const { return kind_; }
    const std::string& manifest_hex() const { return manifest_json_; }

    const BlobRecord* blob(const std::string& name) const;
    bool read_blob(const std::string& name, void* dest_buffer) const;

private:
    bool parse(const std::string& manifest_json);

    std::filesystem::path path_;
    std::ifstream file_;
    McapyHeader header_{};
    std::string kind_;
    std::string contents_;
    std::string manifest_json_;
    std::vector<BlobRecord> blobs_;
    std::unordered_map<std::string, size_t> blob_map_;
};

struct LoadedWeights {
    void* d_embeddings = nullptr;
    void* d_layers = nullptr;
    void* d_final_norm = nullptr;
    void* d_lm_head = nullptr;
    size_t total_vram_bytes = 0;

    void release_gpu();
};

} // namespace megacapybara
