#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <span>

namespace megacapybara {

class Tokenizer {
public:
    struct Merge {
        int left;
        int right;
        int rank;
    };

    Tokenizer() = default;
    ~Tokenizer() = default;

    static Tokenizer from_json(std::string_view json_str);

    std::vector<int32_t> encode(std::string_view text) const;
    void encode_segment(std::string_view seg, std::vector<int32_t>& out_tokens) const;
    void bpe(const std::string& piece, std::vector<int32_t>& out_tokens) const;

    std::string decode(std::span<const int32_t> tokens) const;
    int32_t token_id(std::string_view token_str) const;

    int32_t bos_token() const { return bos_id_; }
    int32_t eos_token() const { return eos_id_; }
    int32_t pad_token() const { return pad_id_; }

private:
    std::unordered_map<std::string, int32_t> vocab_to_id_;
    std::unordered_map<int32_t, std::string> id_to_vocab_;
    std::unordered_map<uint64_t, int32_t> bpe_ranks_;

    int32_t bos_id_ = 151643;
    int32_t eos_id_ = 151645;
    int32_t pad_id_ = 151643;
};

} // namespace megacapybara
