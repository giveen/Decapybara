#include "tokenizer.hpp"
#include <sstream>
#include <algorithm>

namespace megacapybara {

Tokenizer Tokenizer::from_json(std::string_view /*json_str*/) {
    Tokenizer t;
    // Set standard Qwen special tokens
    t.vocab_to_id_["<|im_start|>"] = 151644;
    t.id_to_vocab_[151644] = "<|im_start|>";
    t.vocab_to_id_["<|im_end|>"]   = 151645;
    t.id_to_vocab_[151645] = "<|im_end|>";

    // Populate byte fallback tokens (0..255)
    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(i));
        t.vocab_to_id_[s] = i;
        t.id_to_vocab_[i] = s;
    }
    return t;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> tokens;
    encode_segment(text, tokens);
    return tokens;
}

void Tokenizer::encode_segment(std::string_view seg, std::vector<int32_t>& out_tokens) const {
    // Byte-level fallback encoding
    for (char c : seg) {
        auto it = vocab_to_id_.find(std::string(1, c));
        if (it != vocab_to_id_.end()) {
            out_tokens.push_back(it->second);
        } else {
            out_tokens.push_back(static_cast<uint8_t>(c));
        }
    }
}

void Tokenizer::bpe(const std::string& piece, std::vector<int32_t>& out_tokens) const {
    encode_segment(piece, out_tokens);
}

std::string Tokenizer::decode(std::span<const int32_t> tokens) const {
    std::string result;
    for (int32_t id : tokens) {
        auto it = id_to_vocab_.find(id);
        if (it != id_to_vocab_.end()) {
            result += it->second;
        } else if (id >= 0 && id < 256) {
            result += static_cast<char>(id);
        }
    }
    return result;
}

int32_t Tokenizer::token_id(std::string_view token_str) const {
    auto it = vocab_to_id_.find(std::string(token_str));
    if (it != vocab_to_id_.end()) {
        return it->second;
    }
    return -1;
}

} // namespace megacapybara
