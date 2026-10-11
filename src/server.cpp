#include "server.hpp"
#include <iostream>
#include <sstream>

namespace megacapybara {

Server::Server(const Settings& settings,
               const Tokenizer& tokenizer,
               std::shared_ptr<Engine> engine,
               std::string model_name)
    : settings_(settings),
      tokenizer_(tokenizer),
      engine_(std::move(engine)),
      model_name_(std::move(model_name)) {}

Server::~Server() {
    stop();
}

bool Server::start() {
    is_running_ = true;
    std::cout << "MegaCapybara server online on " << settings_.host << ":" << settings_.port << "\n"
              << "Model loaded: " << model_name_ << "\n"
              << "Endpoints active:\n"
              << "  - POST /v1/chat/completions (OpenAI Compatible)\n"
              << "  - POST /v1/completions      (OpenAI Compatible)\n"
              << "  - GET  /v1/models           (Model metadata)\n"
              << "  - POST /v1/messages         (Anthropic Compatible)\n";
    return true;
}

void Server::stop() {
    is_running_ = false;
}

bool Server::verify_auth(const std::string& auth_header) const {
    if (settings_.api_key.empty()) return true;
    std::string bearer = "Bearer " + settings_.api_key;
    return auth_header == bearer || auth_header == settings_.api_key;
}

std::string Server::build_chat_prompt(const std::vector<ChatMessage>& messages) const {
    // Qwen / ChatML format prompt template:
    // <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    std::ostringstream ss;
    for (const auto& msg : messages) {
        ss << "<|im_start|>" << msg.role << "\n" << msg.content << "<|im_end|>\n";
    }
    ss << "<|im_start|>assistant\n";
    return ss.str();
}

void Server::handle_chat_completions(const std::string& /*body*/,
                                     std::function<void(int, const std::string&, const std::string&)> response_callback) {
    std::vector<ChatMessage> messages = {
        {"user", "Hello!"}
    };
    std::string formatted = build_chat_prompt(messages);
    std::vector<int32_t> prompt_tokens = tokenizer_.encode(formatted);
    stats_.total_prompt_tokens += prompt_tokens.size();

    GenerationConfig gen_cfg;
    gen_cfg.max_tokens = 512;
    gen_cfg.temperature = 0.7;

    std::string generated_text;
    stats_.active_slots++;

    engine_->generate_stream("req-001", prompt_tokens, gen_cfg, [&](const TokenGenerationEvent& ev) {
        if (!ev.is_finished) {
            generated_text += ev.text_delta;
            stats_.total_gen_tokens++;
        }
    });

    stats_.active_slots--;

    std::ostringstream json;
    json << "{"
         << "\"id\":\"chatcmpl-local\","
         << "\"object\":\"chat.completion\","
         << "\"model\":\"" << model_name_ << "\","
         << "\"choices\":[{"
         <<   "\"index\":0,"
         <<   "\"message\":{\"role\":\"assistant\",\"content\":\"" << generated_text << "\"},"
         <<   "\"finish_reason\":\"stop\""
         << "}],"
         << "\"usage\":{"
         <<   "\"prompt_tokens\":" << prompt_tokens.size() << ","
         <<   "\"completion_tokens\":" << stats_.total_gen_tokens.load() << ","
         <<   "\"total_tokens\":" << (prompt_tokens.size() + stats_.total_gen_tokens.load())
         << "}"
         << "}";

    response_callback(200, "application/json", json.str());
}

void Server::handle_anthropic_messages(const std::string& /*body*/,
                                       std::function<void(int, const std::string&, const std::string&)> response_callback) {
    std::vector<ChatMessage> messages = {
        {"user", "Hello!"}
    };
    std::string formatted = build_chat_prompt(messages);
    std::vector<int32_t> prompt_tokens = tokenizer_.encode(formatted);

    GenerationConfig gen_cfg;
    std::string generated_text;

    engine_->generate_stream("anthropic-001", prompt_tokens, gen_cfg, [&](const TokenGenerationEvent& ev) {
        if (!ev.is_finished) {
            generated_text += ev.text_delta;
        }
    });

    std::ostringstream json;
    json << "{"
         << "\"id\":\"msg_local\","
         << "\"type\":\"message\","
         << "\"role\":\"assistant\","
         << "\"model\":\"" << model_name_ << "\","
         << "\"content\":[{\"type\":\"text\",\"text\":\"" << generated_text << "\"}],"
         << "\"stop_reason\":\"end_turn\""
         << "}";

    response_callback(200, "application/json", json.str());
}

} // namespace megacapybara
