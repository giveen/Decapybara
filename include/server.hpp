#pragma once

#include "engine.hpp"
#include "tokenizer.hpp"
#include <string>
#include <memory>
#include <functional>
#include <atomic>

namespace megacapybara {

struct Stats {
    std::atomic<uint64_t> total_prompt_tokens{0};
    std::atomic<uint64_t> total_gen_tokens{0};
    std::atomic<uint32_t> active_slots{0};
    std::atomic<double> tokens_per_second{0.0};
};

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    bool stream = false;
    double temperature = 0.7;
    double top_p = 0.9;
    uint32_t max_tokens = 4096;
    std::vector<std::string> stop;
};

class Server {
public:
    Server(const Settings& settings,
           const Tokenizer& tokenizer,
           std::shared_ptr<Engine> engine,
           std::string model_name = "qwen3.8-27b");
    ~Server();

    bool start();
    void stop();

    Stats& stats() { return stats_; }

private:
    Settings settings_;
    Tokenizer tokenizer_;
    std::shared_ptr<Engine> engine_;
    std::string model_name_;
    Stats stats_;
    bool is_running_ = false;

    bool verify_auth(const std::string& auth_header) const;
    void handle_chat_completions(const std::string& body,
                                 std::function<void(int, const std::string&, const std::string&)> response_callback);
    void handle_anthropic_messages(const std::string& body,
                                  std::function<void(int, const std::string&, const std::string&)> response_callback);
    std::string build_chat_prompt(const std::vector<ChatMessage>& messages) const;
};

} // namespace megacapybara
