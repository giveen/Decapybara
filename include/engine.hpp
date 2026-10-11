#pragma once

#include "settings.hpp"
#include "weight_file.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <mutex>

namespace megacapybara {

struct GenerationConfig {
    double temperature = 0.7;
    double top_p = 0.9;
    uint32_t max_tokens = 4096;
    std::vector<std::string> stop_sequences;
    double thinking_penalty = 0.0;
};

struct TokenGenerationEvent {
    int32_t token_id;
    std::string text_delta;
    bool is_thinking = false;
    bool is_finished = false;
    std::string finish_reason;
};

class LoopGuard {
public:
    explicit LoopGuard(LoopGuardAction action);
    void reset();
    bool observe_token(int32_t token_id);
    bool is_triggered() const { return triggered_; }
    LoopGuardAction action() const { return action_; }

private:
    LoopGuardAction action_;
    std::vector<int32_t> history_;
    bool triggered_ = false;
};

class PinnedPool {
public:
    explicit PinnedPool(size_t pool_bytes);
    ~PinnedPool();
    void* allocate(size_t bytes);
    void free(void* ptr);

private:
    size_t capacity_bytes_;
    void* pool_base_ = nullptr;
    std::mutex mutex_;
};

class DiskSlots {
public:
    DiskSlots(const std::string& cache_dir, size_t max_disk_bytes);
    ~DiskSlots();
    bool spill_slot(uint32_t slot_id, const void* data, size_t bytes);
    bool restore_slot(uint32_t slot_id, void* dest, size_t bytes);

private:
    std::string cache_dir_;
    size_t max_disk_bytes_;
};

class Speculator {
public:
    virtual ~Speculator() = default;
    virtual std::vector<int32_t> draft_tokens(const std::vector<int32_t>& context, uint32_t draft_len) = 0;
    virtual void update_acceptance(uint32_t accepted_count) = 0;
};

class Engine {
public:
    explicit Engine(const Settings& settings);
    ~Engine();

    bool initialize();
    void generate_stream(const std::string& request_id,
                         const std::vector<int32_t>& prompt_tokens,
                         const GenerationConfig& config,
                         std::function<void(const TokenGenerationEvent&)> callback);

    const Settings& settings() const { return settings_; }

private:
    Settings settings_;
    std::unique_ptr<WeightFile> weights_;
    std::unique_ptr<PinnedPool> pinned_pool_;
    std::unique_ptr<DiskSlots> disk_slots_;
    std::unique_ptr<Speculator> speculator_;
    bool initialized_ = false;
};

} // namespace megacapybara
