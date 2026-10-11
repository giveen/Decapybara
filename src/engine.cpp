#include "engine.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace megacapybara {

LoopGuard::LoopGuard(LoopGuardAction action) : action_(action) {}

void LoopGuard::reset() {
    history_.clear();
    triggered_ = false;
}

bool LoopGuard::observe_token(int32_t token_id) {
    history_.push_back(token_id);
    // Simple cycle / repetition detection window
    if (history_.size() >= 16) {
        size_t n = history_.size();
        for (size_t period = 1; period <= 8; ++period) {
            bool matches = true;
            for (size_t i = 0; i < period; ++i) {
                if (history_[n - 1 - i] != history_[n - 1 - i - period]) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                triggered_ = true;
                return true;
            }
        }
    }
    return false;
}

PinnedPool::PinnedPool(size_t pool_bytes) : capacity_bytes_(pool_bytes) {
    // In CUDA runtime, this calls cudaHostAlloc(&pool_base_, capacity_bytes_, cudaHostAllocPortable);
    pool_base_ = ::operator new(pool_bytes, std::nothrow);
}

PinnedPool::~PinnedPool() {
    if (pool_base_) {
        ::operator delete(pool_base_);
    }
}

void* PinnedPool::allocate(size_t /*bytes*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_base_;
}

void PinnedPool::free(void* /*ptr*/) {
    std::lock_guard<std::mutex> lock(mutex_);
}

DiskSlots::DiskSlots(const std::string& cache_dir, size_t max_disk_bytes)
    : cache_dir_(cache_dir), max_disk_bytes_(max_disk_bytes) {}

DiskSlots::~DiskSlots() = default;

bool DiskSlots::spill_slot(uint32_t /*slot_id*/, const void* /*data*/, size_t /*bytes*/) {
    return true;
}

bool DiskSlots::restore_slot(uint32_t /*slot_id*/, void* /*dest*/, size_t /*bytes*/) {
    return true;
}

Engine::Engine(const Settings& settings) : settings_(settings) {}

Engine::~Engine() = default;

bool Engine::initialize() {
    pinned_pool_ = std::make_unique<PinnedPool>(static_cast<size_t>(settings_.ram_tier_gb) * 1024 * 1024 * 1024);
    if (settings_.disk_tier_gb > 0) {
        disk_slots_ = std::make_unique<DiskSlots>("cache", static_cast<size_t>(settings_.disk_tier_gb) * 1024 * 1024 * 1024);
    }
    if (!settings_.model_path.empty()) {
        weights_ = std::make_unique<WeightFile>(settings_.model_path);
        if (!weights_->open()) {
            std::cerr << "Warning: Could not open model: " << settings_.model_path << std::endl;
        }
    }
    initialized_ = true;
    return true;
}

void Engine::generate_stream(const std::string& /*request_id*/,
                             const std::vector<int32_t>& prompt_tokens,
                             const GenerationConfig& config,
                             std::function<void(const TokenGenerationEvent&)> callback) {
    LoopGuard guard(settings_.loop_guard);
    
    // Simulation / stub of model generation loop with speculative verification
    for (uint32_t step = 0; step < std::min(config.max_tokens, 10u); ++step) {
        int32_t next_token = 1000 + step;
        if (guard.observe_token(next_token)) {
            TokenGenerationEvent ev{next_token, "", false, true, "repetition_guard"};
            callback(ev);
            break;
        }
        TokenGenerationEvent ev{next_token, " token_" + std::to_string(step), false, false, ""};
        callback(ev);
    }

    TokenGenerationEvent stop_ev{0, "", false, true, "stop"};
    callback(stop_ev);
}

} // namespace megacapybara
