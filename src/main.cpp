#include "settings.hpp"
#include "engine.hpp"
#include "server.hpp"
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    megacapybara::Settings settings = megacapybara::Settings::load_from_args_and_json(argc, argv);

    std::cout << "MegaCapybara 1.35 Reconstructed Engine\n";
    std::cout << "Configured host: " << settings.host << ":" << settings.port << "\n";
    std::cout << "Context mode: " << (settings.kv_paging ? "pool (paged)" : "fixed slots") << "\n";
    std::cout << "Max context: " << settings.max_context << " tokens\n";
    std::cout << "KV precision: " << settings.kv_mode << "\n";

    auto engine = std::make_shared<megacapybara::Engine>(settings);
    if (!engine->initialize()) {
        std::cerr << "Engine initialization failed\n";
        return 1;
    }

    megacapybara::Tokenizer tokenizer = megacapybara::Tokenizer::from_json("{}");
    megacapybara::Server server(settings, tokenizer, engine, "qwen3.8-27b");
    if (!server.start()) {
        std::cerr << "Server startup failed\n";
        return 1;
    }

    return 0;
}
