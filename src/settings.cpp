#include "settings.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace megacapybara {

Settings Settings::load_from_args_and_json(int argc, char** argv, const std::string& config_file) {
    Settings s;

    // Check environment variable
    if (const char* env_key = std::getenv("MEGACAPYBARA_API_KEY")) {
        s.api_key = env_key;
    }

    // Command-line flag parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_help();
            std::exit(0);
        } else if (arg == "--host" && i + 1 < argc) {
            s.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            s.port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--api-key" && i + 1 < argc) {
            s.api_key = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            s.model_path = argv[++i];
        } else if (arg == "--drafter" && i + 1 < argc) {
            s.drafter_path = argv[++i];
        } else if (arg == "--no-speculation") {
            s.speculator = SpeculatorType::Off;
        } else if (arg == "--max-context" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val != "auto") s.max_context = std::stoul(val);
        } else if (arg == "--max-concurrency" && i + 1 < argc) {
            s.max_concurrency = std::stoul(argv[++i]);
        } else if (arg == "--kv-paging") {
            s.kv_paging = true;
        } else if (arg == "--no-tui") {
            s.tui_enabled = false;
        } else if (arg == "--no-verify") {
            s.verify_checksums = false;
        }
    }

    return s;
}

void Settings::print_help() {
    std::cout << "megacapybara [options]\n"
              << "  --model <file>                 weight file (.mcapy)\n"
              << "  --drafter <file>               speculative drafter (.mcapy)\n"
              << "  --no-speculation               disable speculation\n"
              << "  --max-context <tokens|auto>    context limit per slot or shared pool\n"
              << "  --kv-paging                    shared dynamic KV cache pool\n"
              << "  --max-concurrency <1-15>       concurrent conversations\n"
              << "  --host <addr>                  listen address (default: 127.0.0.1)\n"
              << "  --port <port>                  port (default: 8080)\n"
              << "  --api-key <key>                require bearer authorization token\n"
              << "  --no-tui                       log text to stdout instead of TUI dashboard\n"
              << "  --no-verify                    skip weight file checksum validation\n";
}

} // namespace megacapybara
