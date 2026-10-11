# Decapybara 🦫

**Decapybara** is the clean, open-source C++20 and CUDA reconstruction of the MegaCapybara LLM inference server (targeting Qwen3.8-27B on NVIDIA RTX 5090 Blackwell hardware).

> *"The source code will be published later."* — Well, later is now.

---

## Architecture Overview

Decapybara is structured into three unified subsystems:

1. **Inference Server (`decapybara`)**:
   - Modern C++20 HTTP server supporting dual APIs:
     - **OpenAI Compatible**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
     - **Anthropic Compatible**: `/v1/messages`
   - Multi-tiered hierarchical KV cache (VRAM $\to$ Pinned RAM pool $\to$ Disk slot paging).
   - Exact 144-byte binary container reader for `.mcapy` tensor files.
   - Built-in BPE Tokenizer and ChatML templating engine.
   - Generation loop repetition guard (`LoopGuard`).

2. **Blackwell CUDA Kernel Suite (`decapybara_kernels`)**:
   - `megakernel.cu`: Single-launch persistent decode loop with 216-byte parameter dispatch.
   - `mx_gemv.cu`: Blackwell NVFP4 / MXFP8 block matrix-vector multiplication.
   - `attn_prefill.cu`: FlashAttention prefill with KV dequantization.
   - `deltanet_prefill.cu`: DeltaNet linear recurrent state update loops.
   - `mtp.cu`: Multi-Token Prediction vocabulary slicer and residual gather.
   - `drafter.cu`: DFlash speculative probability tree walk.
   - `moe.cu`: Mixture of Experts router (Top-8 gating and softmax).
   - `decode_ops.cu`: Continuous Rotary Position Embeddings (RoPE) and microscaled FP8 KV preparation.
   - `vision.cu`: Vision transformer patch LayerNorm and 2D spatial grid embeddings.
   - `act_quant.cu`: Dynamic 32-element MXFP8 activation quantizer.
   - `sample.cu`: On-device Philox RNG sampler with presence and frequency penalties.
   - `checksum.cu`: Real-time VRAM bit-flip integrity monitor with warp butterfly shuffles.
   - `probe.cu`: 128-bit cache-streaming memory bandwidth profiler.

3. **Companion Launcher (`DecapybaraLauncher`)**:
   - GDDR7 VRAM and RAM capacity planning estimator.
   - Automatic model discovery and speculative drafter pairing.
   - Preset profiles (`MaxEfficiency`, `Medium`, `MaxQuality`).

---

## Build Instructions

### Prerequisites
- Linux x86_64 (glibc ≥ 2.38)
- C++20 compiler (`g++` ≥ 13 or `clang++` ≥ 17)
- CUDA Toolkit (CUDA 12 or 13 with `nvcc`)
- CMake ≥ 3.20

### Building Everything
```bash
git clone <repo_url>
cd Decapybara
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This builds:
- `bin/decapybara`: Main server
- `bin/DecapybaraLauncher`: Companion launcher
- `libdecapybara_kernels.a`: Static CUDA kernel library

---

## Running

```bash
# Check available options
./build/decapybara --help

# Run launcher for capacity estimation and model launch
./build/DecapybaraLauncher

# Start the server on localhost:8080
./build/decapybara --host 127.0.0.1 --port 8080 --kv-mode 6/6
```

---

## License
MIT
