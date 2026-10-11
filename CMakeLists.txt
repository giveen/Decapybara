cmake_minimum_required(VERSION 3.20)
project(Decapybara LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)

# 1. CUDA Kernels Library
add_library(decapybara_kernels STATIC
    cuda/probe.cu
    cuda/checksum.cu
    cuda/act_quant.cu
    cuda/sample.cu
    cuda/moe.cu
    cuda/decode_ops.cu
    cuda/drafter.cu
    cuda/vision.cu
    cuda/mtp.cu
    cuda/deltanet_prefill.cu
    cuda/mx_gemv.cu
    cuda/attn_prefill.cu
    cuda/megakernel.cu
)

set_target_properties(decapybara_kernels PROPERTIES
    CUDA_ARCHITECTURES "all"
)

# 2. Main Decapybara Inference Server
add_executable(decapybara
    src/main.cpp
    src/settings.cpp
    src/engine.cpp
    src/server.cpp
    src/weight_file.cpp
    src/tokenizer.cpp
)

target_link_libraries(decapybara PRIVATE
    decapybara_kernels
    pthread
)

# 3. Companion Launcher
add_executable(DecapybaraLauncher
    launcher/src/main.cpp
    launcher/src/launcher_core.cpp
    launcher/src/launcher_estimate.cpp
    launcher/src/launcher_hub.cpp
)

target_include_directories(DecapybaraLauncher PRIVATE
    launcher/include
)

target_link_libraries(DecapybaraLauncher PRIVATE
    pthread
)
