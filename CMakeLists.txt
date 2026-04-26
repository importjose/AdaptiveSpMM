cmake_minimum_required(VERSION 3.18)
project(AdaptiveSpMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# A100 target on Delta
set(CMAKE_CUDA_ARCHITECTURES 80)

# Optimisation flags
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)

set(CPU_SOURCES
    src/main.cpp
    src/csr.cpp
    src/spmm_cpu.cpp
    src/bucket.cu
)

set(GPU_SOURCES
    ${CPU_SOURCES}
    src/spmm_naive.cu
)

# ── CPU-only target ───────────────────────────────────────────────────────────
add_executable(spmm_cpu ${CPU_SOURCES})
target_include_directories(spmm_cpu PRIVATE src)

# ── GPU target ────────────────────────────────────────────────────────────────
add_executable(spmm_gpu ${GPU_SOURCES})
target_include_directories(spmm_gpu PRIVATE src)
target_compile_definitions(spmm_gpu PRIVATE HAVE_CUDA)
set_target_properties(spmm_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
