#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ─── CUDA error checking ──────────────────────────────────────────────────────
#ifdef HAVE_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// GPU timer using CUDA events
struct GPUTimer {
    cudaEvent_t start_, stop_;
    GPUTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GPUTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { cudaEventRecord(start_); }
    float stop_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};
#endif  // HAVE_CUDA

// ─── CPU timer ────────────────────────────────────────────────────────────────
struct CPUTimer {
    using clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<clock> t0_;
    void start() { t0_ = clock::now(); }
    float stop_ms() {
        auto dt = clock::now() - t0_;
        return std::chrono::duration<float, std::milli>(dt).count();
    }
};

// ─── Correctness check ────────────────────────────────────────────────────────
// Returns true if all elements agree within atol + rtol * |ref|.
inline bool check_correctness(const float* ref, const float* test,
                               int size, float atol = 1e-4f, float rtol = 1e-4f) {
    int mismatches = 0;
    float max_err = 0.f;
    for (int i = 0; i < size; i++) {
        float err = std::abs(ref[i] - test[i]);
        float tol = atol + rtol * std::abs(ref[i]);
        if (err > tol) {
            mismatches++;
            if (err > max_err) max_err = err;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "  [FAIL] %d/%d mismatches, max_err=%.6e\n",
                mismatches, size, max_err);
        return false;
    }
    return true;
}

// ─── Benchmark statistics ─────────────────────────────────────────────────────
struct BenchResult {
    float mean_ms, min_ms, max_ms, gflops;

    static BenchResult compute(const std::vector<float>& times_ms,
                               long long flops) {
        BenchResult r{};
        r.min_ms = times_ms[0];
        r.max_ms = times_ms[0];
        float sum = 0.f;
        for (float t : times_ms) {
            sum += t;
            if (t < r.min_ms) r.min_ms = t;
            if (t > r.max_ms) r.max_ms = t;
        }
        r.mean_ms = sum / (float)times_ms.size();
        r.gflops = (float)(flops / 1e9) / (r.mean_ms / 1e3f);
        return r;
    }

    void print(const std::string& label) const {
        printf("  %-24s  mean=%7.3f ms  min=%7.3f ms  max=%7.3f ms  %.2f GFLOP/s\n",
               label.c_str(), mean_ms, min_ms, max_ms, gflops);
    }
};
