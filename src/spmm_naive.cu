#include "spmm_naive.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// ─── Kernel ───────────────────────────────────────────────────────────────────

// One CUDA thread handles one row (node).
// For each neighbor j in the adjacency list, it accumulates val * B[col, :] into C[row, :].
// This is optimal for very low-degree rows; for high-degree rows it serialises thousands
// of memory accesses through a single thread — the load-imbalance this project aims to fix.
__global__ void spmm_naive_kernel(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ B,   // N×F row-major
    float*                    C,   // N×F row-major (pre-zeroed)
    int N, int F)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float* c = C + (size_t)row * F;

    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
        int         col = col_idx[j];
        float       val = values[j];
        const float* b  = B + (size_t)col * F;

        for (int f = 0; f < F; f++) {
            c[f] += val * b[f];
        }
    }
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────

BenchResult spmm_naive_benchmark(const CSRGraph& graph,
                                  const float*    h_B,
                                  float*          h_C_out,
                                  int             F,
                                  int             num_warmup,
                                  int             num_runs)
{
    const int N   = graph.N;
    const int nnz = graph.nnz;

    // ── Allocate device memory ──
    int*   d_row_ptr;
    int*   d_col_idx;
    float* d_values;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,  nnz    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values,   nnz    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,        (size_t)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,        (size_t)N * F * sizeof(float)));

    // ── Copy inputs to device ──
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), (N + 1) * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, graph.col_idx.data(),  nnz    * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,  graph.values.data(),   nnz    * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,       h_B,            (size_t)N * F * sizeof(float), cudaMemcpyHostToDevice));

    // ── Kernel launch config ──
    const int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    // ── Warmup ──
    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, (size_t)N * F * sizeof(float)));
        spmm_naive_kernel<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_B, d_C, N, F);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timed runs ──
    GPUTimer timer;
    std::vector<float> times(num_runs);

    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, (size_t)N * F * sizeof(float)));
        timer.start();
        spmm_naive_kernel<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_B, d_C, N, F);
        times[i] = timer.stop_ms();
    }

    // ── Copy result back (from last run) ──
    CUDA_CHECK(cudaMemcpy(h_C_out, d_C, (size_t)N * F * sizeof(float), cudaMemcpyDeviceToHost));

    // ── Free device memory ──
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // 2 FLOPs per nonzero per feature (one multiply + one add)
    long long flops = 2LL * nnz * F;
    return BenchResult::compute(times, flops);
}
