#include "spmm_cusparse.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <vector>

// ─── cuSPARSE error checking ──────────────────────────────────────────────────

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t _s = (call);                                          \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuSPARSE error at %s:%d — code %d\n",            \
                    __FILE__, __LINE__, (int)_s);                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ─── cuSPARSE benchmark ───────────────────────────────────────────────────────

BenchResult spmm_cusparse_benchmark(const CSRGraph& graph,
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

    // ── Create cuSPARSE handle and descriptors ──
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA,
        N, N, nnz,
        d_row_ptr, d_col_idx, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Row-major dense matrices: leading dimension = number of columns = F
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matB, N, F, F, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC, N, F, F, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    float alpha = 1.0f;
    float beta  = 0.0f;

    // ── Query workspace size and allocate ──
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    if (bufferSize > 0) {
        CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    }

    // ── Warmup ──
    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, (size_t)N * F * sizeof(float)));
        CUSPARSE_CHECK(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timed runs ──
    GPUTimer timer;
    std::vector<float> times(num_runs);

    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, (size_t)N * F * sizeof(float)));
        timer.start();
        CUSPARSE_CHECK(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));
        times[i] = timer.stop_ms();
    }

    // ── Copy result back ──
    CUDA_CHECK(cudaMemcpy(h_C_out, d_C, (size_t)N * F * sizeof(float), cudaMemcpyDeviceToHost));

    // ── Cleanup ──
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    long long flops = 2LL * nnz * F;
    return BenchResult::compute(times, flops);
}
