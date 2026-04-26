#include "bucket.cuh"

#include <cstdio>
#include <cstring>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

// ─── CUDA degree kernel ───────────────────────────────────────────────────────
// Each thread computes the degree of one node from row_ptr.
// degree[i] = row_ptr[i+1] - row_ptr[i]
__global__ void compute_degrees_kernel(const int* __restrict__ row_ptr,
                                       int*       __restrict__ degree,
                                       int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) degree[i] = row_ptr[i + 1] - row_ptr[i];
}

#endif  // HAVE_CUDA

// ─── bucket_partition ─────────────────────────────────────────────────────────

BucketPartition bucket_partition(const CSRGraph& graph,
                                 int thresh_low,
                                 int thresh_med)
{
    BucketPartition bp;
    bp.thresh_low = thresh_low;
    bp.thresh_med = thresh_med;

    const int N = graph.N;

#ifdef HAVE_CUDA
    // ── Time the entire preprocessing step with a GPU event pair ──
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    // ── Upload row_ptr to device ──
    int* d_row_ptr = nullptr;
    int* d_degree  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_degree,   N       * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(),
                          (N + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // ── Launch degree kernel ──
    const int BLOCK = 256;
    compute_degrees_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_row_ptr, d_degree, N);
    CUDA_CHECK(cudaGetLastError());

    // ── Copy degrees back to host ──
    std::vector<int> degree(N);
    CUDA_CHECK(cudaMemcpy(degree.data(), d_degree,
                          N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_degree));

#else
    // ── CPU fallback: derive degrees directly from row_ptr ──
    CPUTimer cpu_timer;
    cpu_timer.start();

    std::vector<int> degree(N);
    for (int i = 0; i < N; i++)
        degree[i] = graph.row_ptr[i + 1] - graph.row_ptr[i];
#endif

    // ── Classify nodes into buckets ──
    bp.low_nodes.reserve(N);
    bp.med_nodes.reserve(N / 4);
    bp.high_nodes.reserve(N / 16);

    for (int i = 0; i < N; i++) {
        int d = degree[i];
        if      (d <= thresh_low) bp.low_nodes.push_back(i);
        else if (d <= thresh_med) bp.med_nodes.push_back(i);
        else                      bp.high_nodes.push_back(i);
    }

    bp.n_low  = (int)bp.low_nodes.size();
    bp.n_med  = (int)bp.med_nodes.size();
    bp.n_high = (int)bp.high_nodes.size();

#ifdef HAVE_CUDA
    // ── Upload index arrays to device ──
    if (bp.n_low > 0) {
        CUDA_CHECK(cudaMalloc(&bp.d_low_nodes,  bp.n_low  * sizeof(int)));
        CUDA_CHECK(cudaMemcpy( bp.d_low_nodes,  bp.low_nodes.data(),
                               bp.n_low  * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (bp.n_med > 0) {
        CUDA_CHECK(cudaMalloc(&bp.d_med_nodes,  bp.n_med  * sizeof(int)));
        CUDA_CHECK(cudaMemcpy( bp.d_med_nodes,  bp.med_nodes.data(),
                               bp.n_med  * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (bp.n_high > 0) {
        CUDA_CHECK(cudaMalloc(&bp.d_high_nodes, bp.n_high * sizeof(int)));
        CUDA_CHECK(cudaMemcpy( bp.d_high_nodes, bp.high_nodes.data(),
                               bp.n_high * sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&bp.preprocess_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

#else
    bp.preprocess_ms = cpu_timer.stop_ms();
#endif

    return bp;
}

// ─── bucket_free_device ───────────────────────────────────────────────────────

void bucket_free_device(BucketPartition& bp)
{
#ifdef HAVE_CUDA
    if (bp.d_low_nodes)  { cudaFree(bp.d_low_nodes);  bp.d_low_nodes  = nullptr; }
    if (bp.d_med_nodes)  { cudaFree(bp.d_med_nodes);  bp.d_med_nodes  = nullptr; }
    if (bp.d_high_nodes) { cudaFree(bp.d_high_nodes); bp.d_high_nodes = nullptr; }
#endif
    (void)bp;
}

// ─── bucket_print ─────────────────────────────────────────────────────────────

void bucket_print(const BucketPartition& bp)
{
    int total = bp.n_low + bp.n_med + bp.n_high;
    auto pct = [&](int n) { return total > 0 ? 100.0 * n / total : 0.0; };

    printf("[Bucketing]  thresh_low=%d  thresh_med=%d\n",
           bp.thresh_low, bp.thresh_med);
    printf("  Low  (deg <= %3d) : %7d nodes  (%.1f%%)\n",
           bp.thresh_low, bp.n_low,  pct(bp.n_low));
    printf("  Med  (deg <= %3d) : %7d nodes  (%.1f%%)\n",
           bp.thresh_med, bp.n_med,  pct(bp.n_med));
    printf("  High (deg >  %3d) : %7d nodes  (%.1f%%)\n",
           bp.thresh_med, bp.n_high, pct(bp.n_high));
    printf("  Preprocessing     : %.3f ms\n", bp.preprocess_ms);
}
