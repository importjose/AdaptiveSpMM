#include "spmm_adaptive.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// Number of warps cooperating on a single high-degree node
static constexpr int WARPS_PER_NODE = 4;
static constexpr int THREADS_PER_HIGH_NODE = WARPS_PER_NODE * 32;

// ─── Unified adaptive kernel ─────────────────────────────────────────────────
//
// Single kernel handling all three bucket strategies.  Nodes are laid out in a
// concatenated list:  [low_nodes | med_nodes | high_nodes].
//
// Region boundaries are warp-aligned (multiples of 32) so that hardware warps
// never straddle two different execution paths — this is critical for correct
// __shfl_down_sync behaviour.
//
// Thread regions (by global thread ID):
//   [0, med_offset)                              → 1 thread  per node  (low)
//   [med_offset, high_offset)                    → 1 warp    per node  (med)
//   [high_offset, high_offset + n_high*128)      → 4 warps   per node  (high)

__global__ void spmm_adaptive_kernel(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float*                    C,
    const int*   __restrict__ node_list,   // concatenated [low | med | high]
    int n_low,
    int n_med,
    int n_high,
    int med_offset,    // warp-aligned start of medium region
    int high_offset,   // warp-aligned start of high region
    int total_threads,
    int N, int F)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_threads) return;

    // ── Path A: thread-per-node (low degree) ────────────────────────────────
    if (tid < med_offset) {
        if (tid >= n_low) return;  // padding threads in alignment gap

        int node = node_list[tid];
        int start = row_ptr[node];
        int end   = row_ptr[node + 1];
        float* c = C + (size_t)node * F;

        for (int j = start; j < end; j++) {
            int         col = col_idx[j];
            float       val = values[j];
            const float* b  = B + (size_t)col * F;
            for (int f = 0; f < F; f++) {
                c[f] += val * b[f];
            }
        }
        return;
    }

    // ── Path B: warp-per-node (medium degree) ───────────────────────────────
    if (tid < high_offset) {
        int local    = tid - med_offset;
        int node_idx = local / 32;
        int lane     = local % 32;

        if (node_idx >= n_med) return;  // padding threads in alignment gap

        int node  = node_list[n_low + node_idx];
        int start = row_ptr[node];
        int end   = row_ptr[node + 1];

        for (int f = 0; f < F; f++) {
            float sum = 0.f;
            for (int j = start + lane; j < end; j += 32) {
                sum += values[j] * B[(size_t)col_idx[j] * F + f];
            }
            // Warp-level reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            if (lane == 0) {
                C[(size_t)node * F + f] = sum;
            }
        }
        return;
    }

    // ── Path C: multi-warp-per-node (high degree) ───────────────────────────
    {
        int local         = tid - high_offset;
        int node_idx      = local / THREADS_PER_HIGH_NODE;
        int sub_id        = local % THREADS_PER_HIGH_NODE;
        int warp_in_group = sub_id / 32;
        int lane          = sub_id % 32;

        if (node_idx >= n_high) return;

        int node  = node_list[n_low + n_med + node_idx];
        int start = row_ptr[node];
        int end   = row_ptr[node + 1];
        // Each warp handles a strided portion of the neighbor list
        int warp_start = start + warp_in_group * 32 + lane;
        int stride     = THREADS_PER_HIGH_NODE;  // 128

        for (int f = 0; f < F; f++) {
            float sum = 0.f;
            for (int j = warp_start; j < end; j += stride) {
                sum += values[j] * B[(size_t)col_idx[j] * F + f];
            }
            // Intra-warp reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            // Cross-warp reduction via atomicAdd
            if (lane == 0) {
                atomicAdd(&C[(size_t)node * F + f], sum);
            }
        }
    }
}

// ─── Helper: build concatenated node list and upload ─────────────────────────

// Round up to next multiple of 32 (warp size)
static inline int align_warp(int n) { return ((n + 31) / 32) * 32; }

struct AdaptiveDeviceData {
    int* d_row_ptr;
    int* d_col_idx;
    float* d_values;
    float* d_B;
    float* d_C;
    int* d_node_list;
    int total_threads;
    int n_low, n_med, n_high;
    int med_offset;    // warp-aligned boundary between low and med regions
    int high_offset;   // warp-aligned boundary between med and high regions
};

static AdaptiveDeviceData setup_device(const CSRGraph& graph,
                                        const float* h_B,
                                        const std::vector<int>& low_nodes,
                                        const std::vector<int>& med_nodes,
                                        const std::vector<int>& high_nodes,
                                        int F)
{
    AdaptiveDeviceData d{};
    const int N   = graph.N;
    const int nnz = graph.nnz;

    d.n_low  = (int)low_nodes.size();
    d.n_med  = (int)med_nodes.size();
    d.n_high = (int)high_nodes.size();

    // Warp-align region boundaries so hardware warps never straddle paths
    d.med_offset  = align_warp(d.n_low);
    d.high_offset = d.med_offset + align_warp(d.n_med * 32);
    d.total_threads = d.high_offset + d.n_high * THREADS_PER_HIGH_NODE;

    // Build concatenated node list on host
    std::vector<int> node_list;
    node_list.reserve(d.n_low + d.n_med + d.n_high);
    node_list.insert(node_list.end(), low_nodes.begin(), low_nodes.end());
    node_list.insert(node_list.end(), med_nodes.begin(), med_nodes.end());
    node_list.insert(node_list.end(), high_nodes.begin(), high_nodes.end());

    // Allocate and copy CSR
    CUDA_CHECK(cudaMalloc(&d.d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.d_col_idx,  nnz    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.d_values,   nnz    * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d.d_row_ptr, graph.row_ptr.data(), (N + 1) * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_col_idx, graph.col_idx.data(),  nnz    * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_values,  graph.values.data(),   nnz    * sizeof(float), cudaMemcpyHostToDevice));

    // Dense matrices
    CUDA_CHECK(cudaMalloc(&d.d_B, (size_t)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.d_C, (size_t)N * F * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d.d_B, h_B, (size_t)N * F * sizeof(float), cudaMemcpyHostToDevice));

    // Node list
    int list_size = (int)node_list.size();
    CUDA_CHECK(cudaMalloc(&d.d_node_list, list_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d.d_node_list, node_list.data(), list_size * sizeof(int), cudaMemcpyHostToDevice));

    return d;
}

static void free_device(AdaptiveDeviceData& d) {
    CUDA_CHECK(cudaFree(d.d_row_ptr));
    CUDA_CHECK(cudaFree(d.d_col_idx));
    CUDA_CHECK(cudaFree(d.d_values));
    CUDA_CHECK(cudaFree(d.d_B));
    CUDA_CHECK(cudaFree(d.d_C));
    CUDA_CHECK(cudaFree(d.d_node_list));
}

// ─── 3-bucket adaptive benchmark ─────────────────────────────────────────────

BenchResult spmm_adaptive_benchmark(const CSRGraph& graph,
                                     const BucketPartition& bp,
                                     const float*    h_B,
                                     float*          h_C_out,
                                     int             F,
                                     int             num_warmup,
                                     int             num_runs)
{
    const int N = graph.N;
    AdaptiveDeviceData d = setup_device(graph, h_B,
                                         bp.low_nodes, bp.med_nodes, bp.high_nodes, F);

    const int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((d.total_threads + BLOCK - 1) / BLOCK);

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        spmm_adaptive_kernel<<<grid, block>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    GPUTimer timer;
    std::vector<float> times(num_runs);

    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        timer.start();
        spmm_adaptive_kernel<<<grid, block>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
        times[i] = timer.stop_ms();
    }

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_out, d.d_C, (size_t)N * F * sizeof(float), cudaMemcpyDeviceToHost));

    free_device(d);

    long long flops = 2LL * graph.nnz * F;
    return BenchResult::compute(times, flops);
}

// ─── 2-bucket adaptive benchmark ─────────────────────────────────────────────

BenchResult spmm_2bucket_benchmark(const CSRGraph& graph,
                                    const BucketPartition& bp,
                                    const float*    h_B,
                                    float*          h_C_out,
                                    int             F,
                                    int             num_warmup,
                                    int             num_runs)
{
    const int N = graph.N;

    // Merge low + med into a single "low" bucket (thread-per-node for all <= thresh_med)
    std::vector<int> merged_low;
    merged_low.reserve(bp.n_low + bp.n_med);
    merged_low.insert(merged_low.end(), bp.low_nodes.begin(), bp.low_nodes.end());
    merged_low.insert(merged_low.end(), bp.med_nodes.begin(), bp.med_nodes.end());

    // Empty medium bucket
    std::vector<int> empty_med;

    AdaptiveDeviceData d = setup_device(graph, h_B,
                                         merged_low, empty_med, bp.high_nodes, F);

    const int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((d.total_threads + BLOCK - 1) / BLOCK);

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        spmm_adaptive_kernel<<<grid, block>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    GPUTimer timer;
    std::vector<float> times(num_runs);

    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        timer.start();
        spmm_adaptive_kernel<<<grid, block>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
        times[i] = timer.stop_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_C_out, d.d_C, (size_t)N * F * sizeof(float), cudaMemcpyDeviceToHost));

    free_device(d);

    long long flops = 2LL * graph.nnz * F;
    return BenchResult::compute(times, flops);
}

// ─── Warp-per-node baseline (all nodes in medium bucket) ─────────────────────

BenchResult spmm_warp_benchmark(const CSRGraph& graph,
                                 const float*    h_B,
                                 float*          h_C_out,
                                 int             F,
                                 int             num_warmup,
                                 int             num_runs)
{
    const int N = graph.N;
    std::vector<int> all_nodes(N);
    for (int i = 0; i < N; i++) all_nodes[i] = i;
    std::vector<int> empty;

    AdaptiveDeviceData d = setup_device(graph, h_B, empty, all_nodes, empty, F);

    const int BLOCK = 256;
    dim3 grid((d.total_threads + BLOCK - 1) / BLOCK);

    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        spmm_adaptive_kernel<<<grid, BLOCK>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GPUTimer timer;
    std::vector<float> times(num_runs);
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        timer.start();
        spmm_adaptive_kernel<<<grid, BLOCK>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
        times[i] = timer.stop_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_C_out, d.d_C, (size_t)N * F * sizeof(float), cudaMemcpyDeviceToHost));
    free_device(d);
    return BenchResult::compute(times, 2LL * graph.nnz * F);
}

// ─── Multi-warp-per-node baseline (all nodes in high bucket) ─────────────────

BenchResult spmm_multiwarp_benchmark(const CSRGraph& graph,
                                      const float*    h_B,
                                      float*          h_C_out,
                                      int             F,
                                      int             num_warmup,
                                      int             num_runs)
{
    const int N = graph.N;
    std::vector<int> all_nodes(N);
    for (int i = 0; i < N; i++) all_nodes[i] = i;
    std::vector<int> empty;

    AdaptiveDeviceData d = setup_device(graph, h_B, empty, empty, all_nodes, F);

    const int BLOCK = 256;
    dim3 grid((d.total_threads + BLOCK - 1) / BLOCK);

    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        spmm_adaptive_kernel<<<grid, BLOCK>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GPUTimer timer;
    std::vector<float> times(num_runs);
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d.d_C, 0, (size_t)N * F * sizeof(float)));
        timer.start();
        spmm_adaptive_kernel<<<grid, BLOCK>>>(
            d.d_row_ptr, d.d_col_idx, d.d_values, d.d_B, d.d_C,
            d.d_node_list, d.n_low, d.n_med, d.n_high,
            d.med_offset, d.high_offset, d.total_threads, N, F);
        times[i] = timer.stop_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_C_out, d.d_C, (size_t)N * F * sizeof(float), cudaMemcpyDeviceToHost));
    free_device(d);
    return BenchResult::compute(times, 2LL * graph.nnz * F);
}
