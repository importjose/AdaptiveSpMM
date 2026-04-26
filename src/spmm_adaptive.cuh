#pragma once

#include "csr.h"
#include "bucket.cuh"
#include "utils.h"

// Run the 3-bucket unified adaptive SpMM kernel and benchmark it.
//
//   graph      – CSR graph on the host
//   bp         – bucket partition (with device arrays already uploaded)
//   h_B        – host dense matrix B, row-major, size graph.N * F
//   h_C_out    – [output] result copied back to host, size graph.N * F
//   F          – feature dimension
//   num_warmup – number of un-timed warmup runs
//   num_runs   – number of timed runs
//
// Returns a BenchResult with mean/min/max latency and GFLOP/s.
BenchResult spmm_adaptive_benchmark(const CSRGraph& graph,
                                     const BucketPartition& bp,
                                     const float*    h_B,
                                     float*          h_C_out,
                                     int             F,
                                     int             num_warmup = 10,
                                     int             num_runs   = 200);

// Run the 2-bucket unified adaptive SpMM kernel (HR-SpMM style) and benchmark it.
// Uses threshold at thresh_med: everything <= thresh_med is thread-per-node,
// everything above is multi-warp-per-node.
BenchResult spmm_2bucket_benchmark(const CSRGraph& graph,
                                    const BucketPartition& bp,
                                    const float*    h_B,
                                    float*          h_C_out,
                                    int             F,
                                    int             num_warmup = 10,
                                    int             num_runs   = 200);

// Warp-per-node baseline: all nodes use the medium-bucket (warp-per-node) path.
BenchResult spmm_warp_benchmark(const CSRGraph& graph,
                                 const float*    h_B,
                                 float*          h_C_out,
                                 int             F,
                                 int             num_warmup = 10,
                                 int             num_runs   = 200);

// Multi-warp-per-node baseline: all nodes use the high-bucket (4-warp-per-node) path.
BenchResult spmm_multiwarp_benchmark(const CSRGraph& graph,
                                      const float*    h_B,
                                      float*          h_C_out,
                                      int             F,
                                      int             num_warmup = 10,
                                      int             num_runs   = 200);
