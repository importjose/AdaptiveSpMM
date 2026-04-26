#pragma once

#include "csr.h"
#include "utils.h"

// Run cuSPARSE SpMM as a benchmark baseline.
//
//   graph      – CSR graph on the host
//   h_B        – host dense matrix B, row-major, size graph.N * F
//   h_C_out    – [output] result copied back to host, size graph.N * F
//   F          – feature dimension
//   num_warmup – number of un-timed warmup runs
//   num_runs   – number of timed runs
//
// Returns a BenchResult with mean/min/max latency and GFLOP/s.
BenchResult spmm_cusparse_benchmark(const CSRGraph& graph,
                                     const float*    h_B,
                                     float*          h_C_out,
                                     int             F,
                                     int             num_warmup = 10,
                                     int             num_runs   = 200);
