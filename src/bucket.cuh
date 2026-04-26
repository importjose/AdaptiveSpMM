#pragma once

#include "csr.h"
#include "utils.h"

#include <vector>

// ─── BucketPartition ─────────────────────────────────────────────────────────
//
// Partitions graph nodes into three buckets by out-degree:
//   Low  : degree in [0,  thresh_low]
//   Med  : degree in (thresh_low, thresh_med]
//   High : degree in (thresh_med, ∞)
//
// Host index arrays and device copies are both populated so that Phase-3
// kernels can consume d_*_nodes directly without additional setup.

struct BucketPartition {
    // Thresholds (inclusive upper bound for low and med)
    int thresh_low = 8;
    int thresh_med = 64;

    // Host-side node-ID lists
    std::vector<int> low_nodes;
    std::vector<int> med_nodes;
    std::vector<int> high_nodes;

    int n_low  = 0;
    int n_med  = 0;
    int n_high = 0;

#ifdef HAVE_CUDA
    // Device-side node-ID arrays (allocated by bucket_partition, freed by
    // bucket_free_device).  Null if CUDA is not available.
    int* d_low_nodes  = nullptr;
    int* d_med_nodes  = nullptr;
    int* d_high_nodes = nullptr;
#endif

    float preprocess_ms = 0.f;  // wall time for the entire bucketing step
};

// Partition all nodes of `graph` into three buckets.
//
//   graph      – CSR graph (host)
//   thresh_low – inclusive upper-degree bound for the low bucket  (default 8)
//   thresh_med – inclusive upper-degree bound for the med bucket  (default 64)
//
// The function:
//   1. Launches a CUDA kernel (if HAVE_CUDA) to materialise the degree array,
//      then copies it back; otherwise computes degrees on the CPU.
//   2. Classifies each node on the CPU and builds the index arrays.
//   3. Uploads index arrays to device memory (if HAVE_CUDA).
//   4. Records total preprocessing time in BucketPartition::preprocess_ms.
BucketPartition bucket_partition(const CSRGraph& graph,
                                 int thresh_low = 8,
                                 int thresh_med = 64);

// Free device-side arrays allocated by bucket_partition.
// Safe to call even when HAVE_CUDA is not defined (no-op).
void bucket_free_device(BucketPartition& bp);

// Print a one-block summary of the partition.
void bucket_print(const BucketPartition& bp);
