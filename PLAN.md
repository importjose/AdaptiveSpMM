# AdaptiveSpMM — 10-Day Implementation Plan

Deadline: ~April 27, 2026

---

## Phase 1 — Days 1-2 (Apr 17-18): Scaffold + CPU Baseline + Naive CUDA ✅
- Project structure, Makefile, CMakeLists.txt targeting CUDA 12.x / sm_80
- CSR data structure + graph loader (edge-list format)
- Synthetic graph generator (uniform + power-law)
- CPU SpMM reference implementation
- Correctness checker (CPU vs GPU comparison)
- Naive CUDA kernel (1 thread per row)
- SLURM job script + transfer workflow (`make transfer`)

**Done when:** `make gpu` builds on Delta and output shows `Naive GPU vs CPU: PASS`

---

## Phase 2 — Days 3-4 (Apr 19-20): Preprocessing & Bucketing Module
- CUDA kernel to compute degree of every node from `row_ptr`
- Classify nodes into 3 buckets by configurable thresholds:
  - Low:    degree 1–8
  - Medium: degree 9–64
  - High:   degree 65+
- Build per-bucket index arrays (list of node IDs in each bucket)
- Measure and report preprocessing overhead independently
- Make thresholds a runtime parameter (they are an experimental variable)

**Done when:** given any graph, the program correctly partitions nodes into 3 buckets and prints bucket sizes

---

## Phase 3 — Days 5-7 (Apr 21-23): Three-Bucket Adaptive Kernel
- Implement 3 kernel paths:
  1. Thread-per-node  — low-degree bucket (already done in Phase 1)
  2. Warp-per-node    — medium-degree bucket, 32 threads cooperate via shared memory
  3. Multi-warp-per-node — high-degree bucket, multiple warps split one node, reduce with atomics
- Launch all 3 as separate kernels (one per bucket) using CUDA streams
- Verify correctness of each path independently, then combined
- Also implement a two-bucket variant (HR-SpMM style: threshold at 64) for comparison

**Done when:** adaptive kernel produces `PASS` vs CPU reference on all datasets

---

## Phase 4 — Days 8-9 (Apr 25-26): Benchmarking + Profiling
- Download all datasets on Delta: Cora, ogbn-arxiv, ogbn-products
- Benchmark harness: 200 timed runs, report mean/min/max latency and GFLOP/s
- Run full comparison across all configurations:
  - CPU baseline
  - Naive CUDA (thread-per-node only)
  - cuSPARSE SpMM
  - Two-bucket adaptive (HR-SpMM style)
  - Three-bucket adaptive (this project)
- Threshold sensitivity sweep: vary bucket boundaries, record best per dataset
- Profile with Nsight Compute: L2 cache hit rate, warp efficiency, memory bandwidth, SM occupancy
- Measure preprocessing overhead separately

**Done when:** results table complete for all datasets × all kernels

---

## Phase 5 — Day 10 (Apr 27): Analysis + Report
- Compile results into tables and charts
- Answer the key question: does 3-bucket beat 2-bucket, and when/why?
- Show how degree variance correlates with benefit margin
- Write technical report

---

## Key risks
- Reddit dataset (114M edges) is very large — skip if storage/time is tight
- Multi-warp reduction correctness is tricky — validate thoroughly before moving on
- Delta job queue wait times — submit jobs early, batch experiments
