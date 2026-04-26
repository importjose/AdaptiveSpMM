# AdaptiveSpMM — Phase 1: CPU Baseline & Naive CUDA Kernel

## Table of Contents
1. [What this project is actually doing](#1-what-this-project-is-actually-doing)
2. [The core problem we are solving](#2-the-core-problem-we-are-solving)
3. [Key concepts explained simply](#3-key-concepts-explained-simply)
4. [What Phase 1 builds](#4-what-phase-1-builds)
5. [Project file map](#5-project-file-map)
6. [How to build and run locally (macOS, no GPU)](#6-how-to-build-and-run-locally-macos-no-gpu)
7. [How to transfer and run on Delta (the GPU supercomputer)](#7-how-to-run-on-delta-the-gpu-supercomputer)
8. [Understanding the output](#8-understanding-the-output)
9. [What comes next](#9-what-comes-next)

---

## 1. What this project is actually doing

A **Graph Neural Network (GNN)** is a type of AI model that learns from data shaped like a graph — think of a social network where nodes are people and edges are friendships, or a citation network where nodes are papers and edges are citations.

When a GNN makes a prediction (inference), the most time-consuming step is a math operation called **SpMM** — Sparse Matrix times Dense Matrix multiplication. You can think of it as: *"for every node in the graph, collect the feature vectors of all its neighbors and add them together."*

This project builds a custom GPU kernel (a program that runs on the GPU) that does SpMM *faster* than the generic approach by exploiting the structure of real-world graphs.

---

## 2. The core problem we are solving

In real-world graphs, node degrees (the number of neighbors a node has) are extremely unequal. This is called a **power-law distribution**.

Imagine a social network:
- Most people follow ~50 accounts (low-degree nodes)
- A handful of celebrities are followed by millions (high-degree "hub" nodes)

The standard GPU approach assigns **one strategy** to every node. That is like giving the same amount of staff to a small corner store and to a Walmart — most of the staff at Walmart sits idle while the corner store is overwhelmed.

This project splits nodes into **three buckets** and uses the right strategy for each:

| Bucket | Degree range | Strategy | Why |
|--------|-------------|----------|-----|
| Low | 1–8 neighbors | 1 thread handles 1 node | Cheap, no overhead |
| Medium | 9–64 neighbors | 1 warp (32 threads) handles 1 node | 32 threads cooperate efficiently |
| High | 65+ neighbors | Multiple warps handle 1 node | Splits the big work across many threads |

**Phase 1** sets up the foundation: a CPU reference implementation and the first (naive) GPU kernel. Later phases add the smart bucketing.

---

## 3. Key concepts explained simply

### What is a matrix?
A matrix is just a 2D grid of numbers. Like a spreadsheet. If a graph has N nodes each with F features, we store those features in an N×F matrix (N rows, F columns).

### What is a sparse matrix?
A sparse matrix is a matrix where most values are zero — we don't bother storing the zeros. A graph's adjacency matrix is sparse because most nodes are not connected to each other.

### What is CSR format?
CSR (Compressed Sparse Row) is a compact way to store a sparse matrix using three arrays:

```
Example graph: 4 nodes, edges 0→1, 0→2, 1→3, 2→3

row_ptr = [0, 2, 3, 4, 4]
           ↑           ↑
           node 0      node 3 has no outgoing edges

col_idx  = [1, 2, 3, 3]   ← which node each edge points to
values   = [1, 1, 1, 1]   ← edge weights (all 1.0 here)
```

To find all neighbors of node `i`, look at `col_idx[ row_ptr[i] .. row_ptr[i+1] ]`.

### What is a CUDA kernel?
A CUDA kernel is a function that runs on the GPU in parallel. Instead of running once, it runs thousands of times simultaneously — once per thread. Each thread handles a small piece of the problem (like one node in the graph).

### What is a warp?
The GPU does not actually run threads one-by-one. It groups 32 threads together into a unit called a **warp** and runs all 32 at once in lockstep. If those 32 threads are doing the same thing on adjacent memory, that is very efficient. If they are doing different things, some threads sit idle — this is called **thread divergence** and it wastes GPU capacity.

### What are GFLOP/s?
GFLOP/s = Giga Floating-Point Operations per Second. It measures how much useful math the GPU is doing per second. Higher is faster. The A100 GPU on Delta can theoretically do ~312,000 GFLOP/s (312 TFLOP/s) for FP16 — we will measure how close we get.

### What is SM occupancy?
An SM (Streaming Multiprocessor) is one compute unit on the GPU. The A100 has 108 of them. Occupancy is how busy they are — 100% means all threads are always doing useful work. Low occupancy means threads are stalling, usually waiting for memory.

---

## 4. What Phase 1 builds

Phase 1 produces two things:

### 4a. CPU Baseline
A plain C++ implementation of SpMM. It is single-threaded and slow, but **it is always correct**. Every future GPU kernel will be verified by comparing its output to this baseline. If the numbers do not match (within floating-point rounding), we know the GPU kernel has a bug.

### 4b. Naive CUDA Kernel
The simplest possible GPU version: assign one thread to each graph node. The thread loops over all neighbors of that node and accumulates their feature vectors. This is equivalent to the CPU code, just running in parallel across nodes.

It is called "naive" because:
- It ignores degree imbalance entirely
- A node with 10,000 neighbors gets the same single thread as a node with 1 neighbor
- The thread handling the hub node runs 10,000× longer, blocking the GPU

This is the **intermediate baseline** — we measure how much faster the three-bucket approach is compared to this.

---

## 5. Project file map

```
AdaptiveSpMM/
│
├── Makefile                ← how to compile the project
├── CMakeLists.txt          ← alternative build system (CMake)
│
├── src/
│   ├── main.cpp            ← entry point: ties everything together, runs benchmarks
│   ├── csr.h               ← defines the CSRGraph data structure (header = declaration)
│   ├── csr.cpp             ← actual code: loading graphs, generating test graphs
│   ├── spmm_cpu.h          ← declares the CPU SpMM function
│   ├── spmm_cpu.cpp        ← CPU SpMM implementation (the reference/ground truth)
│   ├── spmm_naive.cuh      ← declares the naive GPU SpMM function (.cuh = CUDA header)
│   ├── spmm_naive.cu       ← naive GPU kernel + benchmark harness (.cu = CUDA file)
│   └── utils.h             ← shared helpers: timers, error checking, result printing
│
├── data/
│   └── README.md           ← instructions for downloading real graph datasets on Delta
│
├── scripts/
│   ├── download_data.sh    ← downloads Cora, ogbn-arxiv, ogbn-products on Delta
│   └── run.sbatch          ← SLURM job script (tells Delta's scheduler what to run)
│
└── results/                ← benchmark output files go here
```

### What each source file does in plain English

**`utils.h`**
A toolbox of small helpers:
- `CUDA_CHECK(...)` — wraps every CUDA call; if anything goes wrong it prints the error and stops instead of silently giving wrong answers
- `GPUTimer` — measures GPU kernel time using CUDA events (more accurate than a regular clock because the GPU runs asynchronously)
- `CPUTimer` — measures wall-clock time for the CPU
- `check_correctness(...)` — compares two output matrices element-by-element; used to verify GPU output matches CPU output
- `BenchResult` — stores and prints timing statistics (mean, min, max latency, GFLOP/s)

**`csr.h` / `csr.cpp`**
Everything to do with graphs:
- `CSRGraph` struct — the three arrays (`row_ptr`, `col_idx`, `values`) plus N and nnz
- `load_from_edge_list(path)` — reads a text file of edges and builds a CSRGraph
- `generate_synthetic(N, avg_degree, seed, powerlaw)` — creates a fake graph for testing without needing a real dataset. The `powerlaw=true` mode creates a skewed graph (few hubs, many low-degree nodes) that mimics real social/citation graphs
- `print_graph_stats(g)` — prints N, nnz, min/median/max degree so you can see the degree distribution at a glance

**`spmm_cpu.h` / `spmm_cpu.cpp`**
The CPU reference. Three nested loops:
1. For each row (node)
2. For each nonzero in that row (neighbor)
3. For each feature dimension

This is the ground truth. It is deliberately simple with no tricks.

**`spmm_naive.cuh` / `spmm_naive.cu`**
The first GPU kernel:
- `spmm_naive_kernel` — the actual CUDA kernel: each thread gets one row, loops over neighbors, accumulates features. One thread per node.
- `spmm_naive_benchmark(...)` — the host-side harness: allocates GPU memory, copies data over PCIe, runs warmup iterations, runs 200 timed iterations using CUDA events, copies result back, frees memory.

**`main.cpp`**
The program you actually run. It:
1. Parses command-line arguments (graph path, N, F, etc.)
2. Loads or generates the graph
3. Generates a random dense feature matrix B
4. Runs and times the CPU baseline
5. Runs and times the GPU naive kernel (if built with CUDA)
6. Checks that GPU output matches CPU output
7. Prints a results table

---

## 6. How to build and run locally (macOS, no GPU)

You do not need a GPU to develop. The CUDA code gets written locally, just not compiled or executed until you transfer to Delta.

### Build
```bash
cd AdaptiveSpMM
make cpu          # compiles only the CPU code — no CUDA needed
```

You should see:
```
g++ -O3 -std=c++17 -Wall -Isrc -o spmm_cpu src/main.cpp src/csr.cpp src/spmm_cpu.cpp
Built: spmm_cpu
```

### Run a quick test
```bash
make test_cpu
```

This runs two synthetic benchmarks automatically. Or run manually:

```bash
# Small uniform graph — every node has exactly 5 neighbors
./spmm_cpu --N 500 --deg 5 --uniform --F 32 --runs 10

# Larger power-law graph — degree ranges from 1 to ~1000
./spmm_cpu --N 10000 --deg 20 --F 64 --runs 200
```

### All command-line options
```
--graph  <path>    Load a real graph from an edge-list file
--N      <int>     Number of nodes (for synthetic graphs)     default: 10000
--deg    <float>   Average degree (for synthetic graphs)      default: 20.0
--F      <int>     Feature dimension (columns of matrix B)    default: 64
--runs   <int>     How many timed iterations to average       default: 200
--warmup <int>     Warmup iterations (not timed)              default: 10
--seed   <int>     Random seed (for reproducibility)          default: 42
--uniform          Use uniform degree distribution instead of power-law
--verbose          Print extra debug info on correctness failures
```

---

## 7. How to run on Delta (the GPU supercomputer)

The workflow is: **edit locally → transfer → build on Delta → run on Delta**. You never compile the GPU code on your Mac — it does not have `nvcc`. You write the code here, send it over, and compile it there.

### One-time setup

Open the `Makefile` and set your Delta username at the top:
```makefile
DELTA_USER := your_username   ← change this to your actual username
```

You only need to do this once. Everything after this uses `make` commands.

---

### Step 1 — Transfer your code to Delta

From your local machine, inside the `AdaptiveSpMM/` folder:
```bash
make transfer
```

This uses `rsync` under the hood, which is a smart file sync tool. It only sends files that have changed — so the first transfer might take a few seconds, but every transfer after that (when you only changed one or two files) is nearly instant.

What it skips:
- `data/` edge-list files (too large, you download those directly on Delta)
- compiled binaries (`spmm_cpu`, `spmm_gpu`) — they were compiled for macOS and would not run on Delta anyway

You will be asked for your Delta password (or it will use your SSH key if you have one set up).

---

### Step 2 — Log in to Delta

```bash
ssh your_username@login.delta.ncsa.illinois.edu
```

You are now on a Delta login node. Think of this like the front desk — you can prepare things here but you cannot run GPU jobs directly. GPU jobs go through a scheduler (SLURM).

---

### Step 3 — Load the CUDA compiler

```bash
cd ~/AdaptiveSpMM
module load cuda/12.8
```

Delta has many software tools installed but keeps them off by default to avoid conflicts. `module load` turns on the one you need. Without this, `nvcc` (the NVIDIA CUDA compiler) does not exist and `make gpu` will fail.

---

### Step 4 — Build the GPU binary

```bash
make gpu
```

This compiles all the source files including the `.cu` CUDA files. It targets `sm_80` which is the A100 GPU's instruction set version (compute capability 8.0). A binary compiled for `sm_80` will not work on older GPUs — but Delta's A100s are exactly `sm_80`.

You should see:
```
nvcc -O3 -std=c++17 -arch=sm_80 -Xcompiler -Wall -DHAVE_CUDA -Isrc -o spmm_gpu ...
Built: spmm_gpu
```

---

### Step 5 — Edit the SLURM job script

SLURM is the job scheduler on Delta. You cannot just run programs directly on a GPU node — you have to submit a job request and SLURM runs it when a GPU node is free.

Open `scripts/run.sbatch` and replace the account placeholder:
```bash
#SBATCH --account=YOUR_ACCOUNT   ← replace with your Delta allocation name
```

Your allocation name is the project/account you were given when you got Delta access. You can find it by running `accounts` on the login node.

---

### Step 6 — Submit the job

```bash
mkdir -p results
sbatch scripts/run.sbatch
```

`sbatch` hands your script to SLURM. SLURM will print something like:
```
Submitted batch job 1234567
```

That number is your job ID. Hold onto it.

---

### Step 7 — Check if your job is running

```bash
squeue --me
```

This shows all your jobs and their status:
- `PD` = pending (waiting in queue for a free GPU node)
- `R`  = running
- nothing shown = finished

Jobs on Delta usually start within a few minutes during off-peak hours.

---

### Step 8 — Read the output

Once the job finishes, output is saved automatically to the `results/` folder:
```bash
cat results/phase1_1234567.out    # the program's printed output
cat results/phase1_1234567.err    # any error messages
```

Replace `1234567` with your actual job ID.

---

### Everyday workflow after setup

Once you have done the one-time setup, your normal loop is just three commands:

```bash
# 1. On your Mac — after making code changes:
make transfer

# 2. On Delta — build and submit:
make gpu
sbatch scripts/run.sbatch

# 3. On Delta — check results:
squeue --me
cat results/phase1_<JOBID>.out
```

---

### Step 9 — Download real graph datasets (optional for Phase 1)

The synthetic graphs built into the program are enough for Phase 1. When you are ready to test on real graphs, do this on Delta:
```bash
module load python/3.10
pip install --user ogb
bash scripts/download_data.sh
```

Then run with a real graph:
```bash
./spmm_gpu --graph data/cora.edgelist --F 64 --runs 200
```

---

## 8. Understanding the output

When you run the program you will see something like this:

```
=== AdaptiveSpMM Phase 1 Benchmark ===

[Graph] Generating synthetic  N=10000  avg_deg=20.0  dist=powerlaw  seed=42
  Nodes: 10000  |  Edges: 179960  |  Avg degree: 18.0
  Min degree: 3  |  Median: 5  |  Max degree: 9999

[Config] F=64  warmup=10  runs=200

[CPU Baseline]
  CPU (CSR)                  mean= 14.231 ms  min= 13.980 ms  max= 15.102 ms  3.21 GFLOP/s

[GPU Kernels]
  Naive (thread-per-node)    mean=  0.847 ms  min=  0.831 ms  max=  0.862 ms  27.18 GFLOP/s

[Correctness]
  Naive GPU vs CPU: PASS
```

### What each line means

**Graph stats:**
- `Nodes: 10000` — the graph has 10,000 nodes
- `Edges: 179960` — 179,960 directed edges (nnz in the sparse matrix)
- `Avg degree: 18.0` — on average each node has 18 neighbors
- `Min: 3 / Median: 5 / Max: 9999` — most nodes have very few neighbors but a handful of hub nodes have thousands. This is the power-law skew we want to exploit.

**Timing columns:**
- `mean` — average time per kernel call over 200 runs (what we care about most)
- `min` — fastest single run (best-case, almost no noise)
- `max` — slowest single run (worst-case, OS or memory interference)
- `GFLOP/s` — computed as `(2 × nnz × F) / mean_time_in_seconds / 1e9`
  - The `2×` is because each neighbor-feature pair costs 1 multiply + 1 add = 2 operations
  - Higher GFLOP/s = faster kernel

**Correctness:**
- `PASS` means every element of the GPU output matches the CPU output within a small floating-point tolerance (0.01% relative error). This confirms the GPU kernel is computing the right answer.
- `FAIL` means there is a bug. Run with `--verbose` to see which values differ.

---

## 9. What comes next

Phase 1 gives us:
- A **correct, working CPU reference** we can always trust
- A **naive GPU kernel** that is fast but load-imbalanced
- A **benchmark harness** that works for all future kernels (just plug in a new kernel and call `BenchResult::compute`)

**Phase 2** will add the preprocessing module:
- Scan the `row_ptr` array to compute each node's degree
- Sort nodes into three buckets (low / medium / high) based on configurable thresholds
- Build a `bucket_indices` array for each bucket so each kernel launch only processes its bucket

**Phase 3** will add the two smarter kernels:
- Warp-per-node (32 threads cooperate on one medium-degree node using shared memory)
- Multi-warp-per-node (multiple warps split one hub node and reduce with atomics)

**Phase 4** will run the full benchmark across all datasets and threshold configurations and write up the analysis.
