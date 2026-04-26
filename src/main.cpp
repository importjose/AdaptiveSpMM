#include "csr.h"
#include "spmm_cpu.h"
#include "utils.h"

#ifdef HAVE_CUDA
#include "spmm_naive.cuh"
#include "spmm_adaptive.cuh"
#include "spmm_cusparse.cuh"
#include "bucket.cuh"
#else
#include "bucket.cuh"   // CPU path is always available
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ─── CLI argument parsing ─────────────────────────────────────────────────────

struct Config {
    std::string graph_file  = "";      // if empty, generate synthetic
    int         N           = 10000;   // nodes (synthetic only)
    double      avg_degree  = 20.0;    // mean degree (synthetic only)
    bool        powerlaw    = true;    // degree distribution
    int         F           = 64;      // feature dimension
    int         num_runs    = 200;     // timed benchmark runs
    int         num_warmup  = 10;      // warmup runs
    int         seed        = 42;
    bool        verbose     = false;
    int         thresh_low  = 8;       // bucket threshold: low  <= thresh_low
    int         thresh_med  = 64;      // bucket threshold: med  <= thresh_med
    bool        csv          = false;   // emit a single CSV result line
    bool        skip_cpu     = false;   // skip CPU baseline (for large graphs)
    bool        skip_uniform = false;   // skip warp-per-node and multi-warp-per-node baselines
};

static void print_usage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("  --graph  <path>   Edge-list file (default: generate synthetic)\n");
    printf("  --N      <int>    Nodes for synthetic graph     (default: 10000)\n");
    printf("  --deg    <float>  Avg degree for synthetic      (default: 20.0)\n");
    printf("  --F      <int>    Feature dimension             (default: 64)\n");
    printf("  --runs   <int>    Benchmark runs                (default: 200)\n");
    printf("  --warmup <int>    Warmup runs                   (default: 10)\n");
    printf("  --seed   <int>    RNG seed                      (default: 42)\n");
    printf("  --uniform         Uniform degree distribution   (default: powerlaw)\n");
    printf("  --verbose         Print detailed output\n");
    printf("  --thresh-low <int> Low bucket upper bound (degree, default: 8)\n");
    printf("  --thresh-med <int> Med bucket upper bound (degree, default: 64)\n");
    printf("  --csv              Emit a single CSV result line (suppresses normal output)\n");
    printf("  --skip-cpu         Skip CPU baseline (useful for large graphs)\n");
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { print_usage(argv[0]); exit(0); }
        else if (a == "--graph"   && i+1 < argc) cfg.graph_file  = argv[++i];
        else if (a == "--N"       && i+1 < argc) cfg.N           = std::atoi(argv[++i]);
        else if (a == "--deg"     && i+1 < argc) cfg.avg_degree  = std::atof(argv[++i]);
        else if (a == "--F"       && i+1 < argc) cfg.F           = std::atoi(argv[++i]);
        else if (a == "--runs"    && i+1 < argc) cfg.num_runs    = std::atoi(argv[++i]);
        else if (a == "--warmup"  && i+1 < argc) cfg.num_warmup  = std::atoi(argv[++i]);
        else if (a == "--seed"    && i+1 < argc) cfg.seed        = std::atoi(argv[++i]);
        else if (a == "--uniform")                   cfg.powerlaw   = false;
        else if (a == "--verbose")                   cfg.verbose    = true;
        else if (a == "--thresh-low" && i+1 < argc)  cfg.thresh_low = std::atoi(argv[++i]);
        else if (a == "--thresh-med" && i+1 < argc)  cfg.thresh_med = std::atoi(argv[++i]);
        else if (a == "--csv")                        cfg.csv          = true;
        else if (a == "--skip-cpu")                   cfg.skip_cpu     = true;
        else if (a == "--skip-uniform")               cfg.skip_uniform = true;
        else { fprintf(stderr, "Unknown arg: %s\n", a.c_str()); print_usage(argv[0]); exit(1); }
    }
    return cfg;
}

// ─── Dense matrix helpers ─────────────────────────────────────────────────────

static void fill_random(float* mat, int size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < size; i++) mat[i] = dist(rng);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (!cfg.csv) printf("=== AdaptiveSpMM Phase 4 Benchmark ===\n\n");

    // ── Load / generate graph ──
    CSRGraph graph;
    if (!cfg.graph_file.empty()) {
        if (!cfg.csv) printf("[Graph] Loading from: %s\n", cfg.graph_file.c_str());
        graph = load_from_edge_list(cfg.graph_file);
    } else {
        if (!cfg.csv) printf("[Graph] Generating synthetic  N=%d  avg_deg=%.1f  dist=%s  seed=%d\n",
               cfg.N, cfg.avg_degree,
               cfg.powerlaw ? "powerlaw" : "uniform", cfg.seed);
        graph = generate_synthetic(cfg.N, cfg.avg_degree, cfg.seed, cfg.powerlaw);
    }
    if (!cfg.csv) { print_graph_stats(graph); printf("\n"); }

    // ── Bucketing / preprocessing ──
    BucketPartition bp = bucket_partition(graph, cfg.thresh_low, cfg.thresh_med);
    if (!cfg.csv) { bucket_print(bp); printf("\n"); }

    const int N = graph.N;
    const int F = cfg.F;
    if (!cfg.csv) printf("[Config] F=%d  warmup=%d  runs=%d\n\n", F, cfg.num_warmup, cfg.num_runs);

    // ── Allocate dense matrices ──
    std::vector<float> B(    (size_t)N * F);
    std::vector<float> C_cpu((size_t)N * F, 0.f);
    fill_random(B.data(), N * F, cfg.seed + 1);

    // FLOP count (shared by all kernel variants)
    long long flops = 2LL * graph.nnz * F;

    // Graph name for CSV output (basename of path, or "synthetic")
    std::string graph_name = cfg.graph_file.empty() ? "synthetic" : cfg.graph_file;
    auto slash = graph_name.rfind('/');
    if (slash != std::string::npos) graph_name = graph_name.substr(slash + 1);

    // ── CPU Baseline ──
    BenchResult r_cpu{};
    if (!cfg.skip_cpu) {
        if (!cfg.csv) printf("[CPU Baseline]\n");
        CPUTimer timer;
        std::vector<float> times(cfg.num_runs);

        // Warmup
        for (int i = 0; i < cfg.num_warmup; i++) {
            std::fill(C_cpu.begin(), C_cpu.end(), 0.f);
            spmm_cpu(graph, B.data(), C_cpu.data(), F);
        }

        for (int i = 0; i < cfg.num_runs; i++) {
            std::fill(C_cpu.begin(), C_cpu.end(), 0.f);
            timer.start();
            spmm_cpu(graph, B.data(), C_cpu.data(), F);
            times[i] = timer.stop_ms();
        }
        r_cpu = BenchResult::compute(times, flops);
        if (!cfg.csv) { r_cpu.print("CPU (CSR)"); printf("\n"); }
    } else {
        // Run CPU once (no timing) to get reference output for correctness checks
        std::fill(C_cpu.begin(), C_cpu.end(), 0.f);
        spmm_cpu(graph, B.data(), C_cpu.data(), F);
    }

    // ── GPU Kernels ──
#ifdef HAVE_CUDA
    if (!cfg.csv) printf("[GPU Kernels]\n");

    // ── Naive kernel (1 thread per row) ──
    std::vector<float> C_naive((size_t)N * F, 0.f);
    BenchResult r_naive = spmm_naive_benchmark(
        graph, B.data(), C_naive.data(), F,
        cfg.num_warmup, cfg.num_runs);
    if (!cfg.csv) r_naive.print("Naive (thread-per-node)");

    // ── cuSPARSE baseline ──
    std::vector<float> C_cusparse((size_t)N * F, 0.f);
    BenchResult r_cusparse = spmm_cusparse_benchmark(
        graph, B.data(), C_cusparse.data(), F,
        cfg.num_warmup, cfg.num_runs);
    if (!cfg.csv) r_cusparse.print("cuSPARSE");

    // ── Warp-per-node baseline (all nodes) ──
    std::vector<float> C_warp((size_t)N * F, 0.f);
    BenchResult r_warp{};
    if (!cfg.skip_uniform) {
        r_warp = spmm_warp_benchmark(
            graph, B.data(), C_warp.data(), F,
            cfg.num_warmup, cfg.num_runs);
        if (!cfg.csv) r_warp.print("Warp-per-node (all)");
    }

    // ── Multi-warp-per-node baseline (all nodes) ──
    std::vector<float> C_multiwarp((size_t)N * F, 0.f);
    BenchResult r_multiwarp{};
    if (!cfg.skip_uniform) {
        r_multiwarp = spmm_multiwarp_benchmark(
            graph, B.data(), C_multiwarp.data(), F,
            cfg.num_warmup, cfg.num_runs);
        if (!cfg.csv) r_multiwarp.print("Multi-warp-per-node (all)");
    }

    // ── 3-bucket adaptive kernel ──
    std::vector<float> C_adapt3((size_t)N * F, 0.f);
    BenchResult r_adapt3 = spmm_adaptive_benchmark(
        graph, bp, B.data(), C_adapt3.data(), F,
        cfg.num_warmup, cfg.num_runs);
    if (!cfg.csv) r_adapt3.print("Adaptive 3-bucket");

    // ── 2-bucket adaptive kernel (HR-SpMM style) ──
    std::vector<float> C_adapt2((size_t)N * F, 0.f);
    BenchResult r_adapt2 = spmm_2bucket_benchmark(
        graph, bp, B.data(), C_adapt2.data(), F,
        cfg.num_warmup, cfg.num_runs);
    if (!cfg.csv) r_adapt2.print("Adaptive 2-bucket");

    // Correctness checks
    if (!cfg.csv) {
        printf("\n[Correctness]\n");
        bool ok_naive     = check_correctness(C_cpu.data(), C_naive.data(),     N * F);
        bool ok_cusparse  = check_correctness(C_cpu.data(), C_cusparse.data(),  N * F);
        bool ok_warp      = check_correctness(C_cpu.data(), C_warp.data(),      N * F);
        bool ok_multiwarp = check_correctness(C_cpu.data(), C_multiwarp.data(), N * F);
        bool ok_adapt3    = check_correctness(C_cpu.data(), C_adapt3.data(),    N * F);
        bool ok_adapt2    = check_correctness(C_cpu.data(), C_adapt2.data(),    N * F);
        printf("  Naive GPU vs CPU:          %s\n", ok_naive     ? "PASS" : "FAIL");
        printf("  cuSPARSE vs CPU:           %s\n", ok_cusparse  ? "PASS" : "FAIL");
        printf("  Warp-per-node (all):       %s\n", ok_warp      ? "PASS" : "FAIL");
        printf("  Multi-warp-per-node (all): %s\n", ok_multiwarp ? "PASS" : "FAIL");
        printf("  Adaptive 3-bucket:         %s\n", ok_adapt3    ? "PASS" : "FAIL");
        printf("  Adaptive 2-bucket:         %s\n", ok_adapt2    ? "PASS" : "FAIL");

        if (cfg.verbose) {
            auto show_mismatches = [&](const char* label, const float* test) {
                int shown = 0;
                for (int i = 0; i < N * F && shown < 10; i++) {
                    if (std::abs(C_cpu[i] - test[i]) > 1e-3f) {
                        printf("    %s [%d] cpu=%.6f  gpu=%.6f  diff=%.6e\n",
                               label, i, C_cpu[i], test[i],
                               std::abs(C_cpu[i] - test[i]));
                        shown++;
                    }
                }
            };
            if (!ok_naive)    show_mismatches("Naive",    C_naive.data());
            if (!ok_cusparse) show_mismatches("cuSPARSE", C_cusparse.data());
            if (!ok_adapt3)   show_mismatches("3-bucket", C_adapt3.data());
            if (!ok_adapt2)   show_mismatches("2-bucket", C_adapt2.data());
        }
    }

    // ── CSV output ──
    if (cfg.csv) {
        // graph,N,nnz,F,thresh_low,thresh_med,cpu_ms,naive_ms,cusparse_ms,warp_ms,multiwarp_ms,adapt3_ms,adapt2_ms,preprocess_ms
        printf("%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
               graph_name.c_str(),
               N, graph.nnz, F,
               cfg.thresh_low, cfg.thresh_med,
               r_cpu.mean_ms, r_naive.mean_ms, r_cusparse.mean_ms,
               r_warp.mean_ms, r_multiwarp.mean_ms,
               r_adapt3.mean_ms, r_adapt2.mean_ms,
               bp.preprocess_ms);
    }

#else
    if (!cfg.csv) {
        printf("[GPU Kernels] Skipped (build with -DHAVE_CUDA)\n");
        printf("[Correctness] Skipped — no GPU build\n");
    }
#endif

    bucket_free_device(bp);

    if (!cfg.csv) printf("\nDone.\n");
    return 0;
}
