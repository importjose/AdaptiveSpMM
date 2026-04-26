#include "csr.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cstdio>

// ─── load_from_edge_list ──────────────────────────────────────────────────────

CSRGraph load_from_edge_list(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open())
        throw std::runtime_error("Cannot open graph file: " + path);

    // Collect edges first, then build CSR
    struct Edge { int src, dst; float w; };
    std::vector<Edge> edges;
    int N = 0;  // will be inferred from max node ID + 1 unless header sets it

    std::string line;
    bool got_header = false;
    int declared_N = 0, declared_M = 0;

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        // Check if it looks like a header "N M"
        if (!got_header) {
            int a, b; float c;
            std::istringstream probe(line);
            probe >> a >> b;
            if (probe && !(probe >> c)) {  // exactly two integers → header
                declared_N = a;
                declared_M = b;
                got_header = true;
                edges.reserve(declared_M);
                continue;
            }
        }

        int src, dst; float w = 1.0f;
        ss >> src >> dst;
        if (ss.fail()) continue;
        ss >> w;  // optional weight; if absent w stays 1.0
        if (ss.fail()) w = 1.0f;

        edges.push_back({src, dst, w});
        N = std::max(N, std::max(src, dst) + 1);
    }

    if (declared_N > 0) N = std::max(N, declared_N);

    // Sort edges by source for CSR construction
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.src < b.src || (a.src == b.src && a.dst < b.dst);
    });

    // Build CSR
    CSRGraph g;
    g.N   = N;
    g.nnz = (int)edges.size();
    g.row_ptr.assign(N + 1, 0);
    g.col_idx.resize(g.nnz);
    g.values.resize(g.nnz);

    for (const auto& e : edges) g.row_ptr[e.src + 1]++;
    for (int i = 1; i <= N; i++) g.row_ptr[i] += g.row_ptr[i - 1];

    for (int k = 0; k < g.nnz; k++) {
        g.col_idx[k] = edges[k].dst;
        g.values[k]  = edges[k].w;
    }

    return g;
}

// ─── generate_synthetic ───────────────────────────────────────────────────────

CSRGraph generate_synthetic(int N, double avg_degree, int seed, bool powerlaw) {
    std::mt19937 rng(seed);

    // Assign per-node degrees
    std::vector<int> degree(N);
    if (!powerlaw) {
        // Uniform: every node gets exactly floor(avg_degree) neighbors
        int d = std::max(1, (int)avg_degree);
        for (int i = 0; i < N; i++) degree[i] = d;
    } else {
        // Zipf-like: degrees are inversely proportional to rank.
        // d_i = max(1, floor(2*avg_deg * N / (i+1))) adjusted so mean ≈ avg_deg.
        // We assign a Zipf rank to each node, then shuffle.
        std::vector<int> sorted_deg(N);
        // Harmonic normalizer H_N = sum 1/i for i=1..N
        double H = 0.0;
        for (int i = 1; i <= N; i++) H += 1.0 / i;
        double C = avg_degree * N / H;  // so that (1/H)*sum(C/i) = avg_deg*N → sum = avg_deg*N*H/H
        // Actually simpler: d_i = max(1, round(C/i)) for rank i=1..N
        // C chosen so mean ≈ avg_degree
        for (int i = 0; i < N; i++) {
            sorted_deg[i] = std::max(1, (int)std::round(C / (i + 1)));
        }
        // Shuffle so high-degree nodes aren't always node 0
        std::shuffle(sorted_deg.begin(), sorted_deg.end(), rng);
        degree = sorted_deg;
    }

    // Build adjacency: for each node pick `degree[i]` unique random neighbors
    std::vector<std::vector<std::pair<int,float>>> adj(N);
    std::uniform_int_distribution<int> node_dist(0, N - 1);

    for (int i = 0; i < N; i++) {
        int d = std::min(degree[i], N - 1);  // can't have more neighbors than N-1
        std::unordered_set<int> chosen;
        chosen.reserve(d * 2);
        int attempts = 0;
        while ((int)chosen.size() < d && attempts < d * 10) {
            int nb = node_dist(rng);
            if (nb != i) chosen.insert(nb);
            attempts++;
        }
        for (int nb : chosen) {
            adj[i].push_back({nb, 1.0f});
        }
        std::sort(adj[i].begin(), adj[i].end());
    }

    // Build CSR from adj
    CSRGraph g;
    g.N = N;
    g.row_ptr.resize(N + 1, 0);

    for (int i = 0; i < N; i++) g.row_ptr[i + 1] = g.row_ptr[i] + (int)adj[i].size();
    g.nnz = g.row_ptr[N];
    g.col_idx.resize(g.nnz);
    g.values.resize(g.nnz);

    for (int i = 0; i < N; i++) {
        int base = g.row_ptr[i];
        for (int k = 0; k < (int)adj[i].size(); k++) {
            g.col_idx[base + k] = adj[i][k].first;
            g.values[base + k]  = adj[i][k].second;
        }
    }

    return g;
}

// ─── print_graph_stats ────────────────────────────────────────────────────────

void print_graph_stats(const CSRGraph& g) {
    if (g.N == 0) { printf("  (empty graph)\n"); return; }

    int min_deg = g.row_ptr[1] - g.row_ptr[0];
    int max_deg = min_deg;
    long long sum_deg = 0;

    // Collect degrees for median computation
    std::vector<int> degs(g.N);
    for (int i = 0; i < g.N; i++) {
        degs[i] = g.row_ptr[i + 1] - g.row_ptr[i];
        sum_deg += degs[i];
        if (degs[i] < min_deg) min_deg = degs[i];
        if (degs[i] > max_deg) max_deg = degs[i];
    }
    std::sort(degs.begin(), degs.end());
    int median_deg = degs[g.N / 2];
    double avg_deg = (double)sum_deg / g.N;

    printf("  Nodes: %d  |  Edges: %d  |  Avg degree: %.1f\n",
           g.N, g.nnz, avg_deg);
    printf("  Min degree: %d  |  Median: %d  |  Max degree: %d\n",
           min_deg, median_deg, max_deg);
}
