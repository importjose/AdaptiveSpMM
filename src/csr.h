#pragma once

#include <string>
#include <vector>

// Compressed Sparse Row graph.
// Stores a weighted, directed adjacency matrix in CSR format.
// row_ptr[i]..row_ptr[i+1] are indices into col_idx/values for row i.
struct CSRGraph {
    int N   = 0;   // number of nodes
    int nnz = 0;   // number of nonzeros (edges)

    std::vector<int>   row_ptr;  // size N+1
    std::vector<int>   col_idx;  // size nnz
    std::vector<float> values;   // size nnz (edge weights; 1.0 for unweighted)
};

// Load from a text edge-list file.
// Format:
//   # comment lines are ignored
//   N M                     ← node count, edge count (optional header)
//   src dst [weight]        ← one directed edge per line, 0-indexed nodes
// If no weight column, all edges get weight 1.0.
CSRGraph load_from_edge_list(const std::string& path);

// Generate a synthetic graph.
//   N          – number of nodes
//   avg_degree – mean out-degree
//   seed       – RNG seed for reproducibility
//   powerlaw   – if true, degrees follow a power-law (Zipf) distribution;
//                if false, all nodes get exactly floor(avg_degree) neighbors
CSRGraph generate_synthetic(int N, double avg_degree, int seed,
                             bool powerlaw = true);

void print_graph_stats(const CSRGraph& g);
