#pragma once

#include "csr.h"

// Compute C = A * B  (SpMM) on the CPU.
//
//   A: sparse N×N matrix in CSR format
//   B: dense N×F matrix, row-major  (B[i*F + f] = feature f of node i)
//   C: dense N×F output, row-major  (must be zeroed before call)
//   F: feature dimension
void spmm_cpu(const CSRGraph& A, const float* B, float* C, int F);
