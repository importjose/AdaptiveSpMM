#include "spmm_cpu.h"

#include <cstring>

void spmm_cpu(const CSRGraph& A, const float* B, float* C, int F) {
    std::memset(C, 0, (size_t)A.N * F * sizeof(float));

    for (int row = 0; row < A.N; row++) {
        float* c_row = C + (size_t)row * F;

        for (int j = A.row_ptr[row]; j < A.row_ptr[row + 1]; j++) {
            int   col = A.col_idx[j];
            float val = A.values[j];
            const float* b_col = B + (size_t)col * F;

            // Inner loop over feature dimension — keep it simple and let the
            // compiler auto-vectorise.
            for (int f = 0; f < F; f++) {
                c_row[f] += val * b_col[f];
            }
        }
    }
}
