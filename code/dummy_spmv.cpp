#ifdef _OPENMP
#include <omp.h>
#endif

#define RESTRICT				__restrict__
using ST=long;

// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Kernel for CSR format.
 */
void
spmv_omp_csr(const ST C, // 1
             const ST num_rows, // n_chunks
             const int * RESTRICT row_ptrs, // chunk_ptrs
             const int * RESTRICT row_lengths, // unused
             const int * RESTRICT col_idxs,
             const double * RESTRICT values,
             double * RESTRICT x,
             double * RESTRICT y,
             int my_rank)
{
    #pragma omp parallel 
    {   
        #pragma omp for schedule(static)
        for (ST row = 0; row < num_rows; ++row) {
            double sum{};

            #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sum)
            for (int j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                sum += values[j] * x[col_idxs[j]];
            }
            y[row] = sum;
        }
    }
}