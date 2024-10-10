#ifdef _OPENMP
#include <omp.h>
#endif

#define RESTRICT				__restrict__
using ST=long;

#include <immintrin.h>  // AVX512 and F16C intrinsics
#include <stdint.h>     // For standard integer types (e.g., int32_t)
#include <x86intrin.h>  // For F16C and AVX512 intrinsics

using IT = int;
using ST = long;

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

// Use safelen to tell comiler there are safely this many elems/row

    #pragma omp parallel 
    {   
        #pragma omp for schedule(static)
        for (ST row = 0; row < num_rows; ++row) {
            double sum{};

            // #pragma omp simd safelen(8) reduction(+:sum)
            for (int j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                sum += values[j] * x[col_idxs[j]];
            }
            y[row] = sum;
        }
    }
}

// // SpMV kernel using double with AVX-512
// // template <typename double, typename IT>
// void spmv_avx512double(
//     // int n, 
//     // const uint16_t *row_ptr, 
//     // const uint16_t *col_indices, 
//     // const double *values, 
//     // const double *x, 
//     // double *y
//     const ST * C, // 1
//     const ST * num_rows, // n_chunks
//     const IT * RESTRICT row_ptrs, // chunk_ptrs
//     const IT * RESTRICT row_lengths, // unused
//     const IT * RESTRICT col_idxs,
//     const double * RESTRICT values,
//     double * RESTRICT x,
//     double * RESTRICT y,
//     const int * my_rank = NULL
//     ) {

//     __m512h accum;
//     // Loop over each row of the matrix
//     #pragma omp parallel for schedule(static)
//     for (int row = 0; row < *num_rows; row++) {
//         accum = _mm512_setzero_ph(); // Initialize the accumulator to zero

//         // Load the current row range from the CSR row_ptr array
//         int row_start = row_ptrs[row];
//         int row_end = row_ptrs[row + 1];

//         // Loop over the non-zero elements in the current row
//         for (int i = row_start; i < row_end; i += 32) {
//             // Ensure we do not go out of bounds
//             int limit = (i + 32 < row_end) ? 32 : row_end - i;

//             // Load values and corresponding x-vector elements
//             __m512h vec_val = _mm512_set_ph(
//                 (limit > 31) ? values[i + 31] : 0,
//                 (limit > 30) ? values[i + 30] : 0,
//                 (limit > 29) ? values[i + 29] : 0,
//                 (limit > 28) ? values[i + 28] : 0,
//                 (limit > 27) ? values[i + 27] : 0,
//                 (limit > 26) ? values[i + 26] : 0,
//                 (limit > 25) ? values[i + 25] : 0,
//                 (limit > 24) ? values[i + 24] : 0,
//                 (limit > 23) ? values[i + 23] : 0,
//                 (limit > 22) ? values[i + 22] : 0,
//                 (limit > 21) ? values[i + 21] : 0,
//                 (limit > 20) ? values[i + 20] : 0,
//                 (limit > 19) ? values[i + 19] : 0,
//                 (limit > 18) ? values[i + 18] : 0,
//                 (limit > 17) ? values[i + 17] : 0,
//                 (limit > 16) ? values[i + 16] : 0,
//                 (limit > 15) ? values[i + 15] : 0,
//                 (limit > 14) ? values[i + 14] : 0,
//                 (limit > 13) ? values[i + 13] : 0,
//                 (limit > 12) ? values[i + 12] : 0,
//                 (limit > 11) ? values[i + 11] : 0,
//                 (limit > 10) ? values[i + 10] : 0,
//                 (limit > 9) ? values[i + 9] : 0,
//                 (limit > 8) ? values[i + 8] : 0,
//                 (limit > 7) ? values[i + 7] : 0,
//                 (limit > 6) ? values[i + 6] : 0,
//                 (limit > 5) ? values[i + 5] : 0,
//                 (limit > 4) ? values[i + 4] : 0,
//                 (limit > 3) ? values[i + 3] : 0,
//                 (limit > 2) ? values[i + 2] : 0,
//                 (limit > 1) ? values[i + 1] : 0,
//                 (limit > 0) ? values[i] : 0
//             );

//             __m512h vec_x = _mm512_set_ph(
//                 (limit > 31) ? x[col_idxs[i + 31]] : 0,
//                 (limit > 30) ? x[col_idxs[i + 30]] : 0,
//                 (limit > 29) ? x[col_idxs[i + 29]] : 0,
//                 (limit > 28) ? x[col_idxs[i + 28]] : 0,
//                 (limit > 27) ? x[col_idxs[i + 27]] : 0,
//                 (limit > 26) ? x[col_idxs[i + 26]] : 0,
//                 (limit > 25) ?  x[col_idxs[i + 25]] : 0,
//                 (limit > 24) ?  x[col_idxs[i + 24]] : 0,
//                 (limit > 23) ?  x[col_idxs[i + 23]] : 0,
//                 (limit > 22) ?  x[col_idxs[i + 22]] : 0,
//                 (limit > 21) ?  x[col_idxs[i + 21]] : 0,
//                 (limit > 20) ?  x[col_idxs[i + 20]] : 0,
//                 (limit > 19) ?  x[col_idxs[i + 19]] : 0,
//                 (limit > 18) ?  x[col_idxs[i + 18]] : 0,
//                 (limit > 17) ?  x[col_idxs[i + 17]] : 0,
//                 (limit > 16) ?  x[col_idxs[i + 16]] : 0,
//                 (limit > 15) ? x[col_idxs[i + 15]] : 0,
//                 (limit > 14) ? x[col_idxs[i + 14]] : 0,
//                 (limit > 13) ? x[col_idxs[i + 13]] : 0,
//                 (limit > 12) ? x[col_idxs[i + 12]] : 0,
//                 (limit > 11) ? x[col_idxs[i + 11]] : 0,
//                 (limit > 10) ? x[col_idxs[i + 10]] : 0,
//                 (limit > 9) ?  x[col_idxs[i + 9]] : 0,
//                 (limit > 8) ?  x[col_idxs[i + 8]] : 0,
//                 (limit > 7) ?  x[col_idxs[i + 7]] : 0,
//                 (limit > 6) ?  x[col_idxs[i + 6]] : 0,
//                 (limit > 5) ?  x[col_idxs[i + 5]] : 0,
//                 (limit > 4) ?  x[col_idxs[i + 4]] : 0,
//                 (limit > 3) ?  x[col_idxs[i + 3]] : 0,
//                 (limit > 2) ?  x[col_idxs[i + 2]] : 0,
//                 (limit > 1) ?  x[col_idxs[i + 1]] : 0,
//                 (limit > 0) ?  x[col_idxs[i]] : 0
//             );

//             // Perform the multiply-add for SpMV
//             accum = _mm512_fmadd_ph(vec_val, vec_x, accum); // Fused multiply-add
//         }

//         // Horizontal addition to sum up the 16 elements in the accumulator
//         float result = _mm512_reduce_add_ph(accum);

//         // Store the result in double form back into the output vector y
//         // y[row] = convert_float_todouble(result); // Convert the result back to double
//         y[row] = result; // Convert the result back to double

//     }
// }