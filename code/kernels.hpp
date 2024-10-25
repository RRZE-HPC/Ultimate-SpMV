#ifndef KERNELS
#define KERNELS

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <immintrin.h>  // AVX512 and F16C intrinsics
#include <stdint.h>     // For standard integer types (e.g., int32_t)
#include <x86intrin.h>  // For F16C and AVX512 intrinsics

#define RESTRICT				__restrict__

// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

template <typename VT, typename IT>
static void
spmv_warmup_omp_csr(const ST *C, // 1
             const ST *num_rows, // n_chunks
             const IT * RESTRICT row_ptrs, // chunk_ptrs
             const IT * RESTRICT row_lengths, // unused
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             const int *my_rank = NULL)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < *num_rows; ++row) {
        VT sum{};

        // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sum)
        #pragma omp simd reduction(+:sum)
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            sum += values[j] * x[col_idxs[j]];
        }
        y[row] = sum;
    }
}

// SpMV kernel using _Float16 with AVX-512
template <typename VT, typename IT>
static void spmv_avx512_float16(
    // int n, 
    // const uint16_t *row_ptr, 
    // const uint16_t *col_indices, 
    // const _Float16 *values, 
    // const _Float16 *x, 
    // _Float16 *y
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const int * my_rank = NULL
    ) {

    __m512h accum;
    // Loop over each row of the matrix
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < *num_rows; row++) {
        accum = _mm512_setzero_ph(); // Initialize the accumulator to zero

        // Load the current row range from the CSR row_ptr array
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];

        // Loop over the non-zero elements in the current row
        for (int i = row_start; i < row_end; i += 32) {
            // Ensure we do not go out of bounds
            int limit = (i + 32 < row_end) ? 32 : row_end - i;

            // Load values and corresponding x-vector elements
            __m512h vec_val = _mm512_set_ph(
                (limit > 31) ? values[i + 31] : 0,
                (limit > 30) ? values[i + 30] : 0,
                (limit > 29) ? values[i + 29] : 0,
                (limit > 28) ? values[i + 28] : 0,
                (limit > 27) ? values[i + 27] : 0,
                (limit > 26) ? values[i + 26] : 0,
                (limit > 25) ? values[i + 25] : 0,
                (limit > 24) ? values[i + 24] : 0,
                (limit > 23) ? values[i + 23] : 0,
                (limit > 22) ? values[i + 22] : 0,
                (limit > 21) ? values[i + 21] : 0,
                (limit > 20) ? values[i + 20] : 0,
                (limit > 19) ? values[i + 19] : 0,
                (limit > 18) ? values[i + 18] : 0,
                (limit > 17) ? values[i + 17] : 0,
                (limit > 16) ? values[i + 16] : 0,
                (limit > 15) ? values[i + 15] : 0,
                (limit > 14) ? values[i + 14] : 0,
                (limit > 13) ? values[i + 13] : 0,
                (limit > 12) ? values[i + 12] : 0,
                (limit > 11) ? values[i + 11] : 0,
                (limit > 10) ? values[i + 10] : 0,
                (limit > 9) ? values[i + 9] : 0,
                (limit > 8) ? values[i + 8] : 0,
                (limit > 7) ? values[i + 7] : 0,
                (limit > 6) ? values[i + 6] : 0,
                (limit > 5) ? values[i + 5] : 0,
                (limit > 4) ? values[i + 4] : 0,
                (limit > 3) ? values[i + 3] : 0,
                (limit > 2) ? values[i + 2] : 0,
                (limit > 1) ? values[i + 1] : 0,
                (limit > 0) ? values[i] : 0
            );

            __m512h vec_x = _mm512_set_ph(
                (limit > 31) ? x[col_idxs[i + 31]] : 0,
                (limit > 30) ? x[col_idxs[i + 30]] : 0,
                (limit > 29) ? x[col_idxs[i + 29]] : 0,
                (limit > 28) ? x[col_idxs[i + 28]] : 0,
                (limit > 27) ? x[col_idxs[i + 27]] : 0,
                (limit > 26) ? x[col_idxs[i + 26]] : 0,
                (limit > 25) ?  x[col_idxs[i + 25]] : 0,
                (limit > 24) ?  x[col_idxs[i + 24]] : 0,
                (limit > 23) ?  x[col_idxs[i + 23]] : 0,
                (limit > 22) ?  x[col_idxs[i + 22]] : 0,
                (limit > 21) ?  x[col_idxs[i + 21]] : 0,
                (limit > 20) ?  x[col_idxs[i + 20]] : 0,
                (limit > 19) ?  x[col_idxs[i + 19]] : 0,
                (limit > 18) ?  x[col_idxs[i + 18]] : 0,
                (limit > 17) ?  x[col_idxs[i + 17]] : 0,
                (limit > 16) ?  x[col_idxs[i + 16]] : 0,
                (limit > 15) ? x[col_idxs[i + 15]] : 0,
                (limit > 14) ? x[col_idxs[i + 14]] : 0,
                (limit > 13) ? x[col_idxs[i + 13]] : 0,
                (limit > 12) ? x[col_idxs[i + 12]] : 0,
                (limit > 11) ? x[col_idxs[i + 11]] : 0,
                (limit > 10) ? x[col_idxs[i + 10]] : 0,
                (limit > 9) ?  x[col_idxs[i + 9]] : 0,
                (limit > 8) ?  x[col_idxs[i + 8]] : 0,
                (limit > 7) ?  x[col_idxs[i + 7]] : 0,
                (limit > 6) ?  x[col_idxs[i + 6]] : 0,
                (limit > 5) ?  x[col_idxs[i + 5]] : 0,
                (limit > 4) ?  x[col_idxs[i + 4]] : 0,
                (limit > 3) ?  x[col_idxs[i + 3]] : 0,
                (limit > 2) ?  x[col_idxs[i + 2]] : 0,
                (limit > 1) ?  x[col_idxs[i + 1]] : 0,
                (limit > 0) ?  x[col_idxs[i]] : 0
            );

            // Perform the multiply-add for SpMV
            accum = _mm512_fmadd_ph(vec_val, vec_x, accum); // Fused multiply-add
        }

        // Horizontal addition to sum up the 16 elements in the accumulator
        float result = _mm512_reduce_add_ph(accum);

        // Store the result in _Float16 form back into the output vector y
        // y[row] = convert_float_to_float16(result); // Convert the result back to _Float16
        y[row] = result; // Convert the result back to _Float16

    }
}

// SpMV kernel using _Float16 with AVX-512
template <typename VT, typename IT>
static void spmv_avx256_float16(
    // int n, 
    // const uint16_t *row_ptr, 
    // const uint16_t *col_indices, 
    // const _Float16 *values, 
    // const _Float16 *x, 
    // _Float16 *y
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const int * my_rank = NULL
    ) {
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("half_prec_spmv_crs_benchmark");
#endif
        __m256h accum;
        // Loop over each row of the matrix
        #pragma omp for schedule(static)
        for (int row = 0; row < *num_rows; row++) {
            accum = _mm256_setzero_ph(); // Initialize the accumulator to zero

            // Load the current row range from the CSR row_ptr array
            int row_start = row_ptrs[row];
            int row_end = row_ptrs[row + 1];

            // Loop over the non-zero elements in the current row
            for (int i = row_start; i < row_end; i += 16) {
                // Ensure we do not go out of bounds
                int limit = (i + 16 < row_end) ? 16 : row_end - i;

                // Load values and corresponding x-vector elements
                __m256h vec_val = _mm256_set_ph(
                    values[i + 15],
                    values[i + 14],
                    values[i + 13],
                    values[i + 12],
                    values[i + 11],
                    values[i + 10],
                    values[i + 9],
                    values[i + 8],
                    values[i + 7],
                    values[i + 6],
                    values[i + 5],
                    values[i + 4],
                    values[i + 3],
                    values[i + 2],
                    values[i + 1],
                    values[i]
                );

                __m256h vec_x = _mm256_set_ph(
                    x[col_idxs[i + 15]],
                    x[col_idxs[i + 14]],
                    x[col_idxs[i + 13]],
                    x[col_idxs[i + 12]],
                    x[col_idxs[i + 11]],
                    x[col_idxs[i + 10]],
                    x[col_idxs[i + 9]] ,
                    x[col_idxs[i + 8]] ,
                    x[col_idxs[i + 7]] ,
                    x[col_idxs[i + 6]] ,
                    x[col_idxs[i + 5]] ,
                    x[col_idxs[i + 4]] ,
                    x[col_idxs[i + 3]] ,
                    x[col_idxs[i + 2]] ,
                    x[col_idxs[i + 1]] ,
                    x[col_idxs[i]]
                );

                // Perform the multiply-add for SpMV
                accum = _mm256_fmadd_ph(vec_val, vec_x, accum); // Fused multiply-add
            }

            // Horizontal addition to sum up the 16 elements in the accumulator
            float result = _mm256_reduce_add_ph(accum);

            // Store the result in _Float16 form back into the output vector y
            // y[row] = convert_float_to_float16(result); // Convert the result back to _Float16
            y[row] = result; // Convert the result back to _Float16

        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("half_prec_spmv_crs_benchmark");
#endif
    }
}

/**
 * Kernel for CSR format.
 */
template <typename VT, typename IT>
static void
spmv_omp_csr(const ST * C, // 1
             const ST * num_rows, // n_chunks
             const IT * RESTRICT row_ptrs, // chunk_ptrs
             const IT * RESTRICT row_lengths, // unused
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             const int * my_rank = NULL)
{
    #pragma omp parallel 
    {   
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *num_rows; ++row) {
            VT sum{};

            #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sum)
            for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                sum += values[j] * x[col_idxs[j]];
            }
            y[row] = sum;
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_crs_benchmark");
#endif
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
spmv_warmup_omp_scs(const ST *C,
             const ST *n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             const int * my_rank = NULL)
{
    #pragma omp parallel 
    {
        #pragma omp for schedule(static)
        for (ST c = 0; c < *n_chunks; ++c) {
            VT tmp[*C];
            for (ST i = 0; i < *C; ++i) {
                tmp[i] = VT{};
            }

            IT cs = chunk_ptrs[c];

            // TODO: use IT wherever possible
            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *C; ++i) {
                    tmp[i] += values[cs + j * *C + i] * x[col_idxs[cs + j * *C + i]];
                }
            }

            for (ST i = 0; i < *C; ++i) {
                y[c * *C + i] = tmp[i];
            }
        }
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
spmv_omp_scs(const ST * C,
             const ST * n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             const int * my_rank = NULL)
{
    #pragma omp parallel 
    {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("spmv_scs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < *n_chunks; ++c) {
            VT tmp[*C];
            for (ST i = 0; i < *C; ++i) {
                tmp[i] = VT{};
            }

            IT cs = chunk_ptrs[c];

            // TODO: use IT wherever possible
            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *C; ++i) {
                    tmp[i] += values[cs + j * *C + i] * x[col_idxs[cs + j * *C + i]];
                }
            }

            for (ST i = 0; i < *C; ++i) {
                y[c * *C + i] = tmp[i];
            }
        }
#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("spmv_scs_benchmark");
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename VT, typename IT>
static void
scs_impl_cpu(const ST * n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         VT * RESTRICT x,
         VT * RESTRICT y)
{

    #pragma omp parallel
    {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("spmv_scs_adv_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < * n_chunks; ++c) {
            VT tmp[C]{};

            IT cs = chunk_ptrs[c];

            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < C; ++i) {
                    tmp[i] += values[cs + j * C + i] * x[col_idxs[cs + j * C + i]];
                }
            }

            #pragma omp simd
            for (IT i = 0; i < C; ++i) {
                y[c * C + i] = tmp[i];
            }
        }
#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("spmv_scs_adv_benchmark");
#endif
    }
}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename VT, typename IT>
static void
spmv_omp_scs_adv(
             const ST * C,
             const ST * n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             const int * my_rank = NULL)
{
    switch (*C)
    {
        #define INSTANTIATE_CS X(1) X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256)

        #define X(CC) case CC: scs_impl_cpu<CC>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl_cpu<SCS_C>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
        break;
#endif
    default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(C));
        exit(1);
    }
}

/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename IT>
static void
scs_ap_impl_cpu(
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y
)
{

    #pragma omp parallel
    {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("spmv_ap_scs_adv_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < *dp_n_chunks; ++c) {
            double dp_tmp[C]{};
            double sp_tmp[C]{};

            IT dp_cs = dp_chunk_ptrs[c];
            IT sp_cs = sp_chunk_ptrs[c];

            for (IT j = 0; j < dp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < C; ++i) {
                    dp_tmp[i] += dp_values[dp_cs + j * C + i] * dp_x[dp_col_idxs[dp_cs + j * C + i]];
                }
            }

            for (IT j = 0; j < sp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < C; ++i) {
                    // sp_tmp[i] += sp_values[sp_cs + j * C + i] * sp_x[sp_col_idxs[sp_cs + j * C + i]];
                    sp_tmp[i] += sp_values[sp_cs + j * C + i] * dp_x[sp_col_idxs[sp_cs + j * C + i]];
                }
            }

            #pragma omp simd
            for (IT i = 0; i < C; ++i) {
                dp_y[c * C + i] = dp_tmp[i] + sp_tmp[i];
            }
        }
#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("spmv_ap_scs_adv_benchmark");
#endif
    }
}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename IT>
static void
spmv_omp_scs_ap_adv(
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
#ifdef HAVE_HALF_MATH
    const ST * hp_C, // lp_C
    const ST * hp_n_chunks, // lp_n_chunks // TODO same, for now.
    const IT * RESTRICT hp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // lp_chunk_lengths
    const IT * RESTRICT hp_col_idxs, // lp_col_idxs
    const _Float16 * RESTRICT hp_values, // lp_values
    _Float16 * RESTRICT hp_x, // lp_x
    _Float16 * RESTRICT hp_y, // lp_y
#endif
    const int * my_rank = NULL
)
{
    switch (*dp_C)
    {
        #define INSTANTIATE_CS X(1) X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256)

        #define X(CC) case CC: scs_ap_impl_cpu<CC,IT>(dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_ap_impl_cpu<SCS_C, IT>(dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y);
        break;
#endif
    default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(dp_C));
        exit(1);
    }
}

template <typename IT>
static void
spmv_warmup_omp_csr_ap(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const int * my_rank = NULL
    )
{
    // if(my_rank == 1){
    //     std::cout << "in-kernel spmv_kernel->dp_local_x" << std::endl;
    //     for(int i = 0; i < num_rows; ++i){
    //         std::cout << dp_x[i] << std::endl;
    //     }
    // }
            // Each thread will traverse the dp struct, then the sp struct
        // Load balancing depends on sparsity pattern AND data distribution

        #pragma omp parallel for schedule(static)
        for (ST row = 0; row < *dp_n_rows; ++row) {
            double dp_sum{};
            #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
            for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j ,dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
            }
#endif
            }
            

            double sp_sum{};
            #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
            for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row + 1]; ++j) {
                // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                sp_sum += sp_values[j] * dp_x[sp_col_idxs[j]];

#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);

            }
#endif
            }

            dp_y[row] = dp_sum + sp_sum; // implicit conversion to double
            // sp_y[row] = dp_sum + sp_sum; // implicit conversion to float. 
            // ^ Required when performing multiple SpMVs 
            // Assumes dp_sum + sp_sum is in the range of numbers representable by float
        }
}

template <typename IT>
static void
spmv_omp_csr_apdpsp(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
#ifdef HAVE_HALF_MATH
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y, // unused
#endif
    const int * my_rank = NULL
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apdpsp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *dp_n_rows; ++row) {
                double dp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                // #pragma omp simd reduction(+:dp_sum)
                for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                    dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
                }
    #endif
                }

                double sp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd reduction(+:sp_sum)
                for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row + 1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    sp_sum += sp_values[j] * dp_x[sp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);

                }
    #endif
                }

                dp_y[row] = dp_sum + sp_sum; // implicit conversion to VTU?
            }

    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apdpsp_crs_benchmark");
    #endif
        }
}

#ifdef HAVE_HALF_MATH
template <typename IT>
static void
spmv_omp_csr_apdphp(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y, // unused
    const int * my_rank = NULL
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apdphp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *dp_n_rows; ++row) {
                double dp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:dp_sum)
                for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                    dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
                }
    #endif
                }

                double hp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                #pragma omp simd reduction(+:hp_sum)
                for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row + 1]; ++j) {
                    hp_sum += hp_values[j] * hp_x[hp_col_idxs[j]];
                    // hp_sum += hp_values[j] * dp_x[hp_col_idxs[j]]; // Conversion how???

    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);

                }
    #endif
                }

                dp_y[row] = dp_sum + hp_sum;
            }

    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apdphp_crs_benchmark");
    #endif
        }
}

template <typename IT>
static void
spmv_omp_csr_apsphp(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y, // unused
    const int * my_rank = NULL
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apsphp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *sp_n_rows; ++row) {
                double sp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:sp_sum)
                for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row+1]; ++j) {
                    sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
                }
    #endif
                }

                double hp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                #pragma omp simd reduction(+:hp_sum)
                for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row + 1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    hp_sum += hp_values[j] * sp_x[hp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);

                }
    #endif
                }

                sp_y[row] = sp_sum + hp_sum; // implicit conversion to VTU?
            }
    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apsphp_crs_benchmark");
    #endif
        }
}

template <typename IT>
static void
spmv_omp_csr_apdpsphp(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y, // unused
    const int * my_rank = NULL
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apdpsphp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *sp_n_rows; ++row) {

                double dp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:dp_sum)
                for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                    dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
                }
    #endif
                }

                double sp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:sp_sum)
                for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row+1]; ++j) {
                    sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
                }
    #endif
                }

                double hp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                #pragma omp simd reduction(+:hp_sum)
                for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row + 1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    hp_sum += hp_values[j] * sp_x[hp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);

                }
    #endif
                }

                dp_y[row] = dp_sum + sp_sum + hp_sum;
            }

    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apdpsphp_crs_benchmark");
    #endif
        }
}

#endif

template <typename IT>
static void
spmv_omp_csr_ap(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const int * my_rank = NULL
    )
{
    // if(my_rank == 1){
    //     std::cout << "in-kernel spmv_kernel->dp_local_x" << std::endl;
    //     for(int i = 0; i < num_rows; ++i){
    //         std::cout << dp_x[i] << std::endl;
    //     }
    // }
            // Each thread will traverse the dp struct, then the sp struct
        // Load balancing depends on sparsity pattern AND data distribution
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_ap_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *dp_n_rows; ++row) {
                double dp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                    dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j ,dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
                }
    #endif
                }
                

                double sp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row + 1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    sp_sum += sp_values[j] * dp_x[sp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);

                }
    #endif
                }

                dp_y[row] = dp_sum + sp_sum; // implicit conversion to double
                // sp_y[row] = dp_sum + sp_sum; // implicit conversion to float. 
                // ^ Required when performing multiple SpMVs 
                // Assumes dp_sum + sp_sum is in the range of numbers representable by float
            }

    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_ap_crs_benchmark");
    #endif
        }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename IT>
static void
spmv_warmup_omp_scs_ap(
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
#ifdef HAVE_HALF_MATH
    const ST * hp_C, // lp_C
    const ST * hp_n_chunks, // lp_n_chunks // TODO same, for now.
    const IT * RESTRICT hp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // lp_chunk_lengths
    const IT * RESTRICT hp_col_idxs, // lp_col_idxs
    const _Float16 * RESTRICT hp_values, // lp_values
    _Float16 * RESTRICT hp_x, // lp_x
    _Float16 * RESTRICT hp_y, // lp_y
#endif
    const int * my_rank = NULL
)
{
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (ST c = 0; c < *dp_n_chunks; ++c) {
            double dp_tmp[*dp_C];
            double sp_tmp[*dp_C];

            for (ST i = 0; i < *dp_C; ++i) {
                dp_tmp[i] = 0.0;
            }
            for (ST i = 0; i < *dp_C; ++i) {
                sp_tmp[i] = 0.0f;
            }

            IT dp_cs = dp_chunk_ptrs[c];
            IT sp_cs = sp_chunk_ptrs[c];

            for (IT j = 0; j < dp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    dp_tmp[i] += dp_values[dp_cs + j * *dp_C + i] * dp_x[dp_col_idxs[dp_cs + j * *dp_C + i]];
                }
            }
            for (IT j = 0; j < sp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    sp_tmp[i] += sp_values[sp_cs + j * *dp_C + i] * sp_x[sp_col_idxs[sp_cs + j * *dp_C + i]];
                }
            }

            for (IT i = 0; i < *dp_C; ++i) {
                dp_y[c * *dp_C + i] = dp_tmp[i] + sp_tmp[i];
            }
        }
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename IT>
static void
spmv_omp_scs_ap(
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
#ifdef HAVE_HALF_MATH
    const ST * hp_C, // lp_C
    const ST * hp_n_chunks, // lp_n_chunks // TODO same, for now.
    const IT * RESTRICT hp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // lp_chunk_lengths
    const IT * RESTRICT hp_col_idxs, // lp_col_idxs
    const _Float16 * RESTRICT hp_values, // lp_values
    _Float16 * RESTRICT hp_x, // lp_x
    _Float16 * RESTRICT hp_y, // lp_y
#endif
    const int * my_rank = NULL
)
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv_ap_scs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < *dp_n_chunks; ++c) {
            double dp_tmp[*dp_C];
            double sp_tmp[*dp_C];

            for (ST i = 0; i < *dp_C; ++i) {
                dp_tmp[i] = 0.0;
                sp_tmp[i] = 0.0;
            }

            IT dp_cs = dp_chunk_ptrs[c];
            IT sp_cs = sp_chunk_ptrs[c];

            for (IT j = 0; j < dp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    dp_tmp[i] += dp_values[dp_cs + j * *dp_C + i] * dp_x[dp_col_idxs[dp_cs + j * *dp_C + i]];
                }
            }
            for (IT j = 0; j < sp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    sp_tmp[i] += sp_values[sp_cs + j * *dp_C + i] * sp_x[sp_col_idxs[sp_cs + j * *dp_C + i]];
                }
            }

            for (IT i = 0; i < *dp_C; ++i) {
                dp_y[c * *dp_C + i] = dp_tmp[i] + sp_tmp[i];
            }
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_ap_scs_benchmark");
#endif
    }
}

#ifdef __CUDACC__
template <typename VT, typename IT>
__global__ void
spmv_gpu_scs(const ST *C,
         const ST *n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y,
         const int * my_rank = NULL)
{
    long row = threadIdx.x + blockDim.x * blockIdx.x;
    int c   = row / (*C);  // the no. of the chunk
    int idx = row % (*C);  // index inside the chunk

    if (row < *n_chunks * (*C)) {
        VT tmp{};
        int cs = chunk_ptrs[c];

        for (int j = 0; j < chunk_lengths[c]; ++j) {
            tmp += values[cs + j * (*C) + idx] * x[col_idxs[cs + j * (*C) + idx]];
        }

        y[row] = tmp;
    }

}

template <typename VT, typename IT>
void spmv_gpu_scs_launcher(
    const ST * C,
    const ST * n_chunks, // n_chunks
    const IT * RESTRICT chunk_ptrs, // chunk_ptrs
    const IT * RESTRICT chunk_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const ST * n_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_scs<<<*n_blocks, THREADS_PER_BLOCK>>>(
        C, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y
    );
}

template <typename IT>
__global__ void
spmv_gpu_ap_scs(
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const int * my_rank = NULL
)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int c   = row / (*dp_C);  // the no. of the chunk
    int idx = row % (*dp_C);  // index inside the chunk
    int C = *dp_C; 

    if (row < *dp_n_chunks * (*dp_C)) {
        double tmp{};

        int dp_cs = dp_chunk_ptrs[c];
        int sp_cs = sp_chunk_ptrs[c];

        for (int j = 0; j < dp_chunk_lengths[c]; ++j) {
            tmp += dp_values[dp_cs + idx + j * C] * dp_x[dp_col_idxs[dp_cs + idx + j * C]];
        }
        for (int j = 0; j < sp_chunk_lengths[c]; ++j) {
            tmp += sp_values[sp_cs + idx + j * (*sp_C)] * sp_x[sp_col_idxs[sp_cs + idx + j * (*sp_C)]];
        }

        dp_y[row] = tmp;

    }
}

template <typename IT>
void spmv_gpu_ap_scs_launcher(
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const ST * n_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_ap_scs<IT><<<*n_blocks, THREADS_PER_BLOCK>>>(
        dp_C,
        dp_n_chunks,
        dp_chunk_ptrs,
        dp_chunk_lengths,
        dp_col_idxs,
        dp_values,
        dp_x,
        dp_y,
        sp_C,
        sp_n_chunks,
        sp_chunk_ptrs,
        sp_chunk_lengths,
        sp_col_idxs,
        sp_values,
        sp_x,
        sp_y,
        my_rank
    );

    // spmv_gpu_scs<double, int><<<*n_blocks, THREADS_PER_BLOCK>>>(
    //     dp_C, dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y
    // );

    // spmv_gpu_scs<float, int><<<*n_blocks, THREADS_PER_BLOCK>>>(
    //     sp_C, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y
    // );

}

template <typename VT, typename IT>
__global__ void 
spmv_gpu_csr(
    const ST *C, // 1
    const ST *num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    const VT * RESTRICT x,
    VT * RESTRICT y,
    const int * my_rank = NULL)
{
    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int thread_idx = block_offset + thread_idx_in_block;
    VT tmp{};

    // TODO: branch not performant!
    if(thread_idx < *num_rows){
        // One thread per row
        for(int nz_idx = row_ptrs[thread_idx]; nz_idx < row_ptrs[thread_idx+1]; ++nz_idx){
            tmp += values[nz_idx] * x[col_idxs[nz_idx]];
        }

        y[thread_idx] = tmp;
    }
}

template <typename VT, typename IT>
void spmv_gpu_csr_launcher(
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const ST * n_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_csr<<<*n_blocks, THREADS_PER_BLOCK>>>(
        C, num_rows, row_ptrs, row_lengths, col_idxs, values, x, y
    );
}

template <typename IT>
__global__ void 
spmv_gpu_ap_csr(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const int * my_rank = NULL
    )
{   
    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int thread_idx = block_offset + thread_idx_in_block;
    double dp_sum{};
    float sp_sum{};

    if(thread_idx < *dp_n_rows){
        for(int nz_idx = dp_row_ptrs[thread_idx]; nz_idx < dp_row_ptrs[thread_idx+1]; ++nz_idx){
            dp_sum += dp_values[nz_idx] * dp_x[dp_col_idxs[nz_idx]];
        }
        for(int nz_idx = sp_row_ptrs[thread_idx]; nz_idx < sp_row_ptrs[thread_idx+1]; ++nz_idx){
            // sp_sum += sp_values[nz_idx] * sp_x[sp_col_idxs[nz_idx]];
            sp_sum += sp_values[nz_idx] * dp_x[sp_col_idxs[nz_idx]];
        }

        dp_y[thread_idx] = dp_sum + sp_sum;
    }
}

template <typename IT>
void spmv_gpu_ap_csr_launcher(
    const ST * dp_C, // 1
    const ST * dp_n_rows, // TODO: (same for both)
    const IT * RESTRICT dp_row_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_row_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_rows, // TODO: (same for both)
    const IT * RESTRICT sp_row_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_row_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const ST * n_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_ap_csr<IT><<<*n_blocks, THREADS_PER_BLOCK>>>(
        dp_C,  
        dp_n_rows,
        dp_row_ptrs, 
        dp_row_lengths, 
        dp_col_idxs, 
        dp_values, 
        dp_x, 
        dp_y,
        sp_C, 
        sp_n_rows, 
        sp_row_ptrs, 
        sp_row_lengths, 
        sp_col_idxs, 
        sp_values, 
        sp_x, 
        sp_y
    );
}

// Advanced SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.


/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename VT, typename IT>
__device__
static void
scs_impl_gpu(const ST * n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;
    ST c   = row / C;  // the no. of the chunk
    ST idx = row % C;  // index inside the chunk

    if (row < *n_chunks * C) {
        VT tmp{};
        IT cs = chunk_ptrs[c];

        for (ST j = 0; j < chunk_lengths[c]; ++j) {
            tmp += values[cs + j * C + idx] * x[col_idxs[cs + j * C +idx]];
        }

        y[row] = tmp;
    }

}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename VT, typename IT>
__global__
static void
spmv_gpu_scs_adv(
             const ST *C,
             const ST *n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{
    switch (*C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64) X(128)

        #define X(CC) case CC: scs_impl_gpu<CC>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl_gpu<SCS_C>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
        break;
#endif
    default:
        //fprintf(stderr,
        //        "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
        //        long(C));
        // exit(1);
    }
}

template <typename VT, typename IT>
void spmv_gpu_scs_adv_launcher(
    const ST *C,
    const ST *n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    const VT * RESTRICT x,
    VT * RESTRICT y,
    const ST * n_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_scs_adv<<<*n_blocks, THREADS_PER_BLOCK>>>(
        C, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y
    );
}

// Advanced SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.


/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename IT>
__device__
static void
scs_ap_impl_gpu(
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y
)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;
    ST c   = row / C;  // the no. of the chunk
    ST idx = row % C;  // index inside the chunk

    if (row < *dp_n_chunks * C) {
        double tmp{};

        IT dp_cs = dp_chunk_ptrs[c];
        IT sp_cs = sp_chunk_ptrs[c];

        for (ST j = 0; j < dp_chunk_lengths[c]; ++j) {
            tmp += dp_values[dp_cs + j * C + idx] * dp_x[dp_col_idxs[dp_cs + j * C + idx]];
        }
        for (ST j = 0; j < sp_chunk_lengths[c]; ++j) {
            tmp += sp_values[sp_cs + j * C + idx] * dp_x[sp_col_idxs[sp_cs + j * C + idx]];
        }

        dp_y[row] = tmp;
    }
}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename IT>
__global__
static void
spmv_gpu_scs_ap_adv(
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y // unused
)
{
    // printf("Do I get on to the device?\n");
    switch (*dp_C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256)

        #define X(CC) case CC: scs_ap_impl_gpu<CC>(dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_ap_impl_gpu<SCS_C>(dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y);
        break;
#endif
    default:
        //fprintf(stderr,
        //        "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
        //        long(C));
        // exit(1);
    }
}

template <typename IT>
void spmv_gpu_ap_scs_adv_launcher(
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y, // unused
    const ST * n_blocks,
    const int * my_rank = NULL
){
    // printf("Do I get to the launcher? n_blocks = %i\n", *n_blocks);
    spmv_gpu_scs_ap_adv<IT><<<*n_blocks, THREADS_PER_BLOCK>>>(
        dp_C, // 1
        dp_n_chunks, // n_chunks (same for both)
        dp_chunk_ptrs, // dp_chunk_ptrs
        dp_chunk_lengths, // unused
        dp_col_idxs,
        dp_values,
        dp_x,
        dp_y, 
        sp_C, // 1
        sp_n_chunks, // n_chunks (same for both)
        sp_chunk_ptrs, // sp_chunk_ptrs
        sp_chunk_lengths, // unused
        sp_col_idxs,
        sp_values,
        sp_x,
        sp_y // unused
    );
}

#endif

#endif