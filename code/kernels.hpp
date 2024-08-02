#ifndef KERNELS
#define KERNELS

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

        #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sum)
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            sum += values[j] * x[col_idxs[j]];
        }
        y[row] = sum;
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

// TODO: known to be wrong
/**
 * Kernel for ELL format, data structures use row major (RM) layout.
 */
template <typename VT, typename IT>
static void
spmv_omp_ell_rm(
    const ST *C, // unused
    const ST *num_rows,
    const IT * RESTRICT chunk_ptrs, // unused
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const int * my_rank = NULL)
{
    ST nelems_per_row = chunk_lengths[1] - chunk_lengths[0];
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < *num_rows; row++) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; i++) {
            VT val = values[  row * nelems_per_row + i];
            IT col = col_idxs[row * nelems_per_row + i];

            sum += val * x[col];
        }
        y[row] = sum;
    }
}

// TODO: known to be wrong
/**
 * Kernel for ELL format, data structures use column major (CM) layout.
 */
template <typename VT, typename IT>
static void
spmv_omp_ell_cm(
    const ST *C, // unused
    const ST *num_rows,
    const IT * RESTRICT chunk_ptrs, // unused
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const int * my_rank = NULL)
{
    ST nelems_per_row = chunk_lengths[1] - chunk_lengths[0];
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < *num_rows; row++) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; i++) {
            VT val = values[row + i * *num_rows];
            IT col = col_idxs[row + i * *num_rows];

            sum += val * x[col];
        }
        y[row] = sum;
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
template <ST C, typename VT, typename IT>
static void
scs_mp_impl_cpu(
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y
)
{

    #pragma omp parallel
    {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("spmv_ap_scs_adv_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < *hp_n_chunks; ++c) {
            VT hp_tmp[C]{};
            VT lp_tmp[C]{};

            IT hp_cs = hp_chunk_ptrs[c];
            IT lp_cs = lp_chunk_ptrs[c];

            for (IT j = 0; j < hp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < C; ++i) {
                    hp_tmp[i] += hp_values[hp_cs + j * C + i] * hp_x[hp_col_idxs[hp_cs + j * C + i]];
                }
            }

            for (IT j = 0; j < lp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < C; ++i) {
                    lp_tmp[i] += lp_values[lp_cs + j * C + i] * lp_x[lp_col_idxs[lp_cs + j * C + i]];
                    // lp_tmp[i] += lp_values[lp_cs + j * C + i] * hp_x[lp_col_idxs[lp_cs + j * C + i]];
                }
            }

            #pragma omp simd
            for (IT i = 0; i < C; ++i) {
                hp_y[c * C + i] = hp_tmp[i] + lp_tmp[i];
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
template <typename VT, typename IT>
static void
spmv_omp_scs_mp_adv(
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const int * my_rank
)
{
    switch (*hp_C)
    {
        #define INSTANTIATE_CS X(1) X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256)

        #define X(CC) case CC: scs_mp_impl_cpu<CC>(hp_n_chunks, hp_chunk_ptrs, hp_chunk_lengths, hp_col_idxs, hp_values, hp_x, hp_y, lp_n_chunks, lp_chunk_ptrs, lp_chunk_lengths, lp_col_idxs, lp_values, lp_x, lp_y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_mp_impl_cpu<SCS_C>(hp_n_chunks, hp_chunk_ptrs, hp_chunk_lengths, hp_col_idxs, hp_values, hp_x, hp_y, lp_n_chunks, lp_chunk_ptrs, lp_chunk_lengths, lp_col_idxs, lp_values, lp_x, lp_y); break;
#endif
    default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(hp_C));
        exit(1);
    }
}

template <typename IT>
static void
spmv_warmup_omp_csr_mp(
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_rows, // TODO: (same for both)
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_row_lengths, // unused for now
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const int * my_rank = NULL
    )
{
    // if(my_rank == 1){
    //     std::cout << "in-kernel spmv_kernel->hp_local_x" << std::endl;
    //     for(int i = 0; i < num_rows; ++i){
    //         std::cout << hp_x[i] << std::endl;
    //     }
    // }
            // Each thread will traverse the hp struct, then the lp struct
        // Load balancing depends on sparsity pattern AND data distribution

        #pragma omp parallel for schedule(static)
        for (ST row = 0; row < *hp_n_rows; ++row) {
            double hp_sum{};
            #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:hp_sum)
            for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row+1]; ++j) {
                hp_sum += hp_values[j] * hp_x[hp_col_idxs[j]];
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                printf("j = %i, hp_col_idxs[j] = %i, hp_x[hp_col_idxs[j]] = %f\n", j ,hp_col_idxs[j], hp_x[hp_col_idxs[j]]);
                printf("hp_sum += %f * %f\n", hp_values[j], hp_x[hp_col_idxs[j]]);
            }
#endif
            }
            

            double lp_sum{};
            #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:lp_sum)
            for (IT j = lp_row_ptrs[row]; j < lp_row_ptrs[row + 1]; ++j) {
                // lp_sum += lp_values[j] * lp_x[lp_col_idxs[j]];
                lp_sum += lp_values[j] * hp_x[lp_col_idxs[j]];

#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                printf("j = %i, lp_col_idxs[j] = %i, lp_x[lp_col_idxs[j]] = %f\n", j, lp_col_idxs[j], lp_x[lp_col_idxs[j]]);
                printf("lp_sum += %5.16f * %f\n", lp_values[j], lp_x[lp_col_idxs[j]]);

            }
#endif
            }

            hp_y[row] = hp_sum + lp_sum; // implicit conversion to double
            // lp_y[row] = hp_sum + lp_sum; // implicit conversion to float. 
            // ^ Required when performing multiple SpMVs 
            // Assumes hp_sum + lp_sum is in the range of numbers representable by float
        }
}

template <typename IT>
static void
spmv_omp_csr_mp(
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_rows, // TODO: (same for both)
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_row_lengths, // unused for now
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const int * my_rank
    )
{
    // if(my_rank == 1){
    //     std::cout << "in-kernel spmv_kernel->hp_local_x" << std::endl;
    //     for(int i = 0; i < num_rows; ++i){
    //         std::cout << hp_x[i] << std::endl;
    //     }
    // }
            // Each thread will traverse the hp struct, then the lp struct
        // Load balancing depends on sparsity pattern AND data distribution
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_ap_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *hp_n_rows; ++row) {
                double hp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:hp_sum)
                for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row+1]; ++j) {
                    hp_sum += hp_values[j] * hp_x[hp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, hp_col_idxs[j] = %i, hp_x[hp_col_idxs[j]] = %f\n", j ,hp_col_idxs[j], hp_x[hp_col_idxs[j]]);
                    printf("hp_sum += %f * %f\n", hp_values[j], hp_x[hp_col_idxs[j]]);
                }
    #endif
                }
                

                double lp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:lp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:lp_sum)
                for (IT j = lp_row_ptrs[row]; j < lp_row_ptrs[row + 1]; ++j) {
                    // lp_sum += lp_values[j] * lp_x[lp_col_idxs[j]];
                    lp_sum += lp_values[j] * hp_x[lp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("j = %i, lp_col_idxs[j] = %i, lp_x[lp_col_idxs[j]] = %f\n", j, lp_col_idxs[j], lp_x[lp_col_idxs[j]]);
                    printf("lp_sum += %5.16f * %f\n", lp_values[j], lp_x[lp_col_idxs[j]]);

                }
    #endif
                }

                hp_y[row] = hp_sum + lp_sum; // implicit conversion to double
                // lp_y[row] = hp_sum + lp_sum; // implicit conversion to float. 
                // ^ Required when performing multiple SpMVs 
                // Assumes hp_sum + lp_sum is in the range of numbers representable by float
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
spmv_warmup_omp_scs_mp(
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const int * my_rank
)
{
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (ST c = 0; c < *hp_n_chunks; ++c) {
            double hp_tmp[*hp_C];
            double lp_tmp[*hp_C];

            for (ST i = 0; i < *hp_C; ++i) {
                hp_tmp[i] = 0.0;
            }
            for (ST i = 0; i < *hp_C; ++i) {
                lp_tmp[i] = 0.0f;
            }

            IT hp_cs = hp_chunk_ptrs[c];
            IT lp_cs = lp_chunk_ptrs[c];

            for (IT j = 0; j < hp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *hp_C; ++i) {
                    hp_tmp[i] += hp_values[hp_cs + j * *hp_C + i] * hp_x[hp_col_idxs[hp_cs + j * *hp_C + i]];
                }
            }
            for (IT j = 0; j < lp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *hp_C; ++i) {
                    lp_tmp[i] += lp_values[lp_cs + j * *hp_C + i] * lp_x[lp_col_idxs[lp_cs + j * *hp_C + i]];
                }
            }

            for (IT i = 0; i < *hp_C; ++i) {
                hp_y[c * *hp_C + i] = hp_tmp[i] + lp_tmp[i];
            }
        }
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename IT>
static void
spmv_omp_scs_mp(
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const int * my_rank
)
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv_ap_scs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < *hp_n_chunks; ++c) {
            double hp_tmp[*hp_C];
            double lp_tmp[*hp_C];

            for (ST i = 0; i < *hp_C; ++i) {
                hp_tmp[i] = 0.0;
                lp_tmp[i] = 0.0;
            }

            IT hp_cs = hp_chunk_ptrs[c];
            IT lp_cs = lp_chunk_ptrs[c];

            for (IT j = 0; j < hp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *hp_C; ++i) {
                    hp_tmp[i] += hp_values[hp_cs + j * *hp_C + i] * hp_x[hp_col_idxs[hp_cs + j * *hp_C + i]];
                }
            }
            for (IT j = 0; j < lp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *hp_C; ++i) {
                    lp_tmp[i] += lp_values[lp_cs + j * *hp_C + i] * lp_x[lp_col_idxs[lp_cs + j * *hp_C + i]];
                }
            }

            for (IT i = 0; i < *hp_C; ++i) {
                hp_y[c * *hp_C + i] = hp_tmp[i] + lp_tmp[i];
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
    spmv_gpu_scs<<<*n_blocks, THREADS_PER_BLOCK>>>(
        C, num_rows, row_ptrs, row_lengths, col_idxs, values, x, y
    );
}

template <typename IT>
__global__ void
spmv_gpu_mp_scs(
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const int * my_rank
)
{
    long row = threadIdx.x + blockDim.x * blockIdx.x;
    int c   = row / (*hp_C);  // the no. of the chunk
    int idx = row % (*hp_C);  // index inside the chunk

    if (row < *hp_n_chunks * (*hp_C)) {
        double hp_tmp{};
        double lp_tmp{};

        int hp_cs = hp_chunk_ptrs[c];
        int lp_cs = lp_chunk_ptrs[c];

        for (int j = 0; j < hp_chunk_lengths[c]; ++j) {
            hp_tmp += hp_values[hp_cs + j * (*hp_C) + idx] * hp_x[hp_col_idxs[hp_cs + j * (*hp_C) + idx]];
        }
        for (int j = 0; j < lp_chunk_lengths[c]; ++j) {
            lp_tmp += lp_values[lp_cs + j * (*lp_C) + idx] * lp_x[lp_col_idxs[lp_cs + j * (*lp_C) + idx]];
            // lp_tmp += lp_values[lp_cs + j * (*lp_C) + idx] * hp_x[lp_col_idxs[lp_cs + j * (*lp_C) + idx]];
        }

        hp_y[row] = hp_tmp + lp_tmp;
    }
}

template <typename IT>
void spmv_gpu_mp_scs_launcher(
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const ST * n_blocks,
    const int * my_rank
){
    spmv_gpu_mp_scs<IT><<<*n_blocks, THREADS_PER_BLOCK>>>(
        hp_C,
        hp_n_chunks,
        hp_chunk_ptrs,
        hp_chunk_lengths,
        hp_col_idxs,
        hp_values,
        hp_x,
        hp_y,
        lp_C,
        lp_n_chunks,
        lp_chunk_ptrs,
        lp_chunk_lengths,
        lp_col_idxs,
        lp_values,
        lp_x,
        lp_y,
        my_rank
    );
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
spmv_gpu_mp_csr(
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_rows, // TODO: (same for both)
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_row_lengths, // unused for now
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const int * my_rank = NULL
    )
{   
    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int thread_idx = block_offset + thread_idx_in_block;
    double hp_sum{};
    float lp_sum{};

    if(thread_idx < *hp_n_rows){
        for(int nz_idx = hp_row_ptrs[thread_idx]; nz_idx < hp_row_ptrs[thread_idx+1]; ++nz_idx){
            hp_sum += hp_values[nz_idx] * hp_x[hp_col_idxs[nz_idx]];
        }
        for(int nz_idx = lp_row_ptrs[thread_idx]; nz_idx < lp_row_ptrs[thread_idx+1]; ++nz_idx){
            // lp_sum += lp_values[nz_idx] * lp_x[lp_col_idxs[nz_idx]];
            lp_sum += lp_values[nz_idx] * hp_x[lp_col_idxs[nz_idx]];
        }

        hp_y[thread_idx] = hp_sum + lp_sum;
    }
}

template <typename IT>
void spmv_gpu_mp_csr_launcher(
    const ST * hp_C, // 1
    const ST * hp_n_rows, // TODO: (same for both)
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_rows, // TODO: (same for both)
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_row_lengths, // unused for now
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const ST * n_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_mp_csr<IT><<<*n_blocks, THREADS_PER_BLOCK>>>(
        hp_C,  
        hp_n_rows,
        hp_row_ptrs, 
        hp_row_lengths, 
        hp_col_idxs, 
        hp_values, 
        hp_x, 
        hp_y,
        lp_C, 
        lp_n_rows, 
        lp_row_ptrs, 
        lp_row_lengths, 
        lp_col_idxs, 
        lp_values, 
        lp_x, 
        lp_y
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
scs_impl_gpu(const ST n_chunks,
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

    if (row < n_chunks * C) {
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

        #define X(CC) case CC: scs_impl_gpu<CC>(*n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl_gpu<SCS_C>(*n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
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
scs_mp_impl_gpu(
    const ST hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y
)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;
    ST c   = row / C;  // the no. of the chunk
    ST idx = row % C;  // index inside the chunk

    if (row < hp_n_chunks * C) {
        double hp_tmp{};
        double lp_tmp{};

        IT hp_cs = hp_chunk_ptrs[c];
        IT lp_cs = lp_chunk_ptrs[c];

        for (ST j = 0; j < hp_chunk_lengths[c]; ++j) {
            hp_tmp += hp_values[hp_cs + j * C + idx] * hp_x[hp_col_idxs[hp_cs + j * C + idx]];
        }
        for (ST j = 0; j < lp_chunk_lengths[c]; ++j) {
            lp_tmp += lp_values[lp_cs + j * C + idx] * lp_x[lp_col_idxs[lp_cs + j * C + idx]];
            // lp_tmp += lp_values[lp_cs + j * C + idx] * hp_x[lp_col_idxs[lp_cs + j * C + idx]];
        }

        hp_y[row] = hp_tmp + lp_tmp;
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
spmv_gpu_scs_mp_adv(
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y // unused
)
{
    switch (*hp_C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256)

        #define X(CC) case CC: scs_mp_impl_gpu<CC>(*hp_n_chunks, hp_chunk_ptrs, hp_chunk_lengths, hp_col_idxs, hp_values, hp_x, hp_y, *lp_n_chunks, lp_chunk_ptrs, lp_chunk_lengths, lp_col_idxs, lp_values, lp_x, lp_y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_mp_impl_gpu<SCS_C>(*hp_n_chunks, hp_chunk_ptrs, hp_chunk_lengths, hp_col_idxs, hp_values, hp_x, hp_y, *lp_n_chunks, lp_chunk_ptrs, lp_chunk_lengths, lp_col_idxs, lp_values, lp_x, lp_y);
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
void spmv_gpu_mp_scs_adv_launcher(
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST * lp_C, // 1
    const ST * lp_n_chunks, // n_chunks (same for both)
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    const ST * n_blocks,
    const int * my_rank
){
    spmv_gpu_scs_mp_adv<IT><<<*n_blocks, THREADS_PER_BLOCK>>>(
        hp_C, // 1
        hp_n_chunks, // n_chunks (same for both)
        hp_chunk_ptrs, // hp_chunk_ptrs
        hp_chunk_lengths, // unused
        hp_col_idxs,
        hp_values,
        hp_x,
        hp_y, 
        lp_C, // 1
        lp_n_chunks, // n_chunks (same for both)
        lp_chunk_ptrs, // lp_chunk_ptrs
        lp_chunk_lengths, // unused
        lp_col_idxs,
        lp_values,
        lp_x,
        lp_y // unused
    );
}

#endif

#endif