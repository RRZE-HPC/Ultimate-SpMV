#ifndef AP_KERNELS
#define AP_KERNELS

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>

#define RESTRICT				__restrict__

// Adaptive precision SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.



/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename IT>
static void
scs_ap_impl_cpu(
    bool warmup_flag,
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
        if(!warmup_flag)
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
        if(!warmup_flag)
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
    bool warmup_flag,
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

        #define X(CC) case CC: scs_ap_impl_cpu<CC,IT>(warmup_flag, dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_ap_impl_cpu<SCS_C, IT>(warmup_flag, dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y);
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
spmv_omp_csr_apdpsp(
    bool warmup_flag,
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
        if(!warmup_flag)
            LIKWID_MARKER_START("spmv_apdpsp_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *dp_n_rows; ++row) {
            double dp_sum{};
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:dp_sum)
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
            // #pragma omp simd simdlen(2*SIMD_LENGTH) reduction(+:sp_sum)
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sp_sum)
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
        if(!warmup_flag)
            LIKWID_MARKER_STOP("spmv_apdpsp_crs_benchmark");
#endif
    }
}

#ifdef HAVE_HALF_MATH
template <typename IT>
static void
spmv_omp_csr_apdphp(
    bool warmup_flag,
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
        if(!warmup_flag)
            LIKWID_MARKER_START("spmv_apdphp_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *dp_n_rows; ++row) {
            double dp_sum{};
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:dp_sum)
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
            // #pragma omp simd simdlen(2*SIMD_LENGTH) reduction(+:sp_sum)
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sp_sum)
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
        if(!warmup_flag)
            LIKWID_MARKER_STOP("spmv_apdphp_crs_benchmark");
#endif
    }
}

template <typename IT>
static void
spmv_omp_csr_apsphp(
    bool warmup_flag,
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
        if(!warmup_flag)
            LIKWID_MARKER_START("spmv_apsphp_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *sp_n_rows; ++row) {
            double sp_sum{};
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:dp_sum)
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
            // #pragma omp simd simdlen(2*SIMD_LENGTH) reduction(+:sp_sum)
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sp_sum)
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
        if(!warmup_flag)
            LIKWID_MARKER_STOP("spmv_apsphp_crs_benchmark");
#endif
    }
}

template <typename IT>
static void
spmv_omp_csr_apdpsphp(
    bool warmup_flag,
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
        if(!warmup_flag)
            LIKWID_MARKER_START("spmv_apdpsphp_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *sp_n_rows; ++row) {

            double dp_sum{};
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:dp_sum)
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
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:dp_sum)
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
            // #pragma omp simd simdlen(2*SIMD_LENGTH) reduction(+:sp_sum)
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sp_sum)
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
        if(!warmup_flag)
            LIKWID_MARKER_STOP("spmv_apdpsphp_crs_benchmark");
#endif
    }
}

#endif

template <typename IT>
static void
spmv_omp_csr_ap(
    bool warmup_flag,
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
        if(!warmup_flag)
            LIKWID_MARKER_START("spmv_ap_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *dp_n_rows; ++row) {
            double dp_sum{};
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:dp_sum)
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
            // #pragma omp simd simdlen(2*SIMD_LENGTH) reduction(+:sp_sum)
            // #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sp_sum)
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
        if(!warmup_flag)
            LIKWID_MARKER_STOP("spmv_ap_crs_benchmark");
#endif
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename IT>
static void
spmv_omp_scs_ap(
    bool warmup_flag,
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
        if(!warmup_flag)
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
        if(!warmup_flag)
            LIKWID_MARKER_STOP("spmv_ap_scs_benchmark");
#endif
    }
}

#ifdef __CUDACC__
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
    bool warmup_flag, // not used
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
    const ST n_thread_block,
    const int * my_rank = NULL
){
    spmv_gpu_ap_csr<IT><<<n_thread_block, THREADS_PER_BLOCK>>>(
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
    bool warmup_flag, // not_used
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
    const ST n_thread_block,
    const int * my_rank = NULL
){
    spmv_gpu_ap_scs<IT><<<n_thread_block, THREADS_PER_BLOCK>>>(
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

    // spmv_gpu_scs<double, int><<<n_thread_block, THREADS_PER_BLOCK>>>(
    //     dp_C, dp_n_chunks, dp_chunk_ptrs, dp_chunk_lengths, dp_col_idxs, dp_values, dp_x, dp_y
    // );

    // spmv_gpu_scs<float, int><<<n_thread_block, THREADS_PER_BLOCK>>>(
    //     sp_C, sp_n_chunks, sp_chunk_ptrs, sp_chunk_lengths, sp_col_idxs, sp_values, sp_x, sp_y
    // );

}

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
    bool warmup_flag, // not used
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
    const ST n_thread_block,
    const int * my_rank = NULL
){
    // printf("Do I get to the launcher? n_thread_blocks = %i\n", n_thread_block);
    spmv_gpu_scs_ap_adv<IT><<<n_thread_block, THREADS_PER_BLOCK>>>(
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