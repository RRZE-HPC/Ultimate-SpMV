#ifndef KERNELS
#define KERNELS

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#define RESTRICT				__restrict__

// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Kernel for CSR format.
 */
template <typename VT, typename IT>
static void
spmv_omp_csr(const ST C, // 1
             const ST num_rows, // n_chunks
             const IT * RESTRICT row_ptrs, // chunk_ptrs
             const IT * RESTRICT row_lengths, // unused
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             int my_rank)
{
    #pragma omp parallel 
    {   
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < num_rows; ++row) {
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
    const ST C, // unused
    const ST num_rows,
    const IT * RESTRICT chunk_ptrs, // unused
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int my_rank)
{
    ST nelems_per_row = chunk_lengths[1] - chunk_lengths[0];
    std::cout << nelems_per_row << std::endl;
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; row++) {
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
    const ST C, // unused
    const ST num_rows,
    const IT * RESTRICT chunk_ptrs, // unused
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int my_rank)
{
    ST nelems_per_row = chunk_lengths[1] - chunk_lengths[0];
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; row++) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; i++) {
            VT val = values[row + i * num_rows];
            IT col = col_idxs[row + i * num_rows];

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
spmv_omp_scs(const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             int my_rank)
{
    #pragma omp parallel 
    {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("spmv_scs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < n_chunks; ++c) {
            VT tmp[C];
            for (ST i = 0; i < C; ++i) {
                tmp[i] = VT{};
            }

            IT cs = chunk_ptrs[c];

            // TODO: use IT wherever possible
            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < (IT)C; ++i) {
                    tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
                }
            }

            for (ST i = 0; i < C; ++i) {
                y[c * C + i] = tmp[i];
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
scs_impl(const ST n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         VT * RESTRICT x,
         VT * RESTRICT y)
{

    #pragma omp parallel for schedule(static)
    for (ST c = 0; c < n_chunks; ++c) {
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
}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename VT, typename IT>
static void
spmv_omp_scs_adv(
             const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             int my_rank)
{
    switch (C)
    {
        #define INSTANTIATE_CS X(1) X(2) X(4) X(8) X(16) X(32) X(64)

        #define X(CC) case CC: scs_impl<CC>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl<SCS_C>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
        break;
#endif
    default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(C));
        exit(1);
    }
}
#endif