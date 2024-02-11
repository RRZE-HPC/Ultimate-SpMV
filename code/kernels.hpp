#ifndef KERNELS
#define KERNELS

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
             const IT * RESTRICT chunk_lengths, // unused
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y,
             int my_rank)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; ++row) {
        VT sum{};
        // #pragma nounroll
        #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sum)
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            // if(my_rank == 1){printf("j = %i, col_idxs[j] = %i, x[col_idxs[j]] = %f\n", j ,col_idxs[j], x[col_idxs[j]]);}

            sum += values[j] * x[col_idxs[j]];
        }
        y[row] = sum;
    }
}

// TODO: I don't yet know how to set-up the signature to enable kernel picker
template <typename IT>
static void
spmv_omp_csr_mp(
    const ST num_rows, // n_chunks (same for both)
    const ST hp_C, // 1
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST lp_C, // 1
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    int my_rank
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
    for (ST row = 0; row < num_rows; ++row) {
        double hp_sum{};
        // #pragma nounroll
        // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:hp_sum)
        #pragma omp simd reduction(+:hp_sum)
        for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row + 1]; ++j) {
            hp_sum += hp_values[j] * hp_x[hp_col_idxs[j]];
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){printf("j = %i, hp_col_idxs[j] = %i, hp_x[hp_col_idxs[j]] = %f\n", j ,hp_col_idxs[j], hp_x[hp_col_idxs[j]]);}
#endif
        }

        float lp_sum{};
        // #pragma nounroll
        // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:lp_sum)
        #pragma omp simd reduction(+:lp_sum)
        for (IT j = lp_row_ptrs[row]; j < lp_row_ptrs[row + 1]; ++j) {
            lp_sum += lp_values[j] * lp_x[lp_col_idxs[j]];
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){printf("j = %i, lp_col_idxs[j] = %i, lp_x[hp_col_idxs[j]] = %f\n", j ,lp_col_idxs[j], lp_x[lp_col_idxs[j]]);}
#endif
        }

        hp_y[row] = hp_sum + lp_sum;
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
    #pragma omp parallel for schedule(static)
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