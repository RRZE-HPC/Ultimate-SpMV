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
scs_impl_cpu(const ST n_chunks,
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

template <typename IT>
static void
spmv_omp_csr_mp_1(
    const ST hp_n_rows, // TODO: (same for both)
    const ST hp_C, // 1
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST lp_n_rows, // TODO: (same for both)
    const ST lp_C, // 1
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_row_lengths, // unused for now
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    int my_rank,
    IT *lp_perm,
    IT *hp_perm,
    IT *lp_inv_perm,
    IT *hp_inv_perm
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
            for (ST row = 0; row < hp_n_rows; ++row) {
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
                

                float lp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:lp_sum)
                #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:lp_sum)
                for (IT j = lp_row_ptrs[row]; j < lp_row_ptrs[row + 1]; ++j) {
                    lp_sum += lp_values[j] * lp_x[lp_col_idxs[j]];
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

template <typename IT>
static void
spmv_omp_csr_mp_2(
    const ST hp_n_rows, // TODO: (same for both)
    const ST hp_C, // 1
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST lp_n_rows, // TODO: (same for both)
    const ST lp_C, // 1
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_row_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    int my_rank
    )
{

    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < hp_n_rows; ++row) {

        double hp_sum{};
        float lp_sum{};

        IT hp_rs = hp_row_ptrs[row];
        IT lp_rs = lp_row_ptrs[row];

        // Should have stored beforehand?
        IT hp_row_length = hp_row_ptrs[row+1] - hp_row_ptrs[row];
        IT lp_row_length = lp_row_ptrs[row+1] - lp_row_ptrs[row];

        IT combined_row = hp_row_length + lp_row_length;

        #pragma omp simd reduction(+:hp_sum,lp_sum)
        for (IT j = 0; j < combined_row; ++j) {
            if(j < hp_row_length){
                hp_sum += hp_values[hp_rs + j] * hp_x[hp_col_idxs[hp_rs + j]];
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                printf("j = %i, hp_col_idxs[j] = %i, hp_x[hp_col_idxs[j]] = %f\n", j ,hp_col_idxs[j], hp_x[hp_col_idxs[j]]);
                printf("hp_sum += %f * %f\n", hp_values[j], hp_x[hp_col_idxs[j]]);
            }
#endif
            }
            else{
                IT lp_shift_j = j - hp_row_length;
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                printf("lp_rs + lp_shift_j = %i\n", lp_rs + lp_shift_j);
                printf("lp_col_idxs[lp_rs + lp_shift_j] = %i\n", lp_col_idxs[lp_rs + lp_shift_j]);
                printf("lp_x[lp_col_idxs[lp_rs + lp_shift_j]] = %f\n", lp_x[lp_col_idxs[lp_rs + lp_shift_j]]);
                printf("lp_values[lp_rs + lp_shift_j] = %f\n", lp_values[lp_rs + lp_shift_j]);
            }
#endif
                lp_sum += lp_values[lp_rs + lp_shift_j] * lp_x[lp_col_idxs[lp_rs + lp_shift_j]];
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                printf("lp_rs + lp_shift_j = %i, lp_col_idxs[lp_rs + lp_shift_j] = %i, lp_x[lp_col_idxs[lp_rs + lp_shift_j]] = %f\n", lp_rs + lp_shift_j, lp_col_idxs[lp_rs + lp_shift_j], lp_x[lp_col_idxs[lp_rs + lp_shift_j]]);
                printf("lp_sum += %f * %f\n", lp_values[lp_rs + lp_shift_j], lp_x[lp_col_idxs[lp_rs + lp_shift_j]]);

            }
#endif
            }
        }
        // ^ at this point, it's basically just SCS, C=1...

        hp_y[row] = hp_sum + lp_sum;
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename IT>
static void
spmv_omp_scs_mp_1(
    const ST hp_n_chunks, // n_chunks (same for both)
    const ST hp_C, // 1
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST lp_n_chunks, // n_chunks (same for both)
    const ST lp_C, // 1
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    int my_rank,
    IT *lp_perm,
    IT *hp_perm,
    IT *lp_inv_perm,
    IT *hp_inv_perm
)
{
    std::vector<double> sum_arr(hp_n_chunks*hp_C,0);

    // std::vector<IT> lp_old_chunk_lengths(hp_n_chunks,0);
    // // apply_permutation<IT, IT>(&(lp_old_chunk_lengths)[0], lp_chunk_lengths, lp_perm, hp_n_chunks);
    // for(int i = 0; i < hp_n_chunks; ++i){
    //     lp_old_chunk_lengths[i] = lp_chunk_lengths[lp_perm[i]];
    //     // std::cout << "Permuting:" << vec_to_permute[i] <<  " to " << vec_to_permute[perm[i]] << std::endl;
    // }
    // std::vector<IT> hp_old_chunk_lengths(hp_n_chunks,0);
    // // apply_permutation<IT, IT>(&(hp_old_chunk_lengths)[0], hp_chunk_lengths, hp_perm, hp_n_chunks);
    // for(int i = 0; i < hp_n_chunks; ++i){
    //     hp_old_chunk_lengths[i] = hp_chunk_lengths[hp_perm[i]];
    //     // std::cout << "Permuting:" << vec_to_permute[i] <<  " to " << vec_to_permute[perm[i]] << std::endl;
    // }

    // std::vector<IT> lp_old_chunk_ptrs(hp_n_chunks+1,0);
    // for(int i = 0; i < hp_n_chunks; ++i){
    //     lp_old_chunk_ptrs[i+1] = lp_chunk_ptrs[lp_perm[i+1]];
    // }

    // std::vector<IT> hp_old_chunk_ptrs(hp_n_chunks+1,0);
    // for(int i = 0; i < hp_n_chunks; ++i){
    //     hp_old_chunk_ptrs[i+1] = hp_chunk_ptrs[hp_perm[i+1]];
    // }

    // std::cout << "hp_old_to_new: " << std::endl;
    // for(int i = 0; i < hp_n_chunks*hp_C; ++i){
    //     std::cout << hp_perm[i] << std::endl;
    // }
    // std::cout << "hp_new_to_old: " << std::endl;
    // for(int i = 0; i < hp_n_chunks*hp_C; ++i){
    //     std::cout << hp_inv_perm[i] << std::endl;
    // }

    // std::cout << "lp_old_to_new: " << std::endl;
    // for(int i = 0; i < hp_n_chunks*hp_C; ++i){
    //     std::cout << lp_perm[i] << std::endl;
    // }
    // std::cout << "lp_new_to_old: " << std::endl;
    // for(int i = 0; i < hp_n_chunks*hp_C; ++i){
    //     std::cout << lp_inv_perm[i] << std::endl;
    // }

    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < hp_n_chunks; ++c) {
            double hp_tmp[hp_C];
            float lp_tmp[hp_C];

            for (ST i = 0; i < hp_C; ++i) {
                hp_tmp[i] = 0.0;
            }
            for (ST i = 0; i < hp_C; ++i) {
                lp_tmp[i] = 0.0f;
            }

            IT hp_cs = hp_chunk_ptrs[c];
            IT lp_cs = lp_chunk_ptrs[c];

            for (IT j = 0; j < hp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < hp_C; ++i) {

                    hp_tmp[i] += hp_values[hp_cs + j * hp_C + i] * hp_x[hp_col_idxs[hp_cs + j * hp_C + i]];
#ifdef DEBUG_MODE_FINE
                    if(my_rank == 0){                    
                        printf("hp_cs = %i, \
                            j = %i, \
                            hp_cs + j * hp_C + i = %i, \
                            hp_chunk_lengths[c] = %i, \
                            j * hp_C = %i, \
                            hp_cs + j * hp_C + i = %i, \
                            hp_values[hp_cs + j * hp_C + i] = %f, \
                            hp_col_idxs[hp_cs + j * hp_C + i] = %i, \
                            hp_x[hp_col_idxs[hp_cs + j * hp_C + i]] = %f\n", \
                            hp_cs, \
                            j, \
                            hp_cs + j * hp_C + i, \
                            hp_chunk_lengths[c], \
                            j * hp_C, \
                            hp_cs + j * hp_C + i, \
                            hp_values[hp_cs + j * hp_C + i], \
                            hp_col_idxs[hp_cs + j * hp_C + i], \
                            hp_x[hp_col_idxs[hp_cs + j * hp_C + i]]);

                        printf("lp_tmp[i]: %f += %f * %f\n", \
                            hp_tmp[i], hp_values[hp_cs + j * hp_C + i], hp_x[hp_col_idxs[hp_cs + j * hp_C + i]]);
                    }
#endif
                }
            }
            for (IT j = 0; j < lp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < hp_C; ++i) {

                    lp_tmp[i] += lp_values[lp_cs + j * hp_C + i] * lp_x[lp_col_idxs[lp_cs + j * hp_C + i]];
#ifdef DEBUG_MODE_FINE
                    if(my_rank == 0){                    
                        printf("lp_cs = %i, \
                            j = %i, \
                            lp_cs + j * hp_C + i = %i, \
                            lp_chunk_lengths[c] = %i, \
                            j * hp_C = %i, \
                            lp_cs + j * hp_C + i = %i, \
                            lp_values[lp_cs + j * hp_C + i] = %f, \
                            lp_col_idxs[lp_cs + j * hp_C + i] = %i, \
                            lp_x[lp_col_idxs[lp_cs + j * hp_C + i]] = %f\n", \
                            lp_cs, \
                            j, \
                            lp_cs + j * hp_C + i, \
                            lp_chunk_lengths[c], \
                            lp_cs + j * hp_C + i, \
                            lp_values[lp_cs + j * hp_C + i], \
                            lp_col_idxs[lp_cs + j * hp_C + i], \
                            lp_x[lp_col_idxs[lp_cs + j * hp_C + i]]);

                        printf("lp_tmp[i]: %f += %f * %f\n", \
                            lp_tmp[i], lp_values[lp_cs + j * hp_C + i], lp_x[lp_col_idxs[lp_cs + j * hp_C + i]]);
                    }
#endif
                }
            }

            // This needs to stay the same, for race condition reasons
            for (ST i = 0; i < hp_C; ++i) {
                // hp_y[c * hp_C + i] = hp_tmp[i] + lp_tmp[i];
                // hp_y[c * hp_C + i] = hp_tmp[i]; // permmed according to hp reordering
                // lp_y[hp_perm[lp_inv_perm[c * hp_C + i]]] = lp_tmp[i]; // permmed according to lp reordering
                // lp_y[c * hp_C + i] = lp_tmp[i];
#ifdef DEBUG_MODE_FINE
                if(my_rank == 0){
                    printf("y[%i] = %f\n", c * hp_C + i, hp_tmp[i] + lp_tmp[i]);
                }
#endif
                sum_arr[c * hp_C + i] += hp_tmp[i];
                sum_arr[hp_perm[lp_inv_perm[c * hp_C + i]]] += lp_tmp[i]; //implicit conversion to double

                // hp_y[c * hp_C + i] += sum_arr[c * hp_C + i];
                // lp_y[c * hp_C + i] += sum_arr[hp_perm[lp_inv_perm[c * hp_C + i]]];
            }
            
        }

        // Combine results
        // // TODO: bandaid, should be a swap

        // nowait?
        #pragma omp for schedule(static)
        for (ST c = 0; c < hp_n_chunks; ++c) {
            for (IT i = 0; i < hp_C; ++i) {
                hp_y[c * hp_C + i] = sum_arr[c * hp_C + i];
                // lp_y[c * hp_C + i] = sum_arr[c * hp_C + i];
                lp_y[c * hp_C + i] = sum_arr[hp_perm[lp_inv_perm[c * hp_C + i]]]; // <- cant work
            }
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_benchmark");
#endif
    }

}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename IT>
static void
spmv_omp_scs_mp_2(
    const ST hp_n_chunks, // n_chunks (same for both)
    const ST hp_C, // 1
    const IT * RESTRICT hp_chunk_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST lp_n_chunks, // n_chunks (same for both)
    const ST lp_C, // 1
    const IT * RESTRICT lp_chunk_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_chunk_lengths, // unused
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    int my_rank
)
{
    #pragma omp parallel for schedule(static)
    for (ST c = 0; c < hp_n_chunks; ++c) {
        double hp_tmp[hp_C];
        float lp_tmp[hp_C];

        for (ST i = 0; i < hp_C; ++i) {
            hp_tmp[i] = 0.0;
        }
        for (ST i = 0; i < hp_C; ++i) {
            lp_tmp[i] = 0.0f;
        }

        IT hp_cs = hp_chunk_ptrs[c];
        IT lp_cs = lp_chunk_ptrs[c];

        IT combined_chunk = hp_chunk_lengths[c]+lp_chunk_lengths[c];

        for (IT j = 0; j < combined_chunk; ++j) {
            if(j < hp_chunk_lengths[c]){
                for (IT i = 0; i < hp_C; ++i) {
                    hp_tmp[i] += hp_values[hp_cs + j * hp_C + i] * hp_x[hp_col_idxs[hp_cs + j * hp_C + i]];
                }  
            }
            else{
                IT lp_shift_j = j - hp_chunk_lengths[c];
                for (IT i = 0; i < hp_C; ++i) {
                    lp_tmp[i] += lp_values[lp_cs + lp_shift_j * hp_C + i] * lp_x[lp_col_idxs[lp_cs + lp_shift_j * hp_C + i]];
                }
            }
        }

        for (ST i = 0; i < hp_C; ++i) {
            hp_y[c * hp_C + i] = hp_tmp[i] + lp_tmp[i];
        }
    }
}

#ifdef __CUDACC__
template <typename VT, typename IT>
__global__ void
spmv_gpu_scs(const ST *_C,
         const ST *_n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y,
         int my_rank = 0)
{
    long C=*_C;
    long n_chunks=*_n_chunks;

    long row = threadIdx.x + blockDim.x * blockIdx.x;
    int c   = row / C;  // the no. of the chunk
    int idx = row % C;  // index inside the chunk

    if (row < n_chunks * C) {
        VT tmp{};
        int cs = chunk_ptrs[c];

        for (int j = 0; j < chunk_lengths[c]; ++j) {
            tmp += values[cs + j * C + idx] * x[col_idxs[cs + j * C + idx]];
        }

        y[row] = tmp;
    }

}

template <typename VT, typename IT>
__global__ void 
spmv_gpu_csr(
    const ST *_C, // 1
    const ST *_num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    const VT * RESTRICT x,
    VT * RESTRICT y,
    int my_rank = 0)
{

    long C=*_C;
    long num_rows=*_num_rows;

    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int thread_idx = block_offset + thread_idx_in_block;
    VT tmp{};

    // TODO: branch not performant!
    if(thread_idx < num_rows){
        // One thread per row
        for(int nz_idx = row_ptrs[thread_idx]; nz_idx < row_ptrs[thread_idx+1]; ++nz_idx){
            tmp += values[nz_idx] * x[col_idxs[nz_idx]];
        }

        y[thread_idx] = tmp;
    }
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

#endif

#endif