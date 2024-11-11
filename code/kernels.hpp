#ifndef KERNELS
#define KERNELS

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>

#define RESTRICT				__restrict__

// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * SpMV Kernel for CSR format.
 */
template <typename VT, typename IT>
static void
spmv_omp_csr(
    bool warmup_flag,
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int *block_size,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {   
#ifdef USE_LIKWID
        if(!warmup_flag)
            LIKWID_MARKER_START("spmv_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *num_rows; ++row) {
            VT sum{};

            #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sum)
            for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                sum += values[j] * x[col_idxs[j]];
            }
            y[row] = sum;
        }
#ifdef USE_LIKWID
        if(!warmup_flag)
            LIKWID_MARKER_STOP("spmv_crs_benchmark");
#endif
    }
}

/**
 * SpMM Kernel for CSR format, where X and Y are held columnwise.
 */
template <typename VT, typename IT>
static void
block_colwise_spmv_omp_csr(
    bool warmup_flag,
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT X,
    VT * RESTRICT Y,
    int * block_size,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {   
#ifdef USE_LIKWID
        if(!warmup_flag)
            LIKWID_MARKER_START("block_colwise_spmv_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *num_rows; ++row) {
            for (int vec_idx = 0; vec_idx < *block_size; ++vec_idx) {
                VT sum{};

                #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sum)
                for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                    sum += values[j] * X[col_idxs[j] + vec_idx*(*num_rows)];
                }
                Y[row + vec_idx*(*num_rows)] = sum;
            }
        }
#ifdef USE_LIKWID
        if(!warmup_flag)
            LIKWID_MARKER_STOP("block_colwise_spmv_crs_benchmark");
#endif
    }
}

/**
 * SpMM Kernel for CSR format, where X and Y are held rowwise.
 */
template <typename VT, typename IT>
static void
block_rowwise_spmv_omp_csr(
    bool warmup_flag,
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT X,
    VT * RESTRICT Y,
    int * block_size,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {   
#ifdef USE_LIKWID
        if(!warmup_flag)
            LIKWID_MARKER_START("block_rowwise_spmv_crs_benchmark");
#endif

        // TODO

#ifdef USE_LIKWID
        if(!warmup_flag)
            LIKWID_MARKER_STOP("block_rowwise_spmv_crs_benchmark");
#endif
    }
}

/**
 * SpMV Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
spmv_omp_scs(
    bool warmup_flag,
    const ST * C,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int *block_size,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {
#ifdef USE_LIKWID
    if(!warmup_flag)
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
    if(!warmup_flag)
        LIKWID_MARKER_STOP("spmv_scs_benchmark");
#endif
    }
}

/**
 * SpMM Kernel for Sell-C-Sigma, where X and Y are held columnwise. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
block_colwise_spmv_omp_scs(
    bool warmup_flag,
    const ST * C,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int *block_size,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {
#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_START("block_colwise_spmv_scs_benchmark");
#endif

        // TODO

#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_STOP("block_colwise_spmv_scs_benchmark");
#endif
    }
}

/**
 * SpMM Kernel for Sell-C-Sigma, where X and Y are held rowwise. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
block_rowwise_spmv_omp_scs(
    bool warmup_flag,
    const ST * C,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int *block_size,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {
#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_START("block_colwise_spmv_scs_benchmark");
#endif

        // TODO

#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_STOP("block_colwise_spmv_scs_benchmark");
#endif
    }
}

/**
 * Sell-C-sigma implementation templated by C. Supports specific Cs > 0.
 */
template <ST C, typename VT, typename IT>
static void
scs_impl_cpu(
    bool warmup_flag,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y
)
{

    #pragma omp parallel
    {
#ifdef USE_LIKWID
    if(!warmup_flag)
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
    if(!warmup_flag)
        LIKWID_MARKER_STOP("spmv_scs_adv_benchmark");
#endif
    }
}

/**
 * Sell-C-sigma implementation templated by C, where X and Y are held columnwise. Supports specific Cs > 0.
 */
template <ST C, typename VT, typename IT>
static void
block_colwise_scs_impl_cpu(
    bool warmup_flag,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y
)
{

    #pragma omp parallel
    {
#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_START("block_colwise_spmv_scs_adv_benchmark");
#endif

        // TODO

#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_STOP("block_colwise_spmv_scs_adv_benchmark");
#endif
    }
}

/**
 * Sell-C-sigma implementation templated by C, where X and Y are held rowwise. Supports specific Cs > 0.
 */
template <ST C, typename VT, typename IT>
static void
block_rowwise_scs_impl_cpu(
    bool warmup_flag,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y
)
{

    #pragma omp parallel
    {
#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_START("block_rowwise_spmv_scs_adv_benchmark");
#endif

        // TODO

#ifdef USE_LIKWID
    if(!warmup_flag)
        LIKWID_MARKER_STOP("block_rowwise_spmv_scs_adv_benchmark");
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
    bool warmup_flag,
    const ST * C,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int *block_size,
    const int * my_rank = NULL
)
{
    switch (*C)
    {
        #define INSTANTIATE_CS X(1) X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256)

        #define X(CC) case CC: scs_impl_cpu<CC>(warmup_flag, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl_cpu<SCS_C>(warmup_flag, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
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
 * Dispatch to Sell-C-sigma kernels templated by C, where X and Y are blcok vectors.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename VT, typename IT>
static void
block_spmv_omp_scs_adv(
    bool warmup_flag,
    const ST * C,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    int *block_size,
    const int * my_rank = NULL
)
{
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
    // TODO
#endif

#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
    // TODO
#endif
}


#ifdef __CUDACC__
template <typename VT, typename IT>
__global__ void
spmv_gpu_scs(
    const ST *C,
    const ST *n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    const VT * RESTRICT x,
    VT * RESTRICT y,
    int *block_size,
    const int * my_rank = NULL
)
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
    bool warmup_flag, // not used
    const ST * C,
    const ST * n_chunks, // n_chunks
    const IT * RESTRICT chunk_ptrs, // chunk_ptrs
    const IT * RESTRICT chunk_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const ST * n_thread_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_scs<<<*n_thread_blocks, THREADS_PER_BLOCK>>>(
        C, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y
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
    bool warmup_flag, // not used
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
    const ST * n_thread_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_csr<<<*n_thread_blocks, THREADS_PER_BLOCK>>>(
        C, num_rows, row_ptrs, row_lengths, col_idxs, values, x, y
    );
}

/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename VT, typename IT>
__device__
static void
scs_impl_gpu(
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    const VT * RESTRICT x,
    VT * RESTRICT y
)
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
    VT * RESTRICT y
)
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
    bool warmup_flag, // not used
    const ST *C,
    const ST *n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    const VT * RESTRICT x,
    VT * RESTRICT y,
    const ST * n_thread_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_scs_adv<<<*n_thread_blocks, THREADS_PER_BLOCK>>>(
        C, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y
    );
}
#endif

#endif