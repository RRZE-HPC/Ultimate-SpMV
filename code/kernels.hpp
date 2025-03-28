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
    int * block_size = nullptr,
    int * vec_length = nullptr,
    const int * my_rank = nullptr
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

            #pragma omp simd simdlen(SIMD_LENGTH)
            for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                sum += values[j] * x[col_idxs[j]];
#ifdef DEBUG_MODE_FINE
                printf("rank %i: %f += %f * %f using col idx %i w/ j=%i, row=%i\n", *my_rank, sum, values[j], x[col_idxs[j]], col_idxs[j], j, row);
#endif
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
 * SpMMV Kernel for CSR format, where X and Y are block vectors.
 */
template <typename VT, typename IT>
static void
block_spmv_omp_csr(
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
    int * vec_length,
    const int * my_rank = NULL
)
{
#ifdef DEBUG_MODE_FINE
    int test_rank = 2;
#endif
    #pragma omp parallel 
    {   
#ifdef USE_LIKWID
        if(!warmup_flag){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_START("block_colwise_spmv_crs_benchmark");
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_START("block_rowwise_spmv_crs_benchmark");
#endif
        }
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < *num_rows; ++row) {
            VT tmp[*block_size];

            for (int vec_idx = 0; vec_idx < *block_size; ++vec_idx) {
                tmp[vec_idx] = VT{};
            }

            for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                #pragma omp simd simdlen(SIMD_LENGTH)
                for (int vec_idx = 0; vec_idx < *block_size; ++vec_idx) {
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                    tmp[vec_idx] += values[j] * X[(*vec_length * vec_idx) + col_idxs[j]];
#ifdef DEBUG_MODE_FINE
                    printf("rank %i: %f += %f * %f using col idx %i w/ j=%i, row=%i\n", *my_rank,tmp[vec_idx], values[j], X[(*vec_length * vec_idx) + col_idxs[j]], col_idxs[j], j, row);
#endif
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                    tmp[vec_idx] += values[j] * X[col_idxs[j] * (*block_size) + vec_idx];
#ifdef DEBUG_MODE_FINE
                    printf("Accessing tmp[%i] += X[%i]\n", vec_idx, col_idxs[j] * (*block_size) + vec_idx);
#endif
#endif
                }
            }

            #pragma omp simd simdlen(SIMD_LENGTH)
            for (int vec_idx = 0; vec_idx < *block_size; ++vec_idx) {
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                Y[row + (vec_idx * *vec_length)] = tmp[vec_idx];
#ifdef DEBUG_MODE_FINE
                if(*my_rank == test_rank){printf("Assigning %f to Y[%i]\n", tmp[vec_idx], row + (vec_idx * *vec_length));}
#endif
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                Y[row * (*block_size) + vec_idx] = tmp[vec_idx];
#ifdef DEBUG_MODE_FINE
                if(*my_rank == test_rank){printf("Assigning %f to Y[%i]\n", tmp[vec_idx], row * (*block_size) + vec_idx);}
#endif
#endif
            }
        }

#ifdef USE_LIKWID
        if(!warmup_flag){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_STOP("block_colwise_spmv_crs_benchmark");
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_STOP("block_rowwise_spmv_crs_benchmark");
#endif
        }
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
    int * vec_length,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {
#ifdef USE_LIKWID
        if(!warmup_flag){
            LIKWID_MARKER_START("spmv_scs_benchmark");
        }
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
#ifdef DEBUG_MODE_FINE
                    printf("Accessing tmp[%i] += X[%i]\n", i, col_idxs[cs + j * *C + i]);
#endif
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
    int * vec_length,
    const int * my_rank = NULL
)
{
    switch (*C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64) X(128)

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
 * SpMMV Kernel for Sell-C-Sigma, where X and Y are block vectors. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
block_spmv_omp_scs_general(
    bool warmup_flag,
    const ST * C,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT X,
    VT * RESTRICT Y,
    int * block_size,
    int * vec_length,
    const int * my_rank = NULL
)
{
    #pragma omp parallel 
    {
#ifdef USE_LIKWID
        if(!warmup_flag){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_START("block_colwise_spmv_scs_benchmark");
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_START("block_rowwise_spmv_scs_benchmark");
#endif
        }
#endif
#ifdef DEBUG_MODE
        int test_rank = 0;
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < *n_chunks; ++c) {
            VT tmp[*C * *block_size];
            for (ST i = 0; i < *C * *block_size; ++i) {
                tmp[i] = VT{};
            }

            IT cs = chunk_ptrs[c];

            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *C; ++i) {
                    #pragma omp simd simdlen(SIMD_LENGTH)
                    for (IT vec_idx = 0; vec_idx < *block_size; ++vec_idx) {
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
	                    tmp[i * (*block_size) + vec_idx] += values[cs + j * *C + i] * X[col_idxs[cs + j * *C + i] + vec_idx * (*vec_length)];
#ifdef DEBUG_MODE
                        if(*my_rank == test_rank){printf("Accessing tmp[%i] += X[%i]\n", i * (*block_size) + vec_idx, col_idxs[cs + j * *C + i] + vec_idx * (*vec_length));}
#endif
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
	                    tmp[i * (*block_size) + vec_idx] += values[cs + j * *C + i] * X[col_idxs[cs + j * *C + i] * (*block_size) + vec_idx];

#ifdef DEBUG_MODE
                        if(*my_rank == test_rank){printf("Accessing tmp[%i] += X[%i]\n", i * (*block_size) + vec_idx, col_idxs[cs + j * *C + i] * (*block_size) + vec_idx);}
#endif
#endif
                    }
                }
            }

            for (IT i = 0; i < *C; ++i) {
                #pragma omp simd simdlen(SIMD_LENGTH)
                for (IT vec_idx = 0; vec_idx < *block_size; ++vec_idx) {
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                        Y[(c * *C + i) + vec_idx * (*vec_length)] = tmp[i * (*block_size) + vec_idx];
#ifdef DEBUG_MODE
                        if(*my_rank == test_rank){printf("Assigning %f to Y[%i]\n", tmp[i * (*block_size) + vec_idx], (c * *C + i) + vec_idx * (*n_chunks * *C));}
#endif
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                        Y[(c * *C + i) * (*block_size) + vec_idx] = tmp[i * (*block_size) + vec_idx];
#ifdef DEBUG_MODE
                        if(*my_rank == test_rank){printf("Assigning %f to Y[%i]\n", tmp[i * (*block_size) + vec_idx], (c * *C + i) * (*block_size) + vec_idx);}
#endif
#endif
                }
            }
        }

#ifdef USE_LIKWID
        if(!warmup_flag){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_STOP("block_colwise_spmv_scs_benchmark");
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_STOP("block_rowwise_spmv_scs_benchmark");
#endif
        }
#endif
    }
}


/**
 * Sell-C-sigma implementation templated by C, where X and Y are block vectors. Supports specific Cs > 0.
 */
template <ST C, typename VT, typename IT>
static void
block_scs_impl_cpu(
    bool warmup_flag,
    const ST * n_chunks,
    const IT * RESTRICT chunk_ptrs,
    const IT * RESTRICT chunk_lengths,
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    const IT * vec_length,
    const IT * block_size,
    VT * RESTRICT X,
    VT * RESTRICT Y
)
{
    const int fixed_block_size = *block_size;
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        if(!warmup_flag){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_START("block_colwise_spmv_scs_adv_benchmark");
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_START("block_rowwise_spmv_scs_adv_benchmark");
#endif
        }
#endif
        #pragma omp for schedule(static)
        for (ST c = 0; c < *n_chunks; ++c) {
            VT tmp[C * fixed_block_size];

            IT cs = chunk_ptrs[c];

            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < C; ++i) {
                    #pragma omp simd simdlen(SIMD_LENGTH)
                    for (IT n = 0; n < (*block_size); ++n) {
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                        tmp[i * (*block_size) + n] += values[cs + j * C + i] * X[col_idxs[cs + j * C + i] + n * (*vec_length)];
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                        tmp[i * (*block_size) + n] += values[cs + j * C + i] * X[col_idxs[cs + j * C + i] * (*block_size) + n];
#endif
                    }
                } 
            }
            
            for (IT i = 0; i < C; ++i) {
                #pragma omp simd simdlen(SIMD_LENGTH)
                for (IT n = 0; n < (*block_size); ++n) {
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                    Y[(c * C + i) + n * (*vec_length)] = tmp[i * (*block_size) + n];
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                    Y[(c * C + i) * (*block_size) + n] = tmp[i * (*block_size) + n];
#endif
                }
            }
        }


#ifdef USE_LIKWID
        if(!warmup_flag){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_STOP("block_colwise_spmv_scs_adv_benchmark");
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
            LIKWID_MARKER_STOP("block_rowwise_spmv_scs_adv_benchmark");
#endif
        }
#endif
    }
}

// TODO: We currently do no dispatching for block vector width
// /**
//  * Dispatch to Sell-C-sigma kernels templated by C, where X and Y are block vectors.
//  *
//  * Note: only works for selected Cs, see INSTANTIATE_CS.
//  */
// template <ST C, typename VT, typename IT>
// static void
// call_scs_block(
//     bool warmup_flag,
//     const ST * CC,
//     const ST * n_chunks,
//     const IT * RESTRICT chunk_ptrs,
//     const IT * RESTRICT chunk_lengths,
//     const IT * RESTRICT col_idxs,
//     const VT * RESTRICT values,
//     VT * RESTRICT x,
//     VT * RESTRICT y,
//     int * block_size,
//     int * vec_length
// )
// {
//     switch (*block_size)
//     {
//         #define INSTANTIATE_CS_B X(2) X(4) X(8) X(16) X(32) X(64) X(128)

//         #define X(BS) case BS: block_scs_impl_cpu<C,BS>(warmup_flag, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, vec_length, x, y); return;
//         INSTANTIATE_CS_B
//         #undef X

// 	default:
//             // Call this kernel, in the case where chunk size C is dispatched, but block_vec_size is not
//             block_spmv_omp_scs_general<VT, IT>(warmup_flag, CC, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y, block_size, vec_length); return;
//     }
// }

/**
 * Dispatch to Sell-C-sigma kernels templated by C.
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
    int * block_size,
    int * vec_length,
    const int * my_rank = NULL
)
{
    switch (*C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64) X(128)

        #define X(CC) case CC: block_scs_impl_cpu<CC>(warmup_flag, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, vec_length, block_size, x, y); break;
        INSTANTIATE_CS
        #undef X

	default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(C));
        exit(1);    
    }
}

#ifdef __CUDACC__
template <typename VT, typename IT>
__global__ void pack_d_send_buf(
    VT **d_to_send_elems, 
    VT *d_local_x, 
    IT *d_perm, 
    IT **d_comm_send_idxs, 
    IT block_offset, 
    IT outgoing_buf_size,
    IT to_proc_idx,
    IT receiving_proc
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outgoing_buf_size) {
        // printf("to_proc_idx = %i\n", to_proc_idx);
        // printf("i = %i\n", i);
        // printf("%f ", d_to_send_elems[to_proc_idx][i]);
        (d_to_send_elems[to_proc_idx])[i] = d_local_x[d_perm[d_comm_send_idxs[receiving_proc][i]] + block_offset];
        // d_to_send_elems[to_proc_idx][i] = (VT) 1.0;
        // printf("to_proc_idx = %i\n", to_proc_idx);
        // printf("outgoing_buf_size = %i\n", outgoing_buf_size);
        // printf("d_to_send_elems[to_proc_idx] = %f\n", d_to_send_elems[to_proc_idx]);
    }
}

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
    int *block_size,
    int * vec_length,
    const ST n_thread_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_scs<<<n_thread_blocks, THREADS_PER_BLOCK>>>(
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
    int * block_size, // not used
    int * vec_length, // not used
    const ST n_thread_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_csr<<<n_thread_blocks, THREADS_PER_BLOCK>>>(
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
    int * block_size,
    int * vec_length,
    const ST n_thread_blocks,
    const int * my_rank = NULL
){
    spmv_gpu_scs_adv<<<n_thread_blocks, THREADS_PER_BLOCK>>>(
        C, n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y
    );
}

template <typename VT, typename IT>
void block_spmv_gpu_csr_launcher(
    bool warmup_flag, // not used
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT X,
    VT * RESTRICT Y,
    int * block_size,
    int * vec_length,
    const ST n_thread_blocks,
    const int * my_rank = NULL
){
    // TODO
    if(my_rank == 0){
        printf("Kernel not yet implemented.\n");
        exit(0);
    }
}

template <typename VT, typename IT>
void block_spmv_gpu_scs_adv_launcher(
    bool warmup_flag, // not used
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT X,
    VT * RESTRICT Y,
    int * block_size,
    int * vec_length,
    const ST n_thread_blocks,
    const int * my_rank = NULL
){
    // TODO
    if(my_rank == 0){
        printf("Kernel not yet implemented.\n");
        exit(0);
    }
}

template <typename VT, typename IT>
void block_spmv_gpu_scs_general_launcher(
    bool warmup_flag, // not used
    const ST * C, // 1
    const ST * num_rows, // n_chunks
    const IT * RESTRICT row_ptrs, // chunk_ptrs
    const IT * RESTRICT row_lengths, // unused
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT X,
    VT * RESTRICT Y,
    int * block_size,
    int * vec_length,
    const ST n_thread_blocks,
    const int * my_rank = NULL
){
    // TODO
    if(my_rank == 0){
        printf("Kernel not yet implemented.\n");
        exit(0);
    }
}

#endif

#endif