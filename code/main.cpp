#include "mmio.h"
#include "utilities.hpp"
#include "kernels.hpp"
#include "ap_kernels.hpp"
#include "mpi_funcs.hpp"
#include "write_results.hpp"
#include "sanity_checker.hpp"
#include "timing.h"

#include <cinttypes>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

// Constants for benchmarking harness
#define WARM_UP_REPS 100
#define MILLI_TO_SEC 0.001

#ifdef _OPENMP
#include <omp.h>
#endif

/**
    @brief Perform spmv kernel, either in "solve" mode or "bench" mode
    @param *config : struct to initialze default values and user input
    @param *local_scs : pointer to process-local scs struct
    @param *(dp/sp/hp)_local_scs : precision-specific copies, used in adaptive precision
    @param *local_context : struct containing communication information
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_y : Process-local results vector, instance of SimpleDenseMatrix class
    @param *(dp/sp/hp)_local_y : precision-specific copies, used in adaptive precision
    @param *local_x : local RHS vector, instance of SimpleDenseMatrix class
    @param *(dp/sp/hp)_local_x : precision-specific copies, used in adaptive precision
    @param *r : a Result struct, in which results of the benchmark are stored
*/
template <typename VT, typename IT>
void bench_spmv(
    Config *config,
    ScsData<VT, IT> *local_scs,
    ScsData<double, IT> *dp_local_scs,
    ScsData<float, IT> *sp_local_scs,
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> *hp_local_scs,
#endif
    ContextData<IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<double> *dp_local_y,
    std::vector<float> *sp_local_y,
#ifdef HAVE_HALF_MATH
    std::vector<_Float16> *hp_local_y,
#endif
    std::vector<VT> *local_x,
    std::vector<double> *dp_local_x,
    std::vector<float> *sp_local_x,
#ifdef HAVE_HALF_MATH
    std::vector<_Float16> *hp_local_x,
#endif
    Result<VT, IT> *r,
    int my_rank,
    int comm_size)
{
    // Permute x, in order to match the permutation which was done to the columns
    // std::vector<VT> local_x_permuted(local_x->size(), VT{}); <- is this more correct?
    std::vector<VT> local_x_permuted(local_x->size(), VT{});
    std::vector<double> dp_local_x_permuted(dp_local_x->size(), 0.0);
    std::vector<float> sp_local_x_permuted(sp_local_x->size(), 0.0f);
#ifdef HAVE_HALF_MATH
    std::vector<_Float16> hp_local_x_permuted(hp_local_x->size(), 0.0f16);
#endif
    // TODO: Something here seems iffy. I think something is going wrong with the inverse perm vec
    for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
        apply_permutation<VT, IT>(&(local_x_permuted)[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)], &(*local_x)[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)], &(local_scs->new_to_old_idx)[0], local_scs->n_rows);
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
        apply_strided_permutation<VT, IT>(&(local_x_permuted)[vec_idx], &(*local_x)[vec_idx], &(local_scs->new_to_old_idx)[0], local_scs->n_rows, config->block_vec_size);
#endif
    }

    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        // Currently, we fix one sigma. That is, we permute dp and sp exactly the same
        apply_permutation<double, IT>(&(dp_local_x_permuted)[0], &(*dp_local_x)[0], &(dp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
        apply_permutation<float, IT>(&(sp_local_x_permuted)[0], &(*sp_local_x)[0], &(sp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
#ifdef HAVE_HALF_MATH
        apply_permutation<_Float16, IT>(&(hp_local_x_permuted)[0], &(*hp_local_x)[0], &(hp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
#endif
    }

    OnePrecKernelArgs<VT, IT> *one_prec_kernel_args_encoded = new OnePrecKernelArgs<VT, IT>;
    MultiPrecKernelArgs<IT> *multi_prec_kernel_args_encoded = new MultiPrecKernelArgs<IT>;
    void *kernel_args_void_ptr;
    void *comm_args_void_ptr;
    void *cusparse_args_void_ptr;
#ifdef USE_CUSPARSE
    CuSparseArgs *cusparse_args_encoded = new CuSparseArgs;
#endif
    CommArgs<VT, IT> *comm_args_encoded = new CommArgs<VT, IT>;

#ifdef USE_MPI
    int nzr_size = local_context->non_zero_receivers.size();
    int nzs_size = local_context->non_zero_senders.size();

    // Allocate a send buffer for each process we're sending a message to
    int to_send_count;
#ifdef SINGLEVEC_MPI_MODE
    to_send_count = nzr_size;
#endif
#ifdef MULTIVEC_MPI_MODE
    to_send_count = nzr_size * config->block_vec_size;
#endif
#ifdef BULKVEC_MPI_MODE
    to_send_count = nzr_size;
#endif

    VT *to_send_elems[to_send_count];
    
#ifdef SINGLEVEC_MPI_MODE
        for(int i = 0; i < nzr_size; ++i){
            int nz_recver = local_context->non_zero_receivers[i];
            to_send_elems[i] = new VT[local_context->comm_send_idxs[nz_recver].size()];
        }
#endif
#ifdef MULTIVEC_MPI_MODE
    // With multivec mode, we need to replicate the to_send buffers for each vector
    for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
        for(int i = 0; i < nzr_size; ++i){
            int nz_recver = local_context->non_zero_receivers[i];
            to_send_elems[i + vec_idx * nzr_size] = new VT[local_context->comm_send_idxs[nz_recver].size()];
        }
    }
#endif
#ifdef BULKVEC_MPI_MODE
    // With bulkvec mode, the send buffers just need to be larger
        for(int i = 0; i < nzr_size; ++i){
            int nz_recver = local_context->non_zero_receivers[i];
            to_send_elems[i] = new VT[local_context->comm_send_idxs[nz_recver].size() * config->block_vec_size];
        }
#endif
    // Delare MPI requests for non-blocking communication
    int recv_request_buf_size = 0;
    int send_request_buf_size = 0;
#ifdef SINGLEVEC_MPI_MODE
    recv_request_buf_size = nzs_size;
    send_request_buf_size = nzr_size;
#endif
#ifdef MULTIVEC_MPI_MODE
    recv_request_buf_size = nzs_size * config->block_vec_size;
    send_request_buf_size = nzr_size * config->block_vec_size;
#endif
#ifdef BULKVEC_MPI_MODE_MPI_MODE
    recv_request_buf_size = nzs_size;
    send_request_buf_size = nzr_size;
#endif
    MPI_Request *recv_requests = new MPI_Request[recv_request_buf_size];
    MPI_Request *send_requests = new MPI_Request[send_request_buf_size];
#endif

#ifdef __CUDACC__
    // If using cuda compiler, move data to device and assign device pointers
    printf("Moving data to device...\n");
    long n_thread_blocks = (local_scs->n_rows_padded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    config->num_blocks = n_thread_blocks; // Just for ease of results printing later
    config->tpb = THREADS_PER_BLOCK;
    
    // NOTE: Allocating all these pointers out here isn't the cleanest...
    VT *d_x = new VT;
    VT *d_y = new VT;
    ST *d_C = new ST;
    ST *d_n_chunks = new ST;
    IT *d_chunk_ptrs = new IT;
    IT *d_chunk_lengths = new IT;
    IT *d_col_idxs = new IT;
    VT *d_values = new VT;
    ST *d_n_thread_blocks = new ST;

    double *d_x_dp = new double;
    double *d_y_dp = new double;
    ST *d_C_dp = new ST;
    ST *d_n_chunks_dp = new ST;
    IT *d_chunk_ptrs_dp = new IT;
    IT *d_chunk_lengths_dp = new IT;
    IT *d_col_idxs_dp = new IT;
    double *d_values_dp = new double;
    float *d_x_sp = new float;
    float *d_y_sp = new float;
    ST *d_C_sp = new ST;
    ST *d_n_chunks_sp = new ST;
    IT *d_chunk_ptrs_sp = new IT;
    IT *d_chunk_lengths_sp = new IT;
    IT *d_col_idxs_sp = new IT;
    float *d_values_sp = new float;
#ifdef HAVE_HALF_MATH
    _Float16 *d_x_sp = new _Float16;
    _Float16 *d_y_sp = new _Float16;
    ST *d_C_sp = new ST;
    ST *d_n_chunks_sp = new ST;
    IT *d_chunk_ptrs_sp = new IT;
    IT *d_chunk_lengths_sp = new IT;
    IT *d_col_idxs_sp = new IT;
    _Float16  *d_values_sp = new _Float16;
#endif

assign_spmv_kernel_gpu_data<VT>(
    config,
    local_scs,
    dp_local_scs,
    sp_local_scs,
#ifdef HAVE_HALF_MATH
    hp_local_scs,
#endif
    local_y->data(),
    dp_local_y->data(),
    sp_local_y->data(),
#ifdef HAVE_HALF_MATH
    hp_local_y->data(),
#endif
    local_x->data(),
    local_x_permuted.data(),
    dp_local_x->data(),
    dp_local_x_permuted.data(),
    sp_local_x->data(),
    sp_local_x_permuted.data(),
#ifdef HAVE_HALF_MATH
    hp_local_x->data(),
    hp_local_x_permuted.data(),
#endif
    d_x,
    d_y,
    d_C,
    d_n_chunks,
    d_chunk_ptrs,
    d_chunk_lengths,
    d_col_idxs,
    d_values,
    d_n_thread_blocks,
    d_x_dp,
    d_y_dp,
    d_C_dp,
    d_n_chunks_dp,
    d_chunk_ptrs_dp,
    d_chunk_lengths_dp,
    d_col_idxs_dp,
    d_values_dp,
    d_x_sp,
    d_y_sp,
    d_C_sp,
    d_n_chunks_sp,
    d_chunk_ptrs_sp,
    d_chunk_lengths_sp,
    d_col_idxs_sp,
    d_values_sp,
#ifdef HAVE_HALF_MATH
    d_x_sp,
    d_y_sp,
    d_C_sp,
    d_n_chunks_sp,
    d_chunk_ptrs_sp,
    d_chunk_lengths_sp,
    d_col_idxs_sp,
    d_values_sp,
#endif
    n_thread_blocks,
#ifdef USE_CUSPARSE
    cusparse_args_encoded,
#endif
    one_prec_kernel_args_encoded,
    multi_prec_kernel_args_encoded
);

#else
    assign_spmv_kernel_cpu_data<VT, int>(
        config,
        local_context,
        local_scs,
        dp_local_scs,
        sp_local_scs,
#ifdef HAVE_HALF_MATH
        hp_local_scs,
#endif
        local_y->data(),
        dp_local_y->data(),
        sp_local_y->data(),
#ifdef HAVE_HALF_MATH
        hp_local_y->data(),
#endif
        local_x->data(),
        local_x_permuted.data(),
        dp_local_x->data(),
        dp_local_x_permuted.data(),
        sp_local_x->data(),
        sp_local_x_permuted.data(),
#ifdef HAVE_HALF_MATH
        hp_local_x->data(),
        hp_local_x_permuted.data(),
#endif
        one_prec_kernel_args_encoded,
        multi_prec_kernel_args_encoded
    );
#endif

    assign_mpi_args<VT, int>(
        comm_args_encoded,
        comm_args_void_ptr,
        local_context,
#ifdef USE_MPI
        local_scs,
        work_sharing_arr,
        to_send_elems,
        recv_requests,
        send_requests,
        &nzs_size,
        &nzr_size,
#endif
        &my_rank,
        &comm_size
    );

    // Pass args to construct spmv_kernel object
    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        kernel_args_void_ptr = (void*) multi_prec_kernel_args_encoded;
    }
    else{
        kernel_args_void_ptr = (void*) one_prec_kernel_args_encoded;
    }

#ifdef USE_CUSPARSE
    cusparse_args_void_ptr = (void*) cusparse_args_encoded;
#endif

    comm_args_void_ptr = (void*) comm_args_encoded;

    SpmvKernel<VT, IT> spmv_kernel(
        config, 
        kernel_args_void_ptr, 
        cusparse_args_void_ptr,
        comm_args_void_ptr
    );

    // Enter main COMM-spmv-SWAP loop, bench mode
    bool warmup_flag = false;
    if(config->mode == 'b'){
#ifdef __CUDACC__
    cudaEvent_t start, stop, warmup_start, warmup_stop;
    cudaEventCreate(&warmup_start);
    cudaEventCreate(&warmup_stop);
#endif

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Warm-up
        bool warmup_flag = true;

#ifdef __CUDACC__
    cudaEventRecord(warmup_start, 0);
    cudaDeviceSynchronize();
#else
        double begin_warm_up_loop_time, end_warm_up_loop_time;
        
#ifdef USE_MPI
        begin_warm_up_loop_time = MPI_Wtime();     
#else
        begin_warm_up_loop_time = getTimeStamp();
#endif
#endif
        for(int k = 0; k < WARM_UP_REPS; ++k){
#ifdef USE_MPI
            begin_communicate_halo_elements<VT, IT>(config, &spmv_kernel);
            finish_communicate_halo_elements<VT, IT>(config, &spmv_kernel);
#endif
            spmv_kernel.execute(warmup_flag);

#ifdef USE_MPI
            if(config->ba_synch)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

#ifdef __CUDACC__
        float warmup_runtime;
        cudaEventRecord(warmup_stop, 0);
        cudaEventSynchronize(warmup_stop);
        cudaEventElapsedTime(&warmup_runtime, warmup_start, warmup_stop);
        std::cout << "warm up time: " << warmup_runtime * MILLI_TO_SEC << std::endl;
#else

#ifdef USE_MPI
        end_warm_up_loop_time = MPI_Wtime();
#else
        end_warm_up_loop_time = getTimeStamp();
#endif
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        if(my_rank == 0)
            std::cout << "warm up time: " << end_warm_up_loop_time - begin_warm_up_loop_time << std::endl;  
#endif
#endif

#ifdef __CUDACC__
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#else
        double begin_bench_loop_time, end_bench_loop_time = 0.0;
#endif

        float runtime;

        // Initialize number of repetitions for actual benchmark
        int n_iter = 2;
        warmup_flag = false;

#ifndef __CUDACC__
#ifdef USE_LIKWID
        register_likwid_markers(config);
#endif
#endif

        if(config->comm_halos){
#ifdef USE_MPI
            do{
                MPI_Barrier(MPI_COMM_WORLD);
                begin_bench_loop_time = MPI_Wtime();
                for(int k=0; k<n_iter; ++k) {
                    begin_communicate_halo_elements<VT, IT>(config, &spmv_kernel);
                    finish_communicate_halo_elements<VT, IT>(config, &spmv_kernel);
                    spmv_kernel.execute(warmup_flag);
                    if(config->ba_synch)
                        MPI_Barrier(MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                n_iter = n_iter*2;
                runtime = MPI_Wtime() - begin_bench_loop_time;
            } while (runtime < config->bench_time);
            n_iter = n_iter/2;
#else
    printf("ERROR: Cannot communicate halo elements.\n \
        Validate that either USE_MPI = 0 and comm_halos = 0,\n \
        or that USE_MPI = 1.\n");
    exit(1);
#endif
        }
        else if(!config->comm_halos){
            do{
#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
                begin_bench_loop_time = MPI_Wtime();
#else
#ifdef __CUDACC__
                cudaEventRecord(start);
#else
                begin_bench_loop_time = getTimeStamp();
#endif
#endif
                
                for(int k=0; k<n_iter; ++k) {
                    spmv_kernel.execute(warmup_flag);
#ifdef USE_MPI
                    if(config->ba_synch)
                        MPI_Barrier(MPI_COMM_WORLD);
#endif
                }
#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                n_iter = n_iter*2;
#ifdef USE_MPI
                runtime = MPI_Wtime() - begin_bench_loop_time;
#else
#ifdef __CUDACC__
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&runtime, start, stop);
#else
                runtime = getTimeStamp() - begin_bench_loop_time;
#endif
#endif
#ifdef __CUDACC__
            } while (runtime * MILLI_TO_SEC < config->bench_time);
#else
            } while (runtime < config->bench_time);
#endif

            n_iter = n_iter/2;
        }
        r->n_calls = n_iter;
#ifdef __CUDACC__
        r->duration_total_s = runtime * MILLI_TO_SEC;
#else
        r->duration_total_s = runtime;
#endif
        r->duration_kernel_s = r->duration_total_s / r->n_calls;
        r->perf_gflops = (double)local_context->total_nnz * 2.0 * config->block_vec_size
                            / r->duration_kernel_s
                            / 1e9;                   // Only count usefull flops
    }
    else if(config->mode == 's') { // Enter main COMM-spmv-SWAP loop, solve mode
        for (int i = 0; i < config->n_repetitions; ++i)
        {
#ifdef DEBUG_MODE_FINE
            int test_rank = 1;
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(my_rank == test_rank){
                SanityChecker::check_vectors_before_comm<VT, IT>(config, local_scs, local_context, &spmv_kernel, my_rank);
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

#ifdef DEBUG_MODE
            if(my_rank == 0){printf("Communicating halo elements\n");}
#endif
#ifdef USE_MPI
            begin_communicate_halo_elements<VT, IT>(config, &spmv_kernel);
            finish_communicate_halo_elements<VT, IT>(config, &spmv_kernel);
#endif
#ifdef DEBUG_MODE
            if(my_rank == 0){printf("Performing SpM(M)V kernel\n");}
#endif
            
#ifdef DEBUG_MODE_FINE
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(my_rank == test_rank){
                SanityChecker::check_vectors_after_comm<VT, IT>(config, local_scs, local_context, &spmv_kernel, my_rank);
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

            spmv_kernel.execute();

#ifdef DEBUG_MODE_FINE
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(my_rank == test_rank){
                SanityChecker::check_vectors_after_spmv<VT, IT>(config, local_scs, local_context, &spmv_kernel, my_rank);
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

#ifdef DEBUG_MODE
            if(my_rank == 0){printf("Swapping X <-> Y\n");}
#endif
            // TODO: Guard against rectangular matrices
            spmv_kernel.swap_local_vectors();


#ifdef DEBUG_MODE_FINE
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(my_rank == test_rank){
                SanityChecker::check_vectors_after_swap<VT, IT>(config, local_scs, local_context, &spmv_kernel, my_rank);
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

#ifdef USE_MPI
#ifdef DEBUG_MODE
            if(my_rank == 0){printf("Synchronizing MPI processes");}
#endif
            if(config->ba_synch)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        copy_back_result(
            config,
            local_scs,
            dp_local_scs,
            sp_local_scs,
#ifdef HAVE_HALF_MATH
            hp_local_scs,
#endif
            local_y,
            spmv_kernel.local_x,
            spmv_kernel.dp_local_x,
            spmv_kernel.sp_local_x,
#ifdef HAVE_HALF_MATH
            spmv_kernel.hp_local_x,
#endif
            dp_local_x_permuted.data(),
            sp_local_x_permuted.data(),
#ifdef HAVE_HALF_MATH
            hp_local_x_permuted.data(),
#endif
            local_context
        );
    }

    // Delete the allocated space for each other process send buffers
#ifdef USE_MPI
    for(int i = 0; i < to_send_count; ++i){
        delete[] to_send_elems[i];
    }
#ifndef BULKVEC_MPI_MODE
    for(int i = 0; i < local_context->non_zero_senders.size(); ++i){
        MPI_Type_free(&local_context->strided_recv_types[i]);
    }
#else
    for(int i = 0; i < local_context->non_zero_senders.size(); ++i){
        MPI_Type_free(&local_context->bulk_recv_types[i]);
    }
#endif

#endif

    double mem_matrix_b =
            (double)sizeof(VT) * local_scs->n_elements     // values
        + (double)sizeof(IT) * local_scs->n_chunks       // chunk_ptrs
        + (double)sizeof(IT) * local_scs->n_chunks       // chunk_lengths
        + (double)sizeof(IT) * local_scs->n_elements;    // col_idxs

    double mem_x_b = (double)sizeof(VT) * local_scs->n_cols;
    double mem_y_b = (double)sizeof(VT) * local_scs->n_rows_padded;
    double mem_b   = mem_matrix_b + mem_x_b + mem_y_b;

    r->mem_mb   = mem_b / 1e6;
    r->mem_m_mb = mem_matrix_b / 1e6;
    r->mem_x_mb  = mem_x_b / 1e6;
    r->mem_y_mb  = mem_y_b / 1e6;

    r->n_rows = local_scs->n_rows;
    r->n_cols = local_scs->n_cols;
    r->nnz    = local_scs->nnz;

    if(config->value_type == "dp")
        r->value_type_str = "double";
    
    else if(config->value_type == "sp")
        r->value_type_str = "float";
    else
        r->value_type_str = "half";

    r->index_type_str = typeid(IT).name();
    r->value_type_size = sizeof(VT);
    r->index_type_size = sizeof(IT);

    // TODO: ????
    // r->was_matrix_sorted = local_scs->is_sorted;
    r->was_matrix_sorted = 1;

    r->fill_in_percent = ((double)local_scs->n_elements / local_scs->nnz - 1.0) * 100.0;
    r->C               = local_scs->C;
    r->sigma           = local_scs->sigma;
    r->beta = (double)local_scs->nnz / local_scs->n_elements;

    // Only relevant for adaptive precision
    r->dp_nnz = dp_local_scs->nnz;
    r->sp_nnz = sp_local_scs->nnz;
#ifdef HAVE_HALF_MATH
    r->hp_nnz = hp_local_scs->nnz;
#endif

    r->dp_beta = (double)dp_local_scs->nnz / dp_local_scs->n_elements;
    if(dp_local_scs->n_elements == 0)
        r->dp_beta = 0;
    r->sp_beta = (double)sp_local_scs->nnz / sp_local_scs->n_elements;
    if(sp_local_scs->n_elements == 0)
        r->sp_beta = 0;
#ifdef HAVE_HALF_MATH
    r->hp_beta = (double)hp_local_scs->nnz / hp_local_scs->n_elements;
    if(hp_local_scs->n_elements == 0)
        r->hp_beta = 0;
#endif

// TODO
// #ifdef USE_CUSPARSE
//     // destroy matrix/vector descriptors
//     cusparseDestroySpMat(matA);
//     cusparseDestroyDnVec(vecX);
//     cusparseDestroyDnVec(vecY);
//     cusparseDestroy(handle);
// #endif

// TODO: Memcheck doesn't like this for some reason
// #ifdef __CUDACC__
//     if(config->value_type == "ap"){
//         cudaFree(d_x_hp);
//         cudaFree(d_y_hp);
//         cudaFree(d_C_hp);
//         cudaFree(d_n_chunks_hp);
//         cudaFree(d_chunk_ptrs_hp);
//         cudaFree(d_chunk_lengths_hp);
//         cudaFree(d_col_idxs_hp);
//         cudaFree(d_values_hp);
//         cudaFree(d_x_lp);
//         cudaFree(d_y_lp);
//         cudaFree(d_C_lp);
//         cudaFree(d_n_chunks_lp);
//         cudaFree(d_chunk_ptrs_lp);
//         cudaFree(d_chunk_lengths_lp);
//         cudaFree(d_col_idxs_lp);
//         cudaFree(d_values_lp);
//     }
//         cudaFree(d_x);
//         cudaFree(d_y);
//         cudaFree(d_C);
//         cudaFree(d_n_chunks);
//         cudaFree(d_chunk_ptrs);
//         cudaFree(d_chunk_lengths);
//         cudaFree(d_col_idxs);
//         cudaFree(d_values);
// #endif

    delete comm_args_encoded;
    delete one_prec_kernel_args_encoded;
    delete multi_prec_kernel_args_encoded;
}

/**
    @brief Gather results (either result of computation, or benchmark metrics) to the root MPI process
    @param *config : struct to initialze default values and user input
    @param *r : a Result struct, in which results of the benchmark are stored
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_x_mkl_copy : copy of local RHS vector used for validation against MKL
    @param *local_y : Process-local results vector, instance of SimpleDenseMatrix class
*/
template <typename VT, typename IT>
void gather_results(
    Config *config,
    ContextData<IT> *local_context,
    Result<VT, IT> *r,
    IT *work_sharing_arr,
    std::vector<VT> *local_x_mkl_copy,
    std::vector<VT> *local_y,
    int my_rank,
    int comm_size,
    int *metis_inv_perm
){

    IT num_local_rows = local_context->num_local_rows;

    if(config->mode == 'b'){

        double *perfs_from_procs_arr = new double[comm_size];
        unsigned long *nnz_per_procs_arr = new unsigned long[comm_size];
        unsigned long *dp_nnz_per_procs_arr = new unsigned long[comm_size];
        unsigned long *sp_nnz_per_procs_arr = new unsigned long[comm_size];
#ifdef HAVE_HALF_MATH
        unsigned long *hp_nnz_per_procs_arr = new unsigned long[comm_size];
#endif
#ifdef USE_MPI

        MPI_Gather(&(r->perf_gflops),
                1,
                MPI_DOUBLE,
                perfs_from_procs_arr,
                1,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

        MPI_Gather(&(r->dp_nnz),
                1,
                MPI_UNSIGNED_LONG,
                dp_nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);

        MPI_Gather(&(r->sp_nnz),
                1,
                MPI_UNSIGNED_LONG,
                sp_nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);

#ifdef HAVE_HALF_MATH
        MPI_Gather(&(r->hp_nnz),
                1,
                MPI_UNSIGNED_LONG,
                hp_nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);
#endif

        MPI_Gather(&(r->nnz),
                1,
                MPI_UNSIGNED_LONG,
                nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);

#else
        perfs_from_procs_arr[0] = r->perf_gflops;
        nnz_per_procs_arr[0] = r->nnz;

        r->cumulative_dp_nnz = r->dp_nnz;
        r->cumulative_sp_nnz = r->sp_nnz;
#ifdef HAVE_HALF_MATH
        r->cumulative_hp_nnz = r->hp_nnz;
#endif
        // NOTE: AP stuff isn't relevant for MPI right now...
        r->total_dp_percent = (r->cumulative_dp_nnz / (double)r->total_nnz) * 100.0;
        r->total_sp_percent = (r->cumulative_sp_nnz / (double)r->total_nnz) * 100.0;
#ifdef HAVE_HALF_MATH
        r->total_hp_percent = (r->cumulative_hp_nnz / (double)r->total_nnz) * 100.0;
#endif

#endif
        // NOTE: Garbage values for all but root process
        r->perfs_from_procs = std::vector<double>(perfs_from_procs_arr, perfs_from_procs_arr + comm_size);
        r->nnz_per_proc = std::vector<unsigned long>(nnz_per_procs_arr, nnz_per_procs_arr + comm_size);

        delete[] perfs_from_procs_arr;
        delete[] nnz_per_procs_arr;

    }
    else if(config->mode == 's'){
        r->x_out = (*local_x_mkl_copy);
        r->y_out = (*local_y); // NOTE: Is this true even with SCS?

#ifdef DEBUG_MODE_FINE
        printf("local_y (size %i)= \n", local_y->size());
        for(int i = 0; i < local_y->size(); ++i){
            printf("%f, \n", (*local_y)[i]);
        }
        printf("]\n");
#endif

        if (config->validate_result){
#ifdef USE_MPI
            MPI_Datatype MPI_ELEM_TYPE = get_mpi_type<VT>();

            // TODO: You really need this size on EVERY process?
            std::vector<VT> total_uspmv_result(local_context->total_n_rows * config->block_vec_size, VT{});
            std::vector<VT> total_x(local_context->total_n_rows * config->block_vec_size, VT{});

            IT counts_arr[comm_size];
            IT displacement_arr_bk[comm_size];

            for(IT i = 0; i < comm_size; ++i){
                counts_arr[i] = IT{};
                displacement_arr_bk[i] = IT{};
            }
            
            for (IT i = 0; i < comm_size; ++i){
                counts_arr[i] = work_sharing_arr[i + 1] - work_sharing_arr[i];
                displacement_arr_bk[i] = work_sharing_arr[i]; // TODO: <- so can't you just replace with WSA?
            }

#ifdef DEBUG_MODE_FINE
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            int test_rank = 1;
            if(my_rank == test_rank){
                SanityChecker::check_vectors_before_gather<VT, IT>(config, local_context, r, my_rank);
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

            for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
                // Gather all y_vector results to root
                MPI_Gatherv(&(r->y_out)[vec_idx * num_local_rows],
                            num_local_rows,
                            MPI_ELEM_TYPE,
                            &total_uspmv_result[vec_idx * r->total_rows],
                            counts_arr,
                            displacement_arr_bk,
                            MPI_ELEM_TYPE,
                            0,
                            MPI_COMM_WORLD);

                // Gather all x_vector copies to root for mkl validation
                MPI_Gatherv(&(r->x_out)[vec_idx * num_local_rows],
                            num_local_rows,
                            MPI_ELEM_TYPE,
                            &total_x[vec_idx * r->total_rows],
                            counts_arr,
                            displacement_arr_bk,
                            MPI_ELEM_TYPE,
                            0,
                            MPI_COMM_WORLD);
            }

            // If we're verifying results, assign total vectors to Result object
            // NOTE: Garbage values for all but root process
            r->total_x = total_x;
            if(config->seg_method == "seg-metis"){
                r->total_uspmv_result.resize(total_uspmv_result.size());
                for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
                    if(my_rank == 0){
                        // Permute back results for comparison against MKL
                        apply_permutation(&(r->total_uspmv_result.data()[vec_idx * local_context->total_n_rows]), &(total_uspmv_result.data()[vec_idx * local_context->total_n_rows]), metis_inv_perm, local_context->total_n_rows);
                    }
                }
            }
            else{
                r->total_uspmv_result = total_uspmv_result;
            }
            
#else

            r->total_x = *local_x_mkl_copy;
            // r->total_uspmv_result.resize(r->y_out.size());
            r->total_uspmv_result = r->y_out;
#endif
            // if(config->seg_method == "seg-metis"){
// #ifdef USE_MPI
// #ifdef DEBUG_MODE_FINE
//             if(my_rank == 0){
//                 printf("total result before perm back: [");
//                 for(int i = 0; i < total_uspmv_result.size(); ++i){
//                     printf("%f, ", total_uspmv_result[i]);
//                 }
//                 printf("]\n");
                
//             }
// #endif
//             for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
//                 if(my_rank == 0){
//                     // Permute back results for comparison against MKL
//                     apply_permutation(&(r->total_uspmv_result.data()[vec_idx * local_context->total_n_rows]), &(total_uspmv_result.data()[vec_idx * local_context->total_n_rows]), metis_inv_perm, local_context->total_n_rows);
//                 }
//             }
// #ifdef DEBUG_MODE_FINE
//             if(my_rank == 0){
//                 printf("total result after perm back: [");
//                 for(int i = 0; i < r->total_uspmv_result.size(); ++i){
//                     printf("%f, ", r->total_uspmv_result[i]);
//                 }
//                 printf("]\n");
                
//             }
// #endif
// #endif
            // }
            // else{
            //     r->total_uspmv_result = r->y_out;
            // }
        }

#ifdef DEBUG_MODE_FINE
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        // NOTE: Only root process has useful data on it 
        if(my_rank == 0){
            SanityChecker::check_vectors_after_gather<VT, IT>(config, local_context, r, my_rank);
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif        
    }
}

/** 
    @brief Perform initialization routines on relevant structures
    @param *local_scs : pointer to process-local scs struct
    @param *(dp/sp/hp)_local_scs : precision-specific copies, used in adaptive precision
    @param *local_context : struct containing local_scs + communication information
    @param *total_mtx : global mtx struct
    @param *config : struct to initialze default values and user input
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *metis_ : various pointers needed by metis graph partitioner
*/
template<typename VT, typename IT>
void init_local_structs(
    ScsData<VT, IT> *local_scs,
    ScsData<double, IT> *dp_local_scs,
    ScsData<float, IT> *sp_local_scs,
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> *hp_local_scs,
#endif
    ContextData<IT> *local_context,
    MtxData<VT, IT> *total_mtx,
    Config *config, // shouldn't this be const?
    IT *work_sharing_arr,
    int my_rank,
    int comm_size,
    int* metis_part = NULL,
    int* metis_perm = NULL,
    int* metis_inv_perm = NULL)
{
    MtxData<VT, IT> *local_mtx = new MtxData<VT, IT>;

    local_context->total_nnz = total_mtx->nnz;
    // extract matrix mean (and give to x-vector if option chosen at cli)
    extract_matrix_min_mean_max(total_mtx, config, my_rank);

#ifdef USE_MPI
    MPI_Bcast(&(local_context->total_nnz), 1, MPI_INT, 0, MPI_COMM_WORLD); // <- necessary?
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Segmenting and sending work to other processes.\n");}
#endif

    seg_and_send_work_sharing_arr<VT, IT>(config, total_mtx, work_sharing_arr, my_rank, comm_size, metis_part, metis_perm, metis_inv_perm);

    seg_and_send_matrix_data<VT, IT>(config, total_mtx, local_mtx, work_sharing_arr, my_rank, comm_size);

    localize_row_idx<VT, IT>(local_mtx);
#else
    local_mtx = total_mtx;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Converting COO matrix to SELL-C-SIG and permuting locally (NOTE: rows only, i.e. nonsymetrically).\n");}
#endif

    // If desired, scale the one-precision matrix
    if(
        (config->value_type == "dp" || 
        config->value_type == "sp" || 
        config->value_type == "hp") && 
        config->equilibrate)
    {
        equilibrate_matrix<VT, IT>(local_mtx);
    }

    // convert local_mtx to local_scs (and permute rows if sigma > 1)
    convert_to_scs<VT, VT, IT>(local_mtx, config->chunk_size, config->sigma, local_scs, NULL, work_sharing_arr, my_rank);


#ifdef DEBUG_MODE_FINE
    SanityChecker::check_perm_vectors<VT, IT>(local_scs);
#endif

    // Only used for adaptive precision
    MtxData<double, int> *dp_local_mtx = new MtxData<double, int>;
    MtxData<float, int> *sp_local_mtx = new MtxData<float, int>;
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *hp_local_mtx = new MtxData<_Float16, int>;
#endif

    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        std::vector<VT> largest_row_elems;//(local_mtx->n_cols, 0.0);
        std::vector<VT> largest_col_elems;//(local_mtx->n_cols, 0.0);

        // Scale local_mtx *and save the largest row and col elements*
        if(config->equilibrate){
            extract_largest_row_elems<VT, IT>(local_mtx, &largest_row_elems);
            scale_matrix_rows<VT, IT>(local_mtx, &largest_row_elems);

            extract_largest_col_elems<VT, IT>(local_mtx, &largest_col_elems);
            scale_matrix_cols<VT, IT>(local_mtx, &largest_col_elems);
        }

        // Pass largest row and col elements to precision partitioner
        partition_precisions<VT,IT>(
            config, 
            local_mtx, 
            dp_local_mtx, 
            sp_local_mtx,
#ifdef HAVE_HALF_MATH
            hp_local_mtx,
#endif
            &largest_row_elems, 
            &largest_col_elems, 
            my_rank
        );

        // We permute the lower precision struct(s) in the exact same way as the higher precision one
        if(config->value_type == "ap[dp_sp]"){
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
        
            // Empty struct, just pass through convert_to_scs for technical reasons
#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, NULL, work_sharing_arr, my_rank);
#endif
        }
        else if (config->value_type == "ap[dp_hp]"){
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
#endif
            // Empty struct, just pass through convert_to_scs for technical reasons
            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, NULL, work_sharing_arr, my_rank);
        
        }
        else if (config->value_type == "ap[sp_hp]"){
            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, NULL, work_sharing_arr, my_rank);
        
#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, &(sp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
#endif
       
            // Empty struct, just pass through convert_to_scs for technical reasons
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

        }
        else if (config->value_type == "ap[dp_sp_hp]"){
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
        
#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
#endif
        }



#ifdef OUTPUT_SPARSITY
        printf("Writing sparsity pattern to output file.\n");
        std::string file_out_name;
        file_out_name = "dp_local_scs";
        dp_local_scs->write_to_mtx_file(my_rank, file_out_name);
        file_out_name = "sp_local_scs";
        sp_local_scs->write_to_mtx_file(my_rank, file_out_name);
#ifdef HAVE_HALF_MATH
        file_out_name = "hp_local_scs";
        hp_local_scs->write_to_mtx_file(my_rank, file_out_name);
#endif
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        exit(0);
#endif
    }

    if (config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){
#ifdef OUTPUT_SPARSITY
        printf("Writing sparsity pattern to output file.\n");
        std::string file_out_name;
        file_out_name = "local_scs";
        local_scs->write_to_mtx_file(my_rank, file_out_name);
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        exit(0);
#endif
    }   

#ifdef USE_MPI
    // TODO: is an array of vectors better?
    // Vector of vecs, Keep track of which remote columns come from which processes
    std::vector<std::vector<IT>> communication_recv_idxs;
    std::vector<std::vector<IT>> communication_send_idxs;
    std::vector<IT> non_zero_receivers;
    std::vector<IT> non_zero_senders;
    std::vector<std::vector<IT>> send_tags;
    std::vector<std::vector<IT>> recv_tags;
    std::vector<IT> recv_counts_cumsum(comm_size + 1, 0);
    std::vector<IT> send_counts_cumsum(comm_size + 1, 0);
    std::vector<MPI_Datatype> strided_recv_types;
    std::vector<MPI_Datatype> bulk_recv_types;

    // Main routine for collecting all sending and receiving information!
    collect_comm_info<VT, IT>(
        config, 
        local_scs, 
        work_sharing_arr, 
        &communication_recv_idxs,
        &communication_send_idxs,
        &non_zero_receivers,
        &non_zero_senders,
        &send_tags,
        &recv_tags,
        &recv_counts_cumsum,
        &send_counts_cumsum,
        &strided_recv_types,
        &bulk_recv_types,
        my_rank,
        comm_size
    );
    
    // Collect all our hard work to single structure for convenience
    // NOTE: not used at all in the no-mpi case
    local_context->comm_send_idxs = communication_send_idxs;
    local_context->comm_recv_idxs = communication_recv_idxs;
    local_context->non_zero_receivers = non_zero_receivers;
    local_context->non_zero_senders = non_zero_senders;
    local_context->send_tags = send_tags;
    local_context->recv_tags = recv_tags;
    local_context->recv_counts_cumsum = recv_counts_cumsum;
    local_context->send_counts_cumsum = send_counts_cumsum;
    local_context->num_local_rows = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
    local_context->strided_recv_types = strided_recv_types;
    local_context->bulk_recv_types = bulk_recv_types;
#else
    local_context->num_local_rows = local_scs->n_rows;
#endif
    local_context->scs_padding = (IT)(local_scs->n_rows_padded - local_scs->n_rows);

    // TODO: For symmetric permutation of matrix data, validate this works as intended
    permute_scs_cols(local_scs, &(local_scs->old_to_new_idx)[0]);

    // TODO: How to permute columns with here?
    // if (config->value_type == "ap"){
    //     // Permute column indices the same as the original scs struct
    //     // But rows are permuted differently (i.e. within the convert_to_scs routine)
    //     // permute_scs_cols(hp_local_scs, &(hp_local_scs->old_to_new_idx)[0]);
    //     // permute_scs_cols(lp_local_scs, &(hp_local_scs->old_to_new_idx)[0]);
    //     for(int i = 0; i < hp_local_scs->n_elements; ++i){
    //         std::cout << "hp_local_scs->col_idxs[" << i << "] = " << hp_local_scs->col_idxs[i] << std::endl;
    //     }
    //     for(int i = 0; i < lp_local_scs->n_elements; ++i){
    //         std::cout << "lp_local_scs->col_idxs[" << i << "] = " << lp_local_scs->col_idxs[i] << std::endl;
    //     }

    //     permute_scs_cols(hp_local_scs, &(hp_local_scs->old_to_new_idx)[0]);
    //     permute_scs_cols(lp_local_scs, &(lp_local_scs->old_to_new_idx)[0]);

    //     for(int i = 0; i < hp_local_scs->n_elements; ++i){
    //         std::cout << "hp_local_scs->col_idxs[" << i << "] = " << hp_local_scs->col_idxs[i] << std::endl;
    //     }
    //     for(int i = 0; i < lp_local_scs->n_elements; ++i){
    //         std::cout << "lp_local_scs->col_idxs[" << i << "] = " << lp_local_scs->col_idxs[i] << std::endl;
    //     }
    // }

}

/**
    @brief The main harness for the spmv kernel, in which we:
        1. Segment and distribute the needed structs to each MPI process (init_local_structs),
        2. Benchmark the selected spmv kernel (bench_spmv),
        3. Gather benchmark results to the root MPI process (gather_results).
    @param *total_mtx : global mtx struct, read from a .mtx file (or generated with ScaMaC TODO)
    @param *config : struct to initialze default values and user input
    @param *r : a Result struct, in which results of the benchmark/computation are stored
*/
template <typename VT, typename IT>
void compute_result(
    MtxData<VT, IT> *total_mtx,
    Config *config,
    Result<VT, IT> *r,
    int my_rank,
    int comm_size,
    int *metis_part,
    int *metis_perm,
    int *metis_inv_perm
)
{
    // TODO: bring back matrix stats!
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    ScsData<VT, IT> local_scs;

    // Declare local structs on each process
    ScsData<double, IT> dp_local_scs;
    ScsData<float, IT> sp_local_scs;
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> hp_local_scs;
#endif

    ContextData<IT> local_context;
    local_context.total_n_rows = total_mtx->n_rows;
#ifdef USE_MPI
    MPI_Bcast(&(local_context.total_n_rows), 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    // Used for distributed work sharing
    // Allocate space for work sharing array
    IT work_sharing_arr[comm_size + 1];
    work_sharing_arr[0] = 0; // Initialize first element, since it's used always

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Init local structures.\n");}
#endif

    init_local_structs<VT, IT>(
        &local_scs,
        &dp_local_scs,
        &sp_local_scs,
#ifdef HAVE_HALF_MATH
        &hp_local_scs,
#endif
        &local_context, 
        total_mtx,
        config, 
        work_sharing_arr, 
        my_rank, 
        comm_size,
        metis_part,
        metis_perm,
        metis_inv_perm
    );



// TODO: Make more uniform! Can do away with SimpleDenseMatrix class... lazy
#ifdef USE_MPI
    int per_vector_padding = std::max(local_context.scs_padding, (local_context.recv_counts_cumsum).back());
#else
    int per_vector_padding = local_context.scs_padding;
#endif

    local_context.per_vector_padding = per_vector_padding;
    local_context.padded_vec_size = per_vector_padding + local_context.num_local_rows;

#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
    DenseMatrixColMaj<VT> local_x(local_scs.n_rows, config->block_vec_size, per_vector_padding);
    DenseMatrixColMaj<VT> local_y(local_scs.n_rows, config->block_vec_size, per_vector_padding);
#else
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
    DenseMatrixRowMaj<VT> local_x(local_scs.n_rows, config->block_vec_size, per_vector_padding);
    DenseMatrixRowMaj<VT> local_y(local_scs.n_rows, config->block_vec_size, per_vector_padding);
#else
    // Declare local vectors to be used
    SimpleDenseMatrix<VT, IT> local_x(&local_context, config);
    SimpleDenseMatrix<VT, IT> local_y(&local_context, config);
#endif
#endif

    // TODO: clean this up too
    // Initialize local_x and y, either randomly, with default values defined in classes_structs.hpp
    local_x.init('x', config, local_scs.n_rows, per_vector_padding);
    local_y.init('y', config, local_scs.n_rows, per_vector_padding);
    
#ifdef DEBUG_MODE_FINE
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    int test_rank = 1;
    if(my_rank == test_rank){
        SanityChecker::check_vector_padding<VT, IT>(config, local_scs, local_context, local_x);
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

    // Must be declared, but only used for mixed precision case
    // TODO: not efficient for storage, but used later for mp interop
    SimpleDenseMatrix<double, IT> dp_local_x(&local_context, config);
    SimpleDenseMatrix<float, IT> sp_local_x(&local_context, config);
#ifdef HAVE_HALF_MATH
    SimpleDenseMatrix<_Float16, IT> hp_local_x(&local_context, config);
#endif

    // NOTE: a low precision y vector is needed for swapping with low precision x
    SimpleDenseMatrix<double, IT> dp_local_y(&local_context, config);
    SimpleDenseMatrix<float, IT> sp_local_y(&local_context, config);
#ifdef HAVE_HALF_MATH
    SimpleDenseMatrix<_Float16, IT> hp_local_y(&local_context, config);
#endif

    copy_data_to_ap_vectors(
        &dp_local_x,
        &sp_local_x,
#ifdef HAVE_HALF_MATH
        &hp_local_x,
#endif
        local_x.vec.data(),
        local_x.vec.size(),
        &dp_local_y,
        &sp_local_y,
#ifdef HAVE_HALF_MATH
        &hp_local_y,
#endif
        local_y.vec.data(),
        local_y.vec.size()
    );

    // Copy contents of local_x for output, and validation against mkl
    // std::vector<VT> local_x_copy = local_x.vec;
    // We want this to be held column-wise always, at least for accuracy validation
    // TODO: Allow to be row-wise for performance comparison against MKL
    // TODO: Ugly. No raw loops!
    std::vector<VT> local_x_mkl_copy(local_scs.n_rows * config->block_vec_size);


// Need to take care to skip padding elements, since this is what is passed to MKL
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
    int shift = 0;
    for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
        for(int i = 0; i < local_scs.n_rows; ++i){
            // NOTE: Basically a permutation, so that local_x_mkl_copy is held columnwise
            // TODO: Easier for validation, but should eventually when comparing performance
            // local_x_mkl_copy[i * config->block_vec_size + j] = local_x.vec[i + local_scs.n_rows * j];
            local_x_mkl_copy[(i + local_scs.n_rows * vec_idx) + shift] = local_x.vec[i * config->block_vec_size + vec_idx];
        }
    }
    // std::swap(local_x_mkl_copy, local_x.vec);

#endif
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
    int vec_idx = 0;
    int shift = 0;
    for(int i = 0; i < local_scs.n_rows * config->block_vec_size; ++i){
        if(vec_idx < local_scs.n_rows){
            local_x_mkl_copy[i] = local_x.vec[i + shift];
        }

        if(vec_idx == local_scs.n_rows - 1){
            vec_idx = 0;
            shift += per_vector_padding;
            continue;
        }

        ++vec_idx;
    } 
#endif

#ifdef DEBUG_MODE_FINE
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if(my_rank == test_rank){
        SanityChecker::check_local_x_vectors<VT, IT>(local_x_mkl_copy, local_x);
    }


#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter bench_spmv.\n");}
#endif
    bench_spmv<VT, IT>(
        config,
        &local_scs,
        &dp_local_scs,
        &sp_local_scs,
#ifdef HAVE_HALF_MATH
        &hp_local_scs,
#endif
        &local_context,
        work_sharing_arr,
        &local_y.vec,
        &dp_local_y.vec,
        &sp_local_y.vec,
#ifdef HAVE_HALF_MATH
        &hp_local_x.vec,
#endif
        &local_x.vec,
        &dp_local_x.vec,
        &sp_local_x.vec,
#ifdef HAVE_HALF_MATH
        &hp_local_x.vec,
#endif
        r,
        my_rank,
        comm_size
    );


#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete bench_spmv.\n");}
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Gather results to root process.\n");}
#endif

    gather_results(config, &local_context, r, work_sharing_arr, &local_x_mkl_copy, &(local_y.vec), my_rank, comm_size, metis_inv_perm);
}

/**
    @brief Entry point for program functionality
    @param *config : struct to initialze default values and user input
*/
void standalone_bench(
    Config config,
    std::string matrix_file_name,
    int my_rank,
    int comm_size,
    double begin_main_time
){
    MtxData<double, int> total_mtx;
    // Replicate structs for each precision
    MtxData<double, int> total_dp_mtx;
    MtxData<float, int> total_sp_mtx;
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> total_hp_mtx;
#endif

    Result<double, int> r_dp;
    Result<float, int> r_sp;
#ifdef HAVE_HALF_MATH
    Result<_Float16, int> r_hp;
#endif

    // The .mtx file is read only by the root process
    if(my_rank == 0){
#ifdef DEBUG_MODE
        if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
        read_mtx(matrix_file_name, config, &total_mtx, my_rank);

        if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
            // copy to total_dp_mtx
            total_dp_mtx.copy(total_mtx);
            r_dp.total_nnz = total_dp_mtx.nnz;
            r_dp.total_rows = total_dp_mtx.n_rows;
        }
        else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
            // copy to total_sp_mtx
            total_sp_mtx.copy(total_mtx);
            r_sp.total_nnz = total_sp_mtx.nnz;
            r_sp.total_rows = total_sp_mtx.n_rows;
        }
        else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
            // copy to total_hp_mtx
            total_hp_mtx.copy(total_mtx);
            r_hp.total_nnz = total_hp_mtx.nnz;
            r_hp.total_rows = total_hp_mtx.n_rows;
#else
            if(my_rank == 0){
                printf("ERROR: Cannot read matrix into HP struct. HAVE_HALF_MATH not defined.\n");
                exit(1);
            }
#endif
        }
        else{
            if(my_rank == 0){
                printf("ERROR: value_type not known.\n");
                exit(1);
            }
        }
    }
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif

    // Used with METIS library, always initialized for convenience
    // Allocate global permutation vectors
    int *metis_part = nullptr;
    int *metis_perm = nullptr;
    int *metis_inv_perm = nullptr;

    // TODO: Guard, since this is only on root process?
#ifdef USE_MPI
    if(config.seg_method == "seg-metis"){
        metis_part = new int[total_mtx.n_rows];
        metis_perm = new int[total_mtx.n_rows];
        for(int i = 0; i < total_mtx.n_rows; ++i){
            metis_perm[i] = i;
        }
        metis_inv_perm = new int[total_mtx.n_rows];
    }
#endif

    // What taks place in this routine depends on "config.mode", i.e. the "result" in
    // "compute_result" is either a measure of performance, or an output vector y to validate
    if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
        compute_result<double, int>(&total_dp_mtx, &config, &r_dp, my_rank, comm_size, metis_part, metis_perm, metis_inv_perm);
    }
    else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
        compute_result<float, int>(&total_sp_mtx, &config, &r_sp, my_rank, comm_size, metis_part, metis_perm, metis_inv_perm);
    }
    else if (config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
        compute_result<_Float16, int>(&total_hp_mtx, &config, &r_hp, my_rank, comm_size, metis_part, metis_perm, metis_inv_perm);
#endif
    }

    double elapsed_main_time;

#ifdef USE_MPI
    elapsed_main_time = MPI_Wtime() - begin_main_time;
#else
    elapsed_main_time = getTimeStamp() - begin_main_time;
#endif

    if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]")
        r_dp.total_walltime = elapsed_main_time;
    else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]")
        r_sp.total_walltime = elapsed_main_time;
    else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
        r_hp.total_walltime = elapsed_main_time;
#endif
    }


#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif

    if(my_rank == 0){
        if(config.mode == 's'){
#ifdef USE_MKL
            if(config.validate_result){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Validating results.\n");}
#endif
                std::vector<double> mkl_result;
                if(config.equilibrate){
                    equilibrate_matrix<double, int>(&total_mtx);
                }
                validate_result(
                    &total_mtx, 
                    &config, 
                    &r_dp, 
                    &r_sp,
#ifdef HAVE_HALF_MATH
                    &r_hp,
#endif 
                    &mkl_result,
                    metis_perm,
                    metis_inv_perm
                );
                
#ifdef DEBUG_MODE
                if(my_rank == 0){printf("Writing validation results to file.\n");}
#endif
                if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
                    write_result_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r_dp, &mkl_result, comm_size);
                }
                else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
                    write_result_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r_sp, &mkl_result, comm_size);
                }
                else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
                    write_result_to_file<_Float16, int>(&matrix_file_name, &(config.seg_method), &config, &r_hp, &mkl_result, comm_size);
#endif
                }
            }
#endif
        }
        else if(config.mode == 'b'){
#ifdef DEBUG_MODE
            if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
            if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
                write_bench_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r_dp, comm_size);
            }
            else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
                write_bench_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r_sp, comm_size);
            }
            else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
                write_bench_to_file<_Float16, int>(&matrix_file_name, &(config.seg_method), &config, &r_hp, comm_size);
#endif            
            }
        }
    }
#ifdef USE_MPI
    // Delete allocated permutation vectors, if metis used
    if(config.seg_method == "seg-metis"){
        delete[] metis_part;
        delete[] metis_perm;
        delete[] metis_inv_perm;
    }
#endif
}

int main(int argc, char *argv[]){

#ifdef DEBUG_MODE
    std::cout << "Using c++ version: " << __cplusplus << std::endl;
#endif

    // Initialize just out of convenience
    int my_rank = 0, comm_size = 1;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Beginning of uspmv main execution.\n");}
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_INIT;
#endif

    // Bogus parallel region pin threads to cores
    bogus_init_pin();
    double begin_main_time;

#ifdef USE_MPI
    begin_main_time = MPI_Wtime();
#else
    begin_main_time = getTimeStamp();
#endif

    Config config;
    std::string matrix_file_name{};

    // Set defaults for cl inputs
    // TODO: Really should be done elsewhere
    std::string seg_method = "seg-rows";
    std::string kernel_format = "scs";
    std::string value_type = "dp";
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;

    parse_cli_inputs(argc, argv, &matrix_file_name, &seg_method, &kernel_format, &value_type, &config, my_rank);

    config.seg_method = seg_method;
    config.kernel_format = kernel_format;
    config.value_type = value_type;

    standalone_bench(config, matrix_file_name, my_rank, comm_size, begin_main_time);

#ifdef USE_MPI
    MPI_Finalize();
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("End of uspmv main execution.\n");}
#endif

    return 0;
}