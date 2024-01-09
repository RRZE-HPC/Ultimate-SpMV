#include "mmio.h"
#include "vectors.h"
#include "utilities.hpp"
#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "write_results.hpp"

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

#define WARM_UP_REPS 100

#ifdef _OPENMP
#include <omp.h>
#endif

/**
    @brief Perform SPMVM kernel, either in "solve" mode or "bench" mode
    @param *config : struct to initialze default values and user input
    @param *local_scs : pointer to local scs struct 
    @param *local_context : struct containing local_scs + communication information
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_y : local results vector, instance of SimpleDenseMatrix class
    @param *local_x : local RHS vector, instance of SimpleDenseMatrix class
    @param *r : a Result struct, in which results of the benchmark are stored
*/
template <typename VT, typename IT>
void bench_spmv(
    Config *config,
    ScsData<VT, IT> *local_scs,
    ContextData<VT, IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<VT> *local_x,
    Result<VT, IT> *r,
    int my_rank,
    int comm_size)
{
    // Allocate a send buffer for each process we're sending a message to
    int nz_comms = local_context->non_zero_receivers.size();
    int nz_recver;
    MPI_Request recv_requests[comm_size];
    MPI_Request send_requests[comm_size];

    VT *to_send_elems[nz_comms];
    for(int i = 0; i < nz_comms; ++i){
        nz_recver = local_context->non_zero_receivers[i];
        to_send_elems[i] = new VT[local_context->comm_send_idxs[nz_recver].size()];
    }

    int nzr_size = local_context->non_zero_receivers.size();
    int nzs_size = local_context->non_zero_senders.size();

    // TODO: Permute x, since local matrix permuted symmetrically
    // std::vector<VT> local_x_permuted(local_x->size(), 0);
    // apply_permutation<VT, IT>(&(local_x_permuted)[0], &(*local_x)[0], &(local_scs->new_to_old_idx)[0], local_scs->n_rows);

    std::vector<VT> sorted_local_y(local_y->size(), 0);

    // Enter main COMM-SPMVM-SWAP loop, bench mode
    if(config->mode == 'b'){
        std::vector<VT> dummy_x(local_x->size(), 1.0);
        std::vector<VT> dummy_y(local_y->size(), 0.0);

        MPI_Barrier(MPI_COMM_WORLD);
        // Warm-up
        double begin_warm_up_loop_time, end_warm_up_loop_time;
        begin_warm_up_loop_time = MPI_Wtime();

            for(int k = 0; k < WARM_UP_REPS; ++k){
#ifdef COMM_HALOS
                communicate_halo_elements<VT, IT>(
                    local_scs,
                    local_context, 
                    local_x, 
                    to_send_elems,
                    work_sharing_arr, 
                    recv_requests,
                    nzs_size,
                    send_requests,
                    nzr_size,
                    local_context->num_local_rows,
                    my_rank,
                    comm_size);
#endif
                    spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                        local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                        local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
                // spmv_omp_csr<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                //                     local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                //                     local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

                    std::swap(dummy_x, dummy_y);
#ifdef BARRIER_SYNC
                    MPI_Barrier(MPI_COMM_WORLD);
#endif
            }

        end_warm_up_loop_time = MPI_Wtime();

        double begin_bench_loop_time, end_bench_loop_time;

        if (config->bench_time <= 0.0){
            // Use warm-up to calculate n_iter for real benchmark
            int n_iter = 10; // NOTE: is it correct for the root process' iteration count to be what is used?
            if(my_rank == 0){
                // Guards against REALLY bad load balancing 
                int num_reps_warm = static_cast<int>((double)WARM_UP_REPS / (end_warm_up_loop_time - begin_warm_up_loop_time));
                if (num_reps_warm > 1)
                    n_iter = num_reps_warm;
            }

            MPI_Bcast(
                &n_iter,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            MPI_Barrier(MPI_COMM_WORLD);
            begin_bench_loop_time = MPI_Wtime();
            
            for(int k = 0; k < n_iter; ++k){
#ifdef COMM_HALOS
                communicate_halo_elements<VT, IT>(
                    local_scs,
                    local_context, 
                    local_x, 
                    to_send_elems,
                    work_sharing_arr, 
                    recv_requests,
                    nzs_size,
                    send_requests,
                    nzr_size,
                    local_context->num_local_rows,
                    my_rank,
                    comm_size);
#endif

                spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
                // spmv_omp_csr<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                //                     local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                //                     local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
                

                std::swap(dummy_x, dummy_y);
#ifdef BARRIER_SYNC
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            }

            MPI_Barrier(MPI_COMM_WORLD);
            end_bench_loop_time = MPI_Wtime();

            r->n_calls = n_iter;
            r->duration_total_s = end_bench_loop_time - begin_bench_loop_time;
            r->duration_kernel_s = r->duration_total_s/ r->n_calls;
            r->perf_gflops = (double)local_context->total_nnz * 2.0
                                / r->duration_kernel_s
                                / 1e9;                   // Only count usefull flops
        }
        else{            
            // Use user-defined time limit to calculate n_iter
            double runtime = 0.0;
            int n_iter = 1;
            int all_procs_finished = 0;
            // begin_bench_loop_time = MPI_Wtime();
            // MPI_Barrier(MPI_COMM_WORLD);

            do{
                MPI_Barrier(MPI_COMM_WORLD);
                begin_bench_loop_time = MPI_Wtime();

                for(int k=0; k<n_iter; ++k) {
#ifdef COMM_HALOS
                    communicate_halo_elements<VT, IT>(
                        local_scs,
                        local_context, 
                        local_x, 
                        to_send_elems,
                        work_sharing_arr, 
                        recv_requests,
                        nzs_size,
                        send_requests,
                        nzr_size,
                        local_context->num_local_rows,
                        my_rank,
                        comm_size);
#endif

                    spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                        local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                        local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
                    // spmv_omp_csr<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                    //                     local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                    //                     local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
                    

                    std::swap(dummy_x, dummy_y);
    #ifdef BARRIER_SYNC
                    MPI_Barrier(MPI_COMM_WORLD);
    #endif
                }
                MPI_Barrier(MPI_COMM_WORLD);
                n_iter = n_iter*2;
                runtime = MPI_Wtime() - begin_bench_loop_time;
            } while (runtime < config->bench_time);
            n_iter = n_iter/2;

            r->n_calls = n_iter;
            r->duration_total_s = runtime;
            r->duration_kernel_s = r->duration_total_s/ r->n_calls;
            r->perf_gflops = (double)local_context->total_nnz * 2.0
                                / r->duration_kernel_s
                                / 1e9;                   // Only count usefull flops
        }
    }
    else if(config->mode == 's'){ // Enter main COMM-SPMVM-SWAP loop, solve mode
        for (int i = 0; i < config->n_repetitions; ++i)
        {
            communicate_halo_elements<VT, IT>(
                local_scs,
                local_context, 
                local_x, 
                to_send_elems,
                work_sharing_arr, 
                recv_requests,
                nzs_size,
                send_requests,
                nzr_size,
                local_context->num_local_rows,
                my_rank,
                comm_size);

            spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
            // spmv_omp_csr<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
            //                     local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
            //                     local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

            // In unsymmetric permutation, y need be sorted every iteration for accurate results
            apply_permutation(&(sorted_local_y)[0], &(*local_y)[0], &(local_scs->old_to_new_idx)[0], local_scs->n_rows);

            std::swap(*local_x, sorted_local_y);
#ifdef BARRIER_SYNC
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
        // Give x results to y as output
        std::swap(*local_x, *local_y);
    }

    // Delete the allocated space for each other process
    for(int i = 0; i < nz_comms; ++i){
        delete[] to_send_elems[i];
    }

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

    r->value_type_str = typeid(VT).name();
    r->index_type_str = typeid(IT).name();
    r->value_type_size = sizeof(VT);
    r->index_type_size = sizeof(IT);

    // r->was_matrix_sorted = local_scs->is_sorted;
    r->was_matrix_sorted = 1;

    r->fill_in_percent = ((double)local_scs->n_elements / local_scs->nnz - 1.0) * 100.0;
    r->C               = local_scs->C;
    r->sigma           = local_scs->sigma;
}

/**
    @brief The main harness for the SPMVM kernel, in which we first segment distribute the needed structs. Validation happens outside this routine
    @param *total_mtx : complete mtx struct, read .mtx file with mtx_reader.h
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a Result struct, in which results of the computation are stored
*/
template <typename VT, typename IT>
void compute_result(
    MtxData<VT, IT> *total_mtx,
    const std::string *seg_method,
    Config *config,
    Result<VT, IT> *r,
    int my_rank,
    int comm_size)
{
    // TODO: bring back matrix stats
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    // Declare local structs on each process
    ScsData<VT, IT> local_scs;
    ContextData<VT, IT> local_context;

    // Allocate space for work sharing array
    IT work_sharing_arr[comm_size + 1];

    // Allocate global permutation vectors
    int *metis_part = NULL;
    int *metis_perm = NULL;
    int *metis_inv_perm = NULL;
    if(*seg_method == "seg-metis"){
        metis_part = new int[total_mtx->n_rows];
        metis_perm = new int[total_mtx->n_rows];
        for(int i = 0; i < total_mtx->n_rows; ++i){
            metis_perm[i] = i;
        }
        metis_inv_perm = new int[total_mtx->n_rows];
    }

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Init local structures.\n");}
#endif
    mpi_init_local_structs<VT, IT>(
        &local_scs,
        &local_context, 
        total_mtx,
        config, 
        seg_method, 
        work_sharing_arr, 
        my_rank, 
        comm_size,
        metis_part,
        metis_perm,
        metis_inv_perm
    );

    // Declare local vectors to be used
    SimpleDenseMatrix<VT, IT> local_x(&local_context);
    SimpleDenseMatrix<VT, IT> local_y(&local_context);

    // Initialize local_x, either randomly, with default values defined in classes_structs.hpp,
    // or with 0s (by default)
    local_x.init(config);

    // Copy contents of local_x for output, and validation against mkl
    std::vector<VT> local_x_copy = local_x.vec;
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter bench_spmv.\n");}
#endif
    bench_spmv<VT, IT>(
        config,
        &local_scs,
        &local_context,
        work_sharing_arr,
        &local_y.vec,
        &local_x.vec,
        r,
        my_rank,
        comm_size
    );
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete bench_spmv.\n");}
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Gather results to root process.\n");}
#endif

    if(config->mode == 'b'){
        double *perfs_from_procs_arr = new double[comm_size];

        MPI_Gather(&(r->perf_gflops),
                1,
                MPI_DOUBLE,
                perfs_from_procs_arr,
                1,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

        // NOTE: Garbage values for all but root process
        r->perfs_from_procs = std::vector<double>(perfs_from_procs_arr, perfs_from_procs_arr + comm_size);

        delete[] perfs_from_procs_arr;
    }
    else if(config->mode == 's'){
        std::vector<VT> sorted_local_y(local_scs.n_rows);
        r->x_out = local_x_copy;
        r->y_out = local_y.vec;

        if (config->validate_result)
        {
            // TODO: is the size correct here?
            std::vector<VT> total_spmvm_result(work_sharing_arr[comm_size], 0);
            std::vector<VT> total_x(work_sharing_arr[comm_size], 0);

            IT num_local_rows = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
            IT counts_arr[comm_size];
            IT displ_arr_bk[comm_size];

            for(IT i = 0; i < comm_size; ++i){
                counts_arr[i] = IT{};
                displ_arr_bk[i] = IT{};
            }
            
            for (IT i = 0; i < comm_size; ++i){
                counts_arr[i] = work_sharing_arr[i + 1] - work_sharing_arr[i];
                displ_arr_bk[i] = work_sharing_arr[i];
            }

            if (typeid(VT) == typeid(double)){
                MPI_Gatherv(&(r->y_out)[0],
                            num_local_rows,
                            MPI_DOUBLE,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);

                MPI_Gatherv(&local_x_copy[0],
                            num_local_rows,
                            MPI_DOUBLE,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);
            }
            else if (typeid(VT) == typeid(float)){
                MPI_Gatherv(&(r->y_out)[0],
                            num_local_rows,
                            MPI_FLOAT,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);

                MPI_Gatherv(&local_x_copy[0],
                            num_local_rows,
                            MPI_FLOAT,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);
            }

            // If we're verifying results, assign total vectors to Result object
            // NOTE: Garbage values for all but root process
            r->total_x = total_x;
            r->total_spmvm_result = total_spmvm_result;
        }
    }

    // Delete allocated permutation vectors, if metis used
    if(*seg_method == "seg-metis"){
        delete[] metis_part;
        delete[] metis_perm;
        delete[] metis_inv_perm;
    }
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Beginning of USpMV execution.\n");}
#endif

    double begin_main_time = MPI_Wtime();

    Config config;
    std::string matrix_file_name{};

    // Set defaults for cl inputs
    std::string seg_method = "seg-rows";
    std::string value_type = "dp";
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;

    double total_walltimes[comm_size];

    verify_and_assign_inputs(argc, argv, &matrix_file_name, &seg_method, &value_type, &config, my_rank);

    if (value_type == "dp")
    {
        MtxData<double, int> total_mtx;
        Result<double, int> r;

        if(my_rank == 0){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
            read_mtx<double, int>(matrix_file_name, config, &total_mtx, my_rank);
        }
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif
        compute_result<double, int>(&total_mtx, &seg_method, &config, &r, my_rank, comm_size);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif
        double time_per_proc = MPI_Wtime() - begin_main_time;
        // Gather all times for printing of results
        MPI_Gather(
            &time_per_proc,
            1,
            MPI_DOUBLE,
            total_walltimes,
            1,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        if(my_rank == 0){
            if(config.mode == 's'){
                if(config.validate_result){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Validating results.\n");}
#endif
                    std::vector<double> mkl_dp_result;
                    validate_dp_result(&total_mtx, &config, &r, &mkl_dp_result);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing validation results to file.\n");}
#endif
                    write_dp_result_to_file(&matrix_file_name, &seg_method, &config, &r, &mkl_dp_result, comm_size);
                }
                else{
                }
            }
            else if(config.mode == 'b'){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
                write_bench_to_file<double, int>(&matrix_file_name, &seg_method, &config, &r, total_walltimes, comm_size);
            }
        }
    }
    else if (value_type == "sp")
    {
        MtxData<float, int> total_mtx;
        Result<float, int> r;

        if(my_rank == 0){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
            read_mtx<float, int>(matrix_file_name, config, &total_mtx, my_rank);
        }
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif
        compute_result<float, int>(&total_mtx, &seg_method, &config, &r, my_rank, comm_size);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif
        double time_per_proc = MPI_Wtime() - begin_main_time;

        // Gather all times for printing of results
        MPI_Gather(
            &time_per_proc,
            1,
            MPI_DOUBLE,
            total_walltimes,
            1,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        if(my_rank == 0){
            if(config.mode == 's'){
                if(config.validate_result){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Validating results.\n");}
#endif
                    std::vector<float> mkl_sp_result;
                    validate_sp_result(&total_mtx, &config, &r, &mkl_sp_result);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing validation results to file.\n");}
#endif
                    write_sp_result_to_file(&matrix_file_name, &seg_method, &config, &r, &mkl_sp_result, comm_size);
                }
                else{
                }
            }
            else if(config.mode == 'b'){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
                write_bench_to_file<float, int>(&matrix_file_name, &seg_method, &config, &r, total_walltimes, comm_size);
            }
        }
    }

    MPI_Finalize();

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("End of USpMV execution.\n");}
#endif
    return 0;
}