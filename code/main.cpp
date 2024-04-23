#include "mmio.h"
#include "vectors.h"
#include "utilities.hpp"
#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "write_results.hpp"
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


#define WARM_UP_REPS 100

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

/**
    @brief Perform SPMV kernel, either in "solve" mode or "bench" mode
    @param *config : struct to initialze default values and user input
    @param *local_scs : pointer to process-local scs struct 
    @param *local_context : struct containing communication information
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_y : Process-local results vector, instance of SimpleDenseMatrix class
    @param *local_x : local RHS vector, instance of SimpleDenseMatrix class
    @param *r : a Result struct, in which results of the benchmark are stored
*/
template <typename VT, typename IT>
void bench_spmv(
    Config *config,
    ScsData<VT, IT> *local_scs,
    ContextData<IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<VT> *local_x,
    Result<VT, IT> *r,
    int my_rank,
    int comm_size)
{

#ifdef USE_MPI
    // Allocate a send buffer for each process we're sending a message to
    int nz_comms = local_context->non_zero_receivers.size();
    int nz_recver;

    VT *to_send_elems[nz_comms];
    for(int i = 0; i < nz_comms; ++i){
        nz_recver = local_context->non_zero_receivers[i];
        to_send_elems[i] = new VT[local_context->comm_send_idxs[nz_recver].size()];
    }

    int nzr_size = local_context->non_zero_receivers.size();
    int nzs_size = local_context->non_zero_senders.size();

    // Delare MPI requests for non-blocking communication
    MPI_Request *recv_requests = new MPI_Request[local_context->non_zero_senders.size()];
    MPI_Request *send_requests = new MPI_Request[local_context->non_zero_receivers.size()];
#endif

    // Permute x, in order to match the permutation which was done to the columns
    std::vector<VT> local_x_permuted(local_x->size(), 0);

    apply_permutation<VT, IT>(&(local_x_permuted)[0], &(*local_x)[0], &(local_scs->new_to_old_idx)[0], local_scs->n_rows);

    void *comm_args_void_ptr;
    void *kernel_args_void_ptr;

    OnePrecKernelArgs<VT, IT> *one_prec_kernel_args_encoded = new OnePrecKernelArgs<VT, IT>;
    CommArgs<VT, IT> *comm_args_encoded = new CommArgs<VT, IT>;

    // Encode kernel args into struct
    one_prec_kernel_args_encoded->C = local_scs->C;
    one_prec_kernel_args_encoded->n_chunks = local_scs->n_chunks;
    one_prec_kernel_args_encoded->chunk_ptrs = local_scs->chunk_ptrs.data();
    one_prec_kernel_args_encoded->chunk_lengths = local_scs->chunk_lengths.data();
    one_prec_kernel_args_encoded->col_idxs = local_scs->col_idxs.data();
    one_prec_kernel_args_encoded->values = local_scs->values.data();
    one_prec_kernel_args_encoded->local_x = &(local_x_permuted)[0];
    one_prec_kernel_args_encoded->local_y = &(*local_y)[0];
    kernel_args_void_ptr = (void*) one_prec_kernel_args_encoded;

#ifdef USE_MPI
    // Encode comm args into struct
    comm_args_encoded->local_context = local_context;
    comm_args_encoded->to_send_elems = to_send_elems;
    comm_args_encoded->work_sharing_arr = work_sharing_arr;
    comm_args_encoded->perm = local_scs->old_to_new_idx.data();
    comm_args_encoded->recv_requests = recv_requests; // pointer to first element of array
    comm_args_encoded->nzs_size = &nzs_size;
    comm_args_encoded->send_requests = send_requests;
    comm_args_encoded->nzr_size = &nzr_size;
    comm_args_encoded->num_local_elems = &(local_context->num_local_rows);
#endif

    comm_args_encoded->my_rank = &my_rank;
    comm_args_encoded->comm_size = &comm_size;
    comm_args_void_ptr = (void*) comm_args_encoded;

    // Pass args to construct spmv_kernel object
    SpmvKernel<VT, IT> spmv_kernel(config, kernel_args_void_ptr, comm_args_void_ptr);

    // Enter main COMM-SPMV-SWAP loop, bench mode
    if(config->mode == 'b'){
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Warm-up
        double begin_warm_up_loop_time, end_warm_up_loop_time;
#ifdef USE_MPI
        begin_warm_up_loop_time = MPI_Wtime();
#else
        begin_warm_up_loop_time = getTimeStamp();
#endif

        for(int k = 0; k < WARM_UP_REPS; ++k){
#ifdef USE_MPI
            spmv_kernel.init_halo_exchange();
            spmv_kernel.finalize_halo_exchange();
#endif
            spmv_kernel.execute_spmv();
            spmv_kernel.swap_local_vectors();
#ifdef USE_MPI
            if(config->ba_synch)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
        }


#ifdef USE_MPI
        end_warm_up_loop_time = MPI_Wtime();
#else
        end_warm_up_loop_time = getTimeStamp();
#endif

        double begin_bench_loop_time, end_bench_loop_time = 0.0;

        // Use user-defined time limit to calculate n_iter
        double runtime;
        int n_iter = 1;
        int all_procs_finished = 0;

#ifdef USE_LIKWID
#pragma omp parallel
{
LIKWID_MARKER_REGISTER("spmv_benchmark");
}
#endif

        if(config->comm_halos){
#ifdef USE_MPI
            do{
                MPI_Barrier(MPI_COMM_WORLD);
                begin_bench_loop_time = MPI_Wtime();
                for(int k=0; k<n_iter; ++k) {
                    spmv_kernel.init_halo_exchange();
                    spmv_kernel.finalize_halo_exchange();
                    spmv_kernel.execute_spmv();
                    spmv_kernel.swap_local_vectors();
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
                begin_bench_loop_time = getTimeStamp();
#endif
                for(int k=0; k<n_iter; ++k) {

                    spmv_kernel.execute_spmv();
                    spmv_kernel.swap_local_vectors();
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
                runtime = getTimeStamp() - begin_bench_loop_time;
#endif
            } while (runtime < config->bench_time);
            n_iter = n_iter/2;
        }
        r->n_calls = n_iter;
        r->duration_total_s = runtime;
        r->duration_kernel_s = r->duration_total_s/ r->n_calls;
        r->perf_gflops = (double)local_context->total_nnz * 2.0
                            / r->duration_kernel_s
                            / 1e9;                   // Only count usefull flops
    }
    else if(config->mode == 's'){ // Enter main COMM-SPMV-SWAP loop, solve mode
        // Selects the first (n_rows)-many elements of a sorted y vector, and chops off padding
        std::vector<VT> sorted_local_y(local_context->num_local_rows, 0);

        for (int i = 0; i < config->n_repetitions; ++i)
        {
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "before comm spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "before comm spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif

#ifdef USE_MPI
            spmv_kernel.init_halo_exchange();
            spmv_kernel.finalize_halo_exchange();
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "after comm spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "after comm spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif
#endif
            spmv_kernel.execute_spmv();
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "after_kernel spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "after_kernel spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif
            spmv_kernel.swap_local_vectors();
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "after_kernel and swap spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "after_kernel and swap spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif

#ifdef USE_MPI
            if(config->ba_synch)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        // Bring local_x out of permuted space
        apply_permutation<VT, IT>(&(sorted_local_y)[0], &(spmv_kernel.local_x)[0], &(local_scs->old_to_new_idx)[0], local_context->num_local_rows);
        
        // Give result to local_y for results output
        for(int i = 0; i < local_context->num_local_rows; ++i){
            (*local_y)[i] = (sorted_local_y)[i];
        }

        // Manually resize for ease later on (and I don't see a better way)
        local_y->resize(local_context->num_local_rows);
    }

    // Delete the allocated space for each other process send buffers
#ifdef USE_MPI
    for(int i = 0; i < nz_comms; ++i){
        delete[] to_send_elems[i];
    }
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

    r->value_type_str = typeid(VT).name();
    r->index_type_str = typeid(IT).name();
    r->value_type_size = sizeof(VT);
    r->index_type_size = sizeof(IT);

    // r->was_matrix_sorted = local_scs->is_sorted;
    r->was_matrix_sorted = 1;

    r->fill_in_percent = ((double)local_scs->n_elements / local_scs->nnz - 1.0) * 100.0;
    r->C               = local_scs->C;
    r->sigma           = local_scs->sigma;

#ifdef USE_MPI
    delete[] recv_requests;
    delete[] send_requests;
#endif

    delete comm_args_encoded;
    delete one_prec_kernel_args_encoded;
}

/**
    @brief Gather results (either result of computation, or benchmark metrics) to the root MPI process
    @param *config : struct to initialze default values and user input
    @param *r : a Result struct, in which results of the benchmark are stored
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_x_copy : copy of local RHS vector used for validation against MKL
    @param *local_y : Process-local results vector, instance of SimpleDenseMatrix class
*/
template <typename VT, typename IT>
void gather_results(
    Config *config,
    Result<VT, IT> *r,
    IT *work_sharing_arr,
    std::vector<VT> *local_x_copy,
    std::vector<VT> *local_y,
    int my_rank,
    int comm_size
){
    IT num_local_rows = 0;

#ifdef USE_MPI
    num_local_rows = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
#else
    num_local_rows = local_y->size();
#endif

    if(config->mode == 'b'){

        double *perfs_from_procs_arr = new double[comm_size];
        unsigned long *nnz_per_procs_arr = new unsigned long[comm_size];
#ifdef USE_MPI

        MPI_Gather(&(r->perf_gflops),
                1,
                MPI_DOUBLE,
                perfs_from_procs_arr,
                1,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

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
#endif

        // NOTE: Garbage values for all but root process
        r->perfs_from_procs = std::vector<double>(perfs_from_procs_arr, perfs_from_procs_arr + comm_size);
        r->nnz_per_proc = std::vector<unsigned long>(nnz_per_procs_arr, nnz_per_procs_arr + comm_size);

        delete[] perfs_from_procs_arr;
        delete[] nnz_per_procs_arr;


    }
    else if(config->mode == 's'){
        std::vector<VT> sorted_local_y(num_local_rows);
        r->x_out = (*local_x_copy);
        r->y_out = (*local_y);

        if (config->validate_result)
        {
#ifdef USE_MPI
            // TODO: is the size correct here?
            std::vector<VT> total_uspmv_result(work_sharing_arr[comm_size], 0);
            std::vector<VT> total_x(work_sharing_arr[comm_size], 0);

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
                // Gather all y_vector results to root
                MPI_Gatherv(&(r->y_out)[0],
                            num_local_rows,
                            MPI_DOUBLE,
                            &total_uspmv_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);
                // Gather all x_vector copies to root for mkl validation
                MPI_Gatherv(&(*local_x_copy)[0],
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
                            &total_uspmv_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);

                MPI_Gatherv(&(*local_x_copy)[0],
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
            r->total_uspmv_result = total_uspmv_result;
#else
            r->total_x = (*local_x_copy);
            r->total_uspmv_result = r->y_out;
#endif
        }
    }
}

/**
    @brief The main harness for the SpMV kernel, in which we:
        1. Segment and distribute the needed structs to each MPI process (mpi_init_local_structs),
        2. Benchmark the selected SpMV kernel (bench_spmv),
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
    int comm_size)
{
    // TODO: bring back matrix stats!
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    // Declare local structs on each process
    ScsData<VT, IT> local_scs;

    ContextData<IT> local_context;

    // Allocate space for work sharing array
    IT work_sharing_arr[comm_size + 1];

    // Allocate global permutation vectors
    int *metis_part = NULL;
    int *metis_perm = NULL;
    int *metis_inv_perm = NULL;

#ifdef USE_MPI
    if(config->seg_method == "seg-metis"){
        metis_part = new int[total_mtx->n_rows];
        metis_perm = new int[total_mtx->n_rows];
        for(int i = 0; i < total_mtx->n_rows; ++i){
            metis_perm[i] = i;
        }
        metis_inv_perm = new int[total_mtx->n_rows];
    }
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Init local structures.\n");}
#endif

    init_local_structs<VT, IT>(
        &local_scs,
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

    // Declare local vectors to be used
    SimpleDenseMatrix<VT, IT> local_x(&local_context);
    SimpleDenseMatrix<VT, IT> local_y(&local_context);

    // Initialize local_x and y, either randomly, with default values defined in classes_structs.hpp,
    // or with 1s (by default)
    local_x.init(config);
    local_y.init(config);

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

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete bench_spmv.\n");}
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Gather results to root process.\n");}
#endif
    gather_results(config, r, work_sharing_arr, &local_x_copy, &(local_y.vec), my_rank, comm_size);

#ifdef USE_MPI
    // Delete allocated permutation vectors, if metis used
    if(config->seg_method == "seg-metis"){
        delete[] metis_part;
        delete[] metis_perm;
        delete[] metis_inv_perm;
    }
#endif
}

int main(int argc, char *argv[]){

// Initialize just out of convenience
int my_rank = 0, comm_size = 1;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Beginning of USpMV main execution.\n");}
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_INIT;
#endif

    double begin_main_time;

#ifdef USE_MPI
    begin_main_time = MPI_Wtime();
#else
    begin_main_time = getTimeStamp();
#endif

    Config config;
    std::string matrix_file_name{};

    // Set defaults for cl inputs
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

    if (config.value_type == "dp")
    {
        MtxData<double, int> total_mtx;
        Result<double, int> r;

        if(my_rank == 0){
            if(my_rank == 0){printf("Reading mtx file.\n");}
            read_mtx<double, int>(matrix_file_name, config, &total_mtx, my_rank);
            r.total_nnz = total_mtx.nnz;
            r.total_rows = total_mtx.n_rows;
        }
    if(my_rank == 0){printf("Enter compute_result.\n");}
        compute_result<double, int>(&total_mtx, &config, &r, my_rank, comm_size);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif

#ifdef USE_MPI
    r.total_walltime = MPI_Wtime() - begin_main_time;
#else
    r.total_walltime = getTimeStamp() - begin_main_time;
#endif
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
                    write_result_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r, &mkl_dp_result, comm_size);
                }
                else{
                }
            }
            else if(config.mode == 'b'){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
                write_bench_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r, comm_size);
            }
        }
    }
    else if (config.value_type == "sp")
    {
        MtxData<float, int> total_mtx;
        Result<float, int> r;

        if(my_rank == 0){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
            read_mtx<float, int>(matrix_file_name, config, &total_mtx, my_rank);
            r.total_nnz = total_mtx.nnz;
            r.total_rows = total_mtx.n_rows;
        }
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif
        compute_result<float, int>(&total_mtx, &config, &r, my_rank, comm_size);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif
#ifdef USE_MPI
    r.total_walltime = MPI_Wtime() - begin_main_time;
#else
    r.total_walltime = getTimeStamp() - begin_main_time;
#endif
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
                    write_result_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r, &mkl_sp_result, comm_size);
                }
                else{
                }
            }
            else if(config.mode == 'b'){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
                write_bench_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r, comm_size);
            }
        }
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("End of USpMV main execution.\n");}
#endif
    return 0;
}