#include "classes_structs.hpp"
#include "mtx_reader.h"
#include "vectors.h"
#include "utilities.hpp"
#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "write_results.hpp"


#define WARM_UP_REPS 15

#ifdef _OPENMP
#include <omp.h>
#endif

/**
    @brief Collect halo element row indices for each local x-vector, and perform SPMVM
    @param *config : struct to initialze default values and user input
    @param *local_scs : pointer to local scs struct
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_y : local results vector, instance of SimpleDenseMatrix class
    @param *local_x : local RHS vector, instance of SimpleDenseMatrix class
    @param *r : a Result struct, in which results of the benchmark are stored
    @param *defaults : a DefaultValues struct, in which default values of x and y can be defined
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
    int comm_size
    )
{
    // Enter main COMM-SPMVM-SWAP loop, bench mode
    if(config->mode == 'b'){
        if(config->log_prof && my_rank == 0) {log("Begin COMM-SPMVM-SWAP loop, bench mode");}
        double begin_csslbm_time = MPI_Wtime();

        std::vector<VT> dummy_x(local_x->size(), 1.0);
        std::vector<VT> dummy_y(local_y->size(), 0.0);

        MPI_Barrier(MPI_COMM_WORLD);
        // Warm-up
        double begin_warm_up_loop_time, end_warm_up_loop_time;
        begin_warm_up_loop_time = MPI_Wtime();
        for(int k = 0; k < WARM_UP_REPS; ++k){
                if(config->comm_halos){
                    communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);
                    MPI_Barrier(MPI_COMM_WORLD);
                }

                spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

                std::swap(dummy_x, dummy_y);

                // if(dummy_x[0]>1.0){ // prevent compiler from eliminating loop
                //     printf("%lf", dummy_x[local_x->size() / 2]);
                //     exit(0);
                // }
            }
        end_warm_up_loop_time = MPI_Wtime();

        // Use warm-up to calculate n_iter for real benchmark
        int n_iter = static_cast<int>((double)WARM_UP_REPS / (end_warm_up_loop_time - begin_warm_up_loop_time));

        // std::cout << "Proc: " << my_rank << ", niter: " << n_iter << std::endl;

        double begin_bench_loop_time, end_bench_loop_time;

        begin_bench_loop_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        
        for(int k = 0; k < n_iter; ++k){
            if(config->comm_halos){
                communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);
                MPI_Barrier(MPI_COMM_WORLD);
            }

            spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

            std::swap(dummy_x, dummy_y);

            // if(dummy_x[0]>1.0){ // prevent compiler from eliminating loop
            //     printf("%lf", dummy_x[local_x->size() / 2]);
            //     exit(0);
            // }
        }
            
        MPI_Barrier(MPI_COMM_WORLD);
        end_bench_loop_time = MPI_Wtime();

        r->n_calls = n_iter;
        r->duration_total_s = end_bench_loop_time - begin_bench_loop_time;
        r->duration_kernel_s = r->duration_total_s/ r->n_calls;
        r->perf_gflops = (double)local_context->total_nnz * 2.0
                            / r->duration_kernel_s
                            / 1e9;                   // Only count usefull flops

        if(config->log_prof && my_rank == 0) {log("Finish COMM-SPMVM-SWAP loop, bench mode", begin_csslbm_time, MPI_Wtime());}
    }
    else if(config->mode == 's'){ // Enter main COMM-SPMVM-SWAP loop, solve mode
        if(config->log_prof && my_rank == 0) {log("Begin COMM-SPMVM-SWAP loop, solve mode");}
        double begin_csslsm_time = MPI_Wtime();
        for (IT i = 0; i < config->n_repetitions; ++i)
        {
            communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);
            MPI_Barrier(MPI_COMM_WORLD);

            spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

            std::swap(*local_x, *local_y);
        }
        std::swap(*local_x, *local_y);

        if(config->log_prof && my_rank == 0) {log("Finish COMM-SPMVM-SWAP loop, solve mode", begin_csslsm_time, MPI_Wtime());}
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
    @brief The main harness for the rest of the functions, in which we segment and execute the work to be done.
        Validation happens outside this routine
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
    ContextData<IT> local_context;

    // Allocate space for work sharing array
    IT work_sharing_arr[comm_size + 1];

    mpi_init_local_structs<VT, IT>(
        &local_scs,
        &local_context, 
        total_mtx,
        config, 
        seg_method, 
        work_sharing_arr, 
        my_rank, 
        comm_size
    );

    // Declare local vectors to be used
    SimpleDenseMatrix<VT, IT> local_x(&local_context);
    SimpleDenseMatrix<VT, IT> local_y(&local_context);

    // Initialize local_x, either randomly, with default values defined in classes_structs.hpp,
    // or with 0s (by default)
    local_x.init(config);

    // Copy contents of local_x for output, and validation against mkl
    std::vector<VT> local_x_copy = local_x.vec;

    if(config->log_prof && my_rank == 0) {log("Begin bench_spmv");}
    double begin_bs_time = MPI_Wtime();
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
    if(config->log_prof && my_rank == 0) {log("Finish bench_spmv", begin_bs_time, MPI_Wtime());}

    if(config->log_prof && my_rank == 0) {log("Begin results gathering");}
    double begin_rg_time = MPI_Wtime();
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
        // Assign proc local x and y to Result object
        r->x_out = local_x_copy;
        r->y_out = local_y.vec;

        if (config->validate_result)
        {
            // TODO: is the size correct here?
            std::vector<VT> total_spmvm_result(work_sharing_arr[comm_size], 0);
            std::vector<VT> total_x(work_sharing_arr[comm_size], 0);

            IT amnt_local_elems = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
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
                MPI_Gatherv(&(local_y.vec)[0],
                            amnt_local_elems,
                            MPI_DOUBLE,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);

                MPI_Gatherv(&local_x_copy[0],
                            amnt_local_elems,
                            MPI_DOUBLE,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);
            }
            else if (typeid(VT) == typeid(float)){
                MPI_Gatherv(&(local_y.vec)[0],
                            amnt_local_elems,
                            MPI_FLOAT,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);

                MPI_Gatherv(&local_x_copy[0],
                            amnt_local_elems,
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
    if(config->log_prof && my_rank == 0) {log("Finish results gathering", begin_rg_time, MPI_Wtime());}
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    double begin_main_time = MPI_Wtime();

    Config config;
    std::string matrix_file_name{};

    // Set defaults for cl inputs
    std::string seg_method = "seg-rows";
    std::string value_type = "dp";
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;

    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double total_walltimes[comm_size];

    verify_and_assign_inputs(argc, argv, &matrix_file_name, &seg_method, &value_type, &config);
    
    if(config.log_prof && my_rank == 0) {log("__________ log start __________");}
    if(config.log_prof && my_rank == 0) {log("Begin main");}

    if (value_type == "dp")
    {
        MtxData<double, int> total_mtx;
        Result<double, int> r;

        if(my_rank == 0){
            if(config.log_prof && my_rank == 0) {log("Begin read_mtx_data");}
            double begin_rmtxd_time = MPI_Wtime();
            total_mtx = read_mtx_data<double, int>(matrix_file_name.c_str(), config.sort_matrix);
            if(config.log_prof && my_rank == 0) {log("Finish read_mtx_data", begin_rmtxd_time, MPI_Wtime());}
        }
        if(config.log_prof && my_rank == 0) {log("Begin compute_result");}
        double begin_cr_time = MPI_Wtime();
        compute_result<double, int>(&total_mtx, &seg_method, &config, &r, my_rank, comm_size);
        if(config.log_prof && my_rank == 0) {log("Finish compute_result",  begin_cr_time, MPI_Wtime());}

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
                    std::vector<double> mkl_dp_result;
                    if(config.log_prof && my_rank == 0) {log("Begin mkl validation");}
                    double begin_mklv_time = MPI_Wtime();
                    validate_dp_result(&total_mtx, &config, &r, &mkl_dp_result);
                    if(config.log_prof && my_rank == 0) {log("Finish mkl validation",  begin_mklv_time, MPI_Wtime());}
                    write_dp_result_to_file(&matrix_file_name, &seg_method, &config, &r, &mkl_dp_result, comm_size);
                }
                else{
                    if(config.log_prof && my_rank == 0) {log("Result not validated");}
                }
            }
            else if(config.mode == 'b'){
                write_bench_to_file<double, int>(&matrix_file_name, &seg_method, &config, &r, total_walltimes, comm_size);
            }
        }
    }
    else if (value_type == "sp")
    {
        MtxData<float, int> total_mtx;
        Result<float, int> r;

        if(my_rank == 0){
            if(config.log_prof && my_rank == 0) {log("Begin read_mtx_data");}
            double begin_rmtxd_time = MPI_Wtime();
            total_mtx = read_mtx_data<float, int>(matrix_file_name.c_str(), config.sort_matrix);
            if(config.log_prof && my_rank == 0) {log("Finish read_mtx_data", begin_rmtxd_time, MPI_Wtime());}
        }
        if(config.log_prof && my_rank == 0) {log("Begin compute_result");}
        double begin_cr_time = MPI_Wtime();
        compute_result<float, int>(&total_mtx, &seg_method, &config, &r, my_rank, comm_size);
        if(config.log_prof && my_rank == 0) {log("Finish compute_result",  begin_cr_time, MPI_Wtime());}

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
                    std::vector<float> mkl_sp_result;

                    if(config.log_prof && my_rank == 0) {log("Begin mkl validation");}
                    double begin_mklv_time = MPI_Wtime();
                    validate_sp_result(&total_mtx, &config, &r, &mkl_sp_result);
                    if(config.log_prof && my_rank == 0) {log("Finish mkl validation",  begin_mklv_time, MPI_Wtime());}
                    write_sp_result_to_file(&matrix_file_name, &seg_method, &config, &r, &mkl_sp_result, comm_size);
                }
                else{
                    if(config.log_prof && my_rank == 0) {log("Result not validated");}
                }
            }
            else if(config.mode == 'b'){
                write_bench_to_file<float, int>(&matrix_file_name, &seg_method, &config, &r, total_walltimes, comm_size);
            }
        }
    }

    if(config.log_prof && my_rank == 0) {log("Finish main", begin_main_time, MPI_Wtime());}
    if(config.log_prof && my_rank == 0) {log("__________ log end __________");}

    MPI_Finalize();

    return 0;
}