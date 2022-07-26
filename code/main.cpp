#include "spmv.h"
#include "structs.hpp"
#include "mtx-reader.h"
#include "vectors.h"
#include "utilities.hpp"

#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "benchmark.hpp"

#include "write_results.hpp"


#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <ctime>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include <set>
#include <mpi.h>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif


/**
    @brief The main harness for the rest of the functions, in which we segment and execute the work to be done.
        Validation happens outside this routine
    @param *total_mtx
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
*/
template <typename VT, typename IT>
void compute_result(
    MtxData<VT, IT> *total_mtx,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<VT, IT> *r,
    const int *my_rank,
    const int *comm_size)
{
    // TODO: bring back matrix stats
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    // Declare local structs on each process
    ScsData<VT, IT> local_scs;
    ContextData<IT> local_context;

    // Allocate space for work sharing array
    IT work_sharing_arr[*comm_size + 1];

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

    SimpleDenseMatrix<VT, IT> local_x(&local_context);
    SimpleDenseMatrix<VT, IT> local_y(&local_context);

    // Initialize local_x, either randomly, with defaults, or with a predefined x_in
    DefaultValues<VT, IT> default_values;
    init_std_vec_with_ptr_or_value(
        local_x.vec, 
        local_x.vec.size(),
        default_values.x, 
        config->random_init_x
    );

    // Copy contents of local_x for output, and validation against mkl
    std::vector<VT> local_x_copy = local_x.vec;

    if(config->log_prof && *my_rank == 0) {log("Begin bench_spmv");}
    clock_t begin_bs_time = std::clock();
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
    if(config->log_prof && *my_rank == 0) {log("Finish bench_spmv", begin_bs_time, std::clock());}

    if(config->log_prof && *my_rank == 0) {log("Begin results gathering");}
    clock_t begin_rg_time = std::clock();
    if(config->mode == 'b'){
        double *perfs_from_procs_arr = new double[*comm_size];

        MPI_Gather(&(r->perf_mflops),
                1,
                MPI_DOUBLE,
                perfs_from_procs_arr,
                1,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

        // NOTE: Garbage values for all but root process
        r->perfs_from_procs = std::vector<double>(perfs_from_procs_arr, perfs_from_procs_arr + *comm_size);

        delete[] perfs_from_procs_arr;
    }
    else if(config->mode == 's'){
        // Assign proc local x and y to benchmark result object
        r->x_out = local_x_copy;
        r->y_out = local_y.vec;

        if (config->validate_result)
        {
            // TODO: is the size correct here?
            std::vector<VT> total_spmvm_result(work_sharing_arr[*comm_size], 0);
            std::vector<VT> total_x(work_sharing_arr[*comm_size], 0);

            IT counts_arr[*comm_size];
            IT displ_arr_bk[*comm_size];

            for(IT i = 0; i < *comm_size; ++i){
                counts_arr[i] = IT{};
                displ_arr_bk[i] = IT{};
            }
            
            for (IT i = 0; i < *comm_size; ++i){
                counts_arr[i] = work_sharing_arr[i + 1] - work_sharing_arr[i];
                displ_arr_bk[i] = work_sharing_arr[i];
            }

            // Collect each of the process local x and y vectors to a global/total vector for validation on root proc
            if (typeid(VT) == typeid(double)){
                MPI_Allgatherv(&(local_y.vec)[0],
                                work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank],
                                MPI_DOUBLE,
                                &total_spmvm_result[0],
                                counts_arr,
                                displ_arr_bk,
                                MPI_DOUBLE,
                                MPI_COMM_WORLD);

                MPI_Allgatherv(&local_x_copy[0],
                                work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank],
                                MPI_DOUBLE,
                                &total_x[0],
                                counts_arr,
                                displ_arr_bk,
                                MPI_DOUBLE,
                                MPI_COMM_WORLD);
            }
            else if (typeid(VT) == typeid(float)){
                MPI_Allgatherv(&(local_y.vec)[0],
                                work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank],
                                MPI_FLOAT,
                                &total_spmvm_result[0],
                                counts_arr,
                                displ_arr_bk,
                                MPI_FLOAT,
                                MPI_COMM_WORLD);

                MPI_Allgatherv(&local_x_copy[0],
                                work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank],
                                MPI_FLOAT,
                                &total_x[0],
                                counts_arr,
                                displ_arr_bk,
                                MPI_FLOAT,
                                MPI_COMM_WORLD);
            }

            // If we're verifying results, assign total vectors to benchmark result object
            r->total_x = total_x;
            r->total_spmvm_result = total_spmvm_result;
        }
    }
    if(config->log_prof && *my_rank == 0) {log("Finish results gathering", begin_rg_time, std::clock());}
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    Config config;
    std::string matrix_file_name{};

    clock_t begin_main_time = std::clock();

    // Set defaults for cl inputs
    std::string seg_method = "seg-rows";
    std::string value_type = "dp";
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;

    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MARKER_INIT();

    verify_and_assign_inputs(argc, argv, &matrix_file_name, &seg_method, &value_type, &config);
    
    if(config.log_prof && my_rank == 0) {log("__________ log start __________");}
    if(config.log_prof && my_rank == 0) {log("Begin main");}

    if (value_type == "dp")
    {
        MtxData<double, int> total_mtx;
        BenchmarkResult<double, int> r;

        if(my_rank == 0){
            if(config.log_prof && my_rank == 0) {log("Begin read_mtx_data");}
            clock_t begin_rmtxd_time = std::clock();
            total_mtx = read_mtx_data<double, int>(matrix_file_name.c_str(), config.sort_matrix);
            if(config.log_prof && my_rank == 0) {log("Finish read_mtx_data", begin_rmtxd_time, std::clock());}
        }
        if(config.log_prof && my_rank == 0) {log("Begin compute_result");}
        clock_t begin_cr_time = std::clock();
        compute_result<double, int>(&total_mtx, &seg_method, &config, &r, &my_rank, &comm_size);
        if(config.log_prof && my_rank == 0) {log("Finish compute_result",  begin_cr_time, std::clock());}

        if(my_rank == 0){
            if(config.mode == 's'){
                if(config.validate_result){
                    std::vector<double> mkl_dp_result;
                    if(config.log_prof && my_rank == 0) {log("Begin mkl validation");}
                    clock_t begin_mklv_time = std::clock();
                    validate_dp_result(&total_mtx, &config, &r, &mkl_dp_result);
                    if(config.log_prof && my_rank == 0) {log("Finish mkl validation",  begin_mklv_time, std::clock());}
                    write_dp_result_to_file(&matrix_file_name, &seg_method, &config, &r, &mkl_dp_result, &comm_size);
                }
                else{
                    if(config.log_prof && my_rank == 0) {log("Result not validated");}
                }
            }
            else if(config.mode == 'b'){
                write_bench_to_file<double, int>(&matrix_file_name, &seg_method, &config, &r, &comm_size);
            }
        }
    }
    else if (value_type == "sp")
    {
        MtxData<float, int> total_mtx;
        BenchmarkResult<float, int> r;

        if(my_rank == 0){
            if(config.log_prof && my_rank == 0) {log("Begin read_mtx_data");}
            clock_t begin_rmtxd_time = std::clock();
            total_mtx = read_mtx_data<float, int>(matrix_file_name.c_str(), config.sort_matrix);
            if(config.log_prof && my_rank == 0) {log("Finish read_mtx_data", begin_rmtxd_time, std::clock());}
        }
        if(config.log_prof && my_rank == 0) {log("Begin compute_result");}
        clock_t begin_cr_time = std::clock();
        compute_result<float, int>(&total_mtx, &seg_method, &config, &r, &my_rank, &comm_size);
        if(config.log_prof && my_rank == 0) {log("Finish compute_result",  begin_cr_time, std::clock());}

        if(my_rank == 0){
            if(config.mode == 's'){
                if(config.validate_result){
                    std::vector<float> mkl_sp_result;

                    if(config.log_prof && my_rank == 0) {log("Begin mkl validation");}
                    clock_t begin_mklv_time = std::clock();
                    validate_sp_result(&total_mtx, &config, &r, &mkl_sp_result);
                    if(config.log_prof && my_rank == 0) {log("Finish mkl validation",  begin_mklv_time, std::clock());}
                    write_sp_result_to_file(&matrix_file_name, &seg_method, &config, &r, &mkl_sp_result, &comm_size);
                }
                else{
                    if(config.log_prof && my_rank == 0) {log("Result not validated");}
                }
            }
            else if(config.mode == 'b'){
                write_bench_to_file<float, int>(&matrix_file_name, &seg_method, &config, &r, &comm_size);
            }
        }
    }

    if(config.log_prof && my_rank == 0) {log("Finish main", begin_main_time, std::clock());}

    MPI_Finalize();

    MARKER_DEINIT();

    if(config.log_prof && my_rank == 0) {log("__________ log end __________");}

    return 0;
}