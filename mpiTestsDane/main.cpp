#include "spmv.h"
#include "mtx-reader.h"
#include "vectors.h"
#include "utilities.hpp"
#include "structs.hpp"
#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "format.hpp"

#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
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

#ifdef _OPENMP
#include <omp.h>
#endif

/**
    Description...
    @param *file_name : 
    @param *y_total : 
    @param name : 
    @param sort_matrix : 
    @return
*/
template <typename VT, typename IT>
void check_if_result_valid(
    const char *file_name,
    std::vector<VT> *y_total,
    const std::string name,
    bool sort_matrix)
{
    // DefaultValues<VT, IT> defaults;

    // Root proc reads all of mtx
    // MtxData<VT, IT> mtx = read_mtx_data<VT, IT>(file_name, sort_matrix);

    // std::vector<VT> x_total(mtx.n_cols, 0);
    // std::uninitialized_fill_n(x_total.data(), x_total.n_rows, idk.y);

    // recreate_x_total()

    // TODO: Only works since seed is same. Not flexible to swapping.
    // init_std_vec_with_ptr_or_value<VT>(x_total, mtx.n_cols, nullptr, defaults.x);

    // bool is_result_valid = spmv_verify<VT, IT>(name, mtx, x_total, *y_total);

    // std::cout << result_valid << std::endl;

    // TODO: come back to validity checking later
    // if (is_result_valid)
    // {
    //     printf("Results valid.\n");
    // }
    // else
    // {
    //     printf("Results NOT valid.\n");
    // }
}

/**
    Select the matrix format, and benchmark the spmvm kernal
    @param *config : 
    @param *mtx : 
    @param *work_sharing_arr : 
    @param *y_out : 
    @param *defaults : 
    @param *x_in : 
    @return
*/
template <typename VT, typename IT>
static void
bench_spmv(
    Config *config,
    MtxData<VT, IT> *mtx,
    int *work_sharing_arr,
    std::vector<VT> *y_out,
    DefaultValues<VT, IT> *defaults,
    const std::vector<VT> *x_in = nullptr
)
{
    // BenchmarkResult r;

    // std::vector<VT> y_out;
    // std::vector<VT> x_out;

    // DefaultValues<VT, IT> default_values;

    // if (!defaults)
    // {
    //     defaults = &default_values;
    // }

    if(config->matrix_format == "csr"){
        // r = bench_spmv_csr<VT, IT>(config,
        //                            mtx,
        //                            k_entry, *defaults,
        //                            x_out, y_out, x_in);
    }
    else if(config->matrix_format == "ellrm"){}
    else if(config->matrix_format == "ellcm"){}
    else if(config->matrix_format == "ell"){
        // r = bench_spmv_ell<VT, IT>(config,
        //                            mtx,
        //                            k_entry, *defaults,
        //                            x_out, y_out, x_in);
    }
    else if(config->matrix_format == "scs"){
        // r = bench_spmv_scs<VT, IT>(config,
        //                            mtx,
        //                            k_entry, *defaults,
        //                            x_out, y_out, x_in);
        bench_spmv_scs<VT, IT>(
            config,
            mtx,
            work_sharing_arr,
            y_out,
            defaults,
            x_in
        );
    }
    else{
        fprintf(stderr, "ERROR: SpMV format for kernel %s is not implemented.\n", config->matrix_format.c_str());
    }
        // return r;

    // if (y_out_opt)
    //     *y_out_opt = std::move(y_out);

    // if (print_matrices) {
    //     printf("Matrices for kernel: %s\n", kernel_name.c_str());
    //     printf("A, is_col_major: %d\n", A.is_col_major);
    //     print(A);
    //     printf("b\n");
    //     print(b);
    //     printf("x\n");
    //     print(x);
    // }

    // return r;
}


/**
    Description...
    @param *file_name : 
    @param *seg_method : 
    @param *config : 
    @return
*/
template <typename VT, typename IT>
void compute_result(
    const std::string *file_name,
    const std::string *seg_method,
    Config *config)
{
    // BenchmarkResult result;

    // TODO: bring back matrix stats
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    // Initialize MPI variables
    IT my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Declare struct on each process
    MtxData<VT, IT> local_mtx;

    // Allocate space for work sharing array. Is populated in seg_and_send_data function
    IT work_sharing_arr[comm_size + 1];
    for(IT i = 0; i < comm_size + 1; ++i){
        work_sharing_arr[i] = IT{};
    }

    seg_and_send_data<VT, IT, ST>(&local_mtx, config, seg_method, file_name, work_sharing_arr, my_rank, comm_size);

    // Each process must initially allocate space for total y vector, eventhough it is potentially resized later
    std::vector<VT> y_out(work_sharing_arr[comm_size], 0);
    // std::vector<VT> result(work_sharing_arr[comm_size], 0);

    DefaultValues<VT, IT> default_values;

    // Calculate y_out, which will either the global or local solution vector
    bench_spmv<VT, IT>(
        config,
        &local_mtx,
        work_sharing_arr,
        &y_out,
        &default_values
    );

    if (config->verify_result)
    {
        log("verify begin\n");
        if(my_rank == 0){
            printf("\n");
            std::cout << "Resulting vector with: " << config->n_repetitions << " revisions" << std::endl; 
            for(IT i = 0; i < y_out.size(); ++i){
                std::cout << y_out[i] << std::endl;
            }
        }

            // MtxData<VT, IT> mtx = read_mtx_data<VT, IT>(file_name.c_str(), config.sort_matrix);
        
            // bool ok = spmv_verify(config.matrix_format, &mtx, x_out, y_out);
            // r.is_result_valid = ok;

        log("verify end\n");
    }

    // TODO: verify results
    // if (print_proc_local_stats)
    // {
    //     print_results(print_list, name, matrix_stats, result, n_cpu_threads, print_details);
    // }
    // if (config.verify_result)
    // {
    //     // But have root proc check results, because all processes have the same y_total
    //     if (my_rank == 0)
    //     {
    //         check_if_result_valid<VT, IT>(file_name, &y_total, name, config.sort_matrix);
    //     }
    // }
}

/**
    Description...
    @param *file_name_str : 
    @param *seg_method : 
    @param *value_type : 
    @param *random_init_x : 
    @param *config : 
    @return
*/
void verifyAndAssignInputs(
    int argc,
    char *argv[],
    std::string *file_name_str,
    std::string *seg_method,
    std::string *value_type,
    bool *random_init_x,
    Config *config)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                        "options [defaults]: -c[%li], -s[%li], -rev[%li], -rand-x[%i], -sp/dp[%s], -seg-nnz/seg-rows[%s]\n",
                argv[0], config->chunk_size, config->sigma, config->n_repetitions, *random_init_x, value_type->c_str(), seg_method->c_str());
        exit(1);
    }

    *file_name_str = argv[1];

    int args_start_index = 2;
    for (int i = args_start_index; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-c")
        {
            config->chunk_size = atoi(argv[++i]);

            if (config->chunk_size < 1)
            {
                fprintf(stderr, "ERROR: chunk size must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-s")
        {

            config->sigma = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->sigma < 1)
            {
                fprintf(stderr, "ERROR: sigma must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-rev")
        {
            config->n_repetitions = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->n_repetitions < 1)
            {
                fprintf(stderr, "ERROR: revisions must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-rand-x")
        {
            *random_init_x = true;
        }
        else if (arg == "-dp")
        {
            *value_type = "dp";
        }
        else if (arg == "-sp")
        {
            *value_type = "sp";
        }
        else if (arg == "-seg-rows")
        {
            *seg_method = "seg-rows";
        }
        else if (arg == "-seg-nnz")
        {
            *seg_method = "seg-nnz";
        }
        // else if (arg == "-bench")
        // {
        //     config->mode = "bench";
        // }
        // else if (arg == "-solver")
        // {
        //     config->mode = "solver";
        // }
        else
        {
            fprintf(stderr, "ERROR: unknown argument.\n");
            exit(1);
        }
    }

    if (config->sigma > config->chunk_size)
    {
        fprintf(stderr, "ERROR: sigma must be smaller than chunk size.\n");
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    Config config;
    std::string file_name_str{};

    // Set defaults for cl inputs
    std::string seg_method{"seg-rows"};
    std::string value_type = {"dp"};
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;
    bool random_init_x = false;

    MARKER_INIT();

    verifyAndAssignInputs(argc, argv, &file_name_str, &seg_method, &value_type, &random_init_x, &config);

    if (value_type == "sp")
    {
        compute_result<float, int>(&file_name_str, &seg_method, &config);
    }
    else if (value_type == "dp")
    {
        compute_result<double, int>(&file_name_str, &seg_method, &config);
    }

    log("benchmarking kernel: scs end\n");

    MPI_Finalize();

    log("main end\n");

    MARKER_DEINIT();

    return 0;
}