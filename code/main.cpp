#include "spmv.h"
#include "mtx-reader.h"
#include "vectors.h"

#include "utilities.hpp"
#include "structs.hpp"
#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "format.hpp"
#include "benchmarks.hpp"
#include "write_results.hpp"


#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
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
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
    @brief 
    @param 
*/
template <typename VT, typename IT>
void init_structs(
    MtxData<VT, IT> *local_mtx,
    IT *work_sharing_arr,
    Config *config,
    const std::string *seg_method,
    const std::string *file_name_str,
    std::vector<VT> *dummy_x,
    std::vector<VT> *local_x,
    std::vector<VT> *local_y,
    ContextData<IT> *local_context,
    const int *my_rank,
    const int *comm_size,
    const int *amnt_local_elems
){
    std::vector<IT> local_needed_heri;
    collect_local_needed_heri<VT, IT>(&local_needed_heri, local_mtx, work_sharing_arr, my_rank, comm_size);

    IT local_needed_heri_size = local_needed_heri.size();
    IT global_needed_heri_size;

    // TODO: Is this actually necessary?
    MPI_Allreduce(&local_needed_heri_size,
                  &global_needed_heri_size,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD
    );

    IT *global_needed_heri = new IT[global_needed_heri_size];

    for (IT i = 0; i < global_needed_heri_size; ++i)
    {
        global_needed_heri[i] = IT{};
    }

    // "to_send_heri" are all halo elements that this process is to send
    std::vector<IT> to_send_heri;
    collect_to_send_heri<IT>(
        &to_send_heri,
        &local_needed_heri,
        global_needed_heri,
        my_rank,
        comm_size
    );

    // The shift array is used in the tag-generation scheme in halo communication.
    // the row idx is the "from_proc", the column is the "to_proc", and the element is the shift
    // after the local element index to make for the incoming halo elements
    std::vector<IT> shift_arr((*comm_size) * (*comm_size), 0);
    std::vector<IT> incidence_arr((*comm_size) * (*comm_size), 0);
    // IT *shift_arr = new IT[(*comm_size) * (*comm_size)];
    // IT *incidence_arr = new IT[(*comm_size) * (*comm_size)];

    for (IT i = 0; i < (*comm_size) * (*comm_size); ++i)
    {
        shift_arr[i] = IT{};
        incidence_arr[i] = IT{};
    }

    calc_heri_shifts<IT>(
        global_needed_heri, 
        &global_needed_heri_size, 
        &shift_arr, 
        &incidence_arr, 
        comm_size
    ); // NOTE: always symmetric?

    // if(*my_rank == 1){
    //     for(int row = 0; row < *comm_size; ++row){
    //         for(int col = 0; col < *comm_size; ++col){
    //             std::cout << shift_arr[*comm_size * row + col] << ", ";
    //         }
    //         printf("\n");
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);

    // NOTE: should always be a multiple of 3
    IT local_x_needed_padding = local_needed_heri.size() / 3;
    IT x_y_padding = std::max(local_x_needed_padding, (int)config->chunk_size);

    // Prepare buffers for communication
    dummy_x->resize(x_y_padding + *amnt_local_elems, 0);
    local_x->resize(x_y_padding + *amnt_local_elems, 0);
    local_y->resize(x_y_padding + *amnt_local_elems, 0);

    // Initialize local_x, either randomly, with defaults, or with a predefined x_in
    DefaultValues<VT, IT> default_values;
    // const std::vector<VT> x_in;// = nullptr; //what to do about this?
    init_std_vec_with_ptr_or_value(
        *local_x, 
        local_x->size(),
        default_values.x, 
        config->random_init_x
    );

    // Copy x to the dummy vector, so spmvm can use this for swapping and multiplication
    std::copy(local_x->begin(), local_x->end(), dummy_x->begin());

    local_context->local_needed_heri = local_needed_heri;
    local_context->to_send_heri = to_send_heri;
    local_context->shift_arr = shift_arr;
    local_context->incidence_arr = incidence_arr;

    delete[] global_needed_heri;
}

/**
    @brief The main harness for the rest of the functions, in which we segment and execute the work to be done.
        Validation happens outside this routine
    @param *file_name : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
*/
template <typename VT, typename IT>
void compute_result(
    const std::string *file_name,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<VT, IT> *r,
    const int *my_rank,
    const int *comm_size)
{
    // TODO: bring back matrix stats
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    // Declare mtx struct on each process
    MtxData<VT, IT> local_mtx;

    // Allocate space for work sharing array. Is populated in seg_and_send_mtx function
    IT work_sharing_arr[*comm_size + 1];

    for(IT i = 0; i < *comm_size + 1; ++i){
        work_sharing_arr[i] = IT{};
    }

    seg_and_send_mtx<VT, IT>(
        &local_mtx, 
        config, 
        seg_method, 
        file_name, 
        work_sharing_arr, 
        my_rank, 
        comm_size
    );

    IT amnt_local_elems = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];

    ContextData<IT> local_context;
    std::vector<VT> local_y; //(amnt_local_elements, 0);
    std::vector<VT> local_x; //(amnt_local_elements, 0);
    std::vector<VT> dummy_x; //(size of local x, 0);

    // Resize and populate local x and y
    // populate local context
    init_structs<VT, IT>(
        &local_mtx, 
        work_sharing_arr, 
        config, 
        seg_method, 
        file_name, 
        &local_x, 
        &dummy_x,
        &local_y, 
        &local_context, 
        my_rank, 
        comm_size,
        &amnt_local_elems
    );

    // if(*my_rank == 0){
    //     for(int i = 0; i < work_sharing_arr[*comm_size]; ++i){
    //         std::cout << local_x[i] << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    bench_spmv<VT, IT>(
        config,
        &local_mtx,
        &local_context,
        work_sharing_arr,
        &local_y,
        &dummy_x,
        r,
        my_rank,
        comm_size
    );

    // if(*my_rank == 0){
    //     for(int i = 0; i < work_sharing_arr[*comm_size]; ++i){
    //         std::cout << local_y[i] << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);

    // Assign proc local x and y to benchmark result object
    r->x_out = local_x;
    r->y_out = local_y;

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

        // Collect each of the process local x and y vectors to a global/total vector for validation
        if (typeid(VT) == typeid(double)){
            MPI_Allgatherv(&local_y[0],
                            amnt_local_elems,
                            MPI_DOUBLE,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            MPI_COMM_WORLD);

            MPI_Allgatherv(&local_x[0],
                            amnt_local_elems,
                            MPI_DOUBLE,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            MPI_COMM_WORLD);
        }
        else if (typeid(VT) == typeid(float)){
            MPI_Allgatherv(&local_y[0],
                            amnt_local_elems,
                            MPI_FLOAT,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            MPI_COMM_WORLD);

            MPI_Allgatherv(&local_x[0],
                            amnt_local_elems,
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
    // if(*my_rank == 0){
    //     printf("\n");
    //     for(int i = 0; i < work_sharing_arr[*comm_size]; ++i){
    //         std::cout << (r->total_spmvm_result)[i] << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    Config config;
    std::string matrix_file_name{};

    // Set defaults for cl inputs
    std::string seg_method{"seg-rows"};
    std::string value_type = {"dp"};
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;

    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MARKER_INIT();

    verify_and_assign_inputs(argc, argv, &matrix_file_name, &seg_method, &value_type, &config);

    if (value_type == "dp")
    {
        BenchmarkResult<double, int> r;
        compute_result<double, int>(&matrix_file_name, &seg_method, &config, &r, &my_rank, &comm_size);

        if(my_rank == 0){
            std::cout << "SPMVM(s) completed. Mode: " << config.mode << std::endl;

            if(config.mode == 's'){
                if(config.validate_result){
                    std::string output_filename = {"spmv_mkl_compare_dp.txt"};
                    std::cout << "Validating..." << std::endl;
                    std::vector<double> mkl_dp_result;
                    validate_dp_result(&matrix_file_name, &seg_method, &config, &r, &mkl_dp_result);
                    write_dp_result_to_file(&output_filename, &matrix_file_name, &seg_method, &config, &r, &mkl_dp_result, &comm_size);
                    std::cout << "See " << output_filename << std::endl;
                }
                else{
                    std::cout << "Result not validated" << std::endl;
                }
            }
            else if(config.mode == 'b'){
                std::cout << "See BenchmarkResult" << std::endl;
            }
        }
    }
    else if (value_type == "sp")
    {
        BenchmarkResult<float, int> r;
        compute_result<float, int>(&matrix_file_name, &seg_method, &config, &r, &my_rank, &comm_size);

        if(my_rank == 0){
            std::cout << "SPMVM(s) completed. Mode: " << config.mode << std::endl;

            if(config.mode == 's'){
                if(config.validate_result){
                    std::string output_filename = {"spmv_mkl_compare_sp.txt"};
                    std::cout << "Validating..." << std::endl;
                    std::vector<float> mkl_sp_result;
                    validate_sp_result(&matrix_file_name, &seg_method, &config, &r, &mkl_sp_result);
                    write_sp_result_to_file(&output_filename, &matrix_file_name, &seg_method, &config, &r, &mkl_sp_result, &comm_size);
                    std::cout << "See " << output_filename << std::endl;
                }
                else{
                    std::cout << "Result not validated" << std::endl;
                }
            }
            else if(config.mode == 'b'){
                std::cout << "See BenchmarkResult" << std::endl;
            }
        }
    }

    MPI_Finalize();

    MARKER_DEINIT();

    return 0;
}