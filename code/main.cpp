#include "spmv.h"
#include "mtx-reader.h"
#include "vectors.h"
#include "mkl.h"

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
    @brief Select the matrix format, and benchmark the spmvm kernal
    @param *config : struct to initialze default values and user input
    @param *mtx : mtx data struct, read in from matrix market format reader mtx-reader.h
    @param *work_sharing_arr : the array describing the partitioning of the rows of the global mtx struct
    @param *y_out : the vector declared to either hold the process local result, 
        or the global result if verification is selected as an option
    @param *defaults : a DefaultValues struct, in which default values of x and y can be defined
    @param *x_in : if one wishes to start with a pre-defined x vector
*/
template <typename VT, typename IT>
static void
bench_spmv(
    Config *config,
    MtxData<VT, IT> *local_mtx,
    const int *work_sharing_arr,
    std::vector<VT> *y_out,
    std::vector<VT> *x_out,
    DefaultValues<VT, IT> *defaults,
    BenchmarkResult<VT, IT> *r,
    const std::vector<VT> *x_in = nullptr
)
{
    if(config->kernel_format == "csr"){
        // bench_spmv_csr<VT, IT>(
        //     config,
        //     local_mtx,
        //     work_sharing_arr,
        //     y_out,
        //     x_out,
        //     defaults,
        //     r,
        //     x_in
        // );
    }
    else if(config->kernel_format == "ellrm"){}
    else if(config->kernel_format == "ellcm"){}
    else if(config->kernel_format == "ell"){
        // bench_spmv_ell<VT, IT>(
        //     config,
        //     local_mtx,
        //     work_sharing_arr,
        //     y_out,
        //     x_out,
        //     defaults,
        //     r,
        //     x_in
        // );
    }
    else if(config->kernel_format == "scs"){
        bench_spmv_scs<VT, IT>(
            config,
            local_mtx,
            work_sharing_arr,
            y_out,
            x_out,
            defaults,
            r,
            x_in
        );
    }
    else{
        fprintf(stderr, "ERROR: SpMV format for kernel %s is not implemented.\n", config->kernel_format.c_str());
    }
}

/**
    @brief Write the double precision comparison results to an external text file for validation
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *y_out : the vector declared to either hold the process local result, 
        or the global result if verification is selected as an option
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
    @param *x : the output from the mkl routine, against which we verify our spmvm result
*/
void write_dp_result_to_file(
    const std::string *file_name_str,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<double, int> *r,
    std::vector<double> *x
){
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    char filename[] = "spmv_mkl_compare.txt";
    int width;
    double relative_diff, max_relative_diff, max_relative_diff_elem_spmvm, max_relative_diff_elem_mkl;
    double absolute_diff, max_absolute_diff, max_absolute_diff_elem_spmvm, max_absolute_diff_elem_mkl;
    std::fstream appendFileToWorkWith;

    max_relative_diff = 0;
    max_relative_diff_elem_spmvm = r->total_spmvm_result[0];
    max_relative_diff_elem_mkl = (*x)[0];

    max_absolute_diff = 0;
    max_absolute_diff_elem_spmvm = r->total_spmvm_result[0];
    max_absolute_diff_elem_mkl = (*x)[0];

    std::cout.precision(16);

    // Print parameters
    appendFileToWorkWith.open(filename, std::fstream::in | std::fstream::out | std::fstream::app);
    appendFileToWorkWith << *file_name_str << " with " << comm_size << " MPI processes" << std::endl; 
    appendFileToWorkWith << "C: " << config->chunk_size << ", data_type: " <<
    'd' << ", revisions: " << config->n_repetitions << ", and seg_method: " << *seg_method << std::endl;
    appendFileToWorkWith << std::endl;

    // Print header
    if(config->verbose_validation == 1){
        width = 16;

        appendFileToWorkWith << std::left << std::setw(width) << "mkl results:"
                    << std::left << std::setw(width) << "spmv results:"
                    << std::left << std::setw(width) << "rel. diff(%):" 
                    << std::left << std::setw(width) << "abs. diff:" << std::endl;

        appendFileToWorkWith << std::left << std::setw(width) << "-----------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "---------" << std::endl;
    }
    else if(config->verbose_validation == 0){
        width = 18;
        appendFileToWorkWith 
                    << std::left << std::setw(width-2) << "mkl rel. elem:"
                    << std::left << std::setw(width) << "spmvm rel. elem:"
                    << std::left << std::setw(width) << "MAX rel. diff(%):" 
                    << std::left << std::setw(width-1) << "mkl abs. elem:"
                    << std::left << std::setw(width) << "spmvm abs. elem:"
                    << std::left << std::setw(width) << "MAX abs. diff:"
                    << std::endl;

        appendFileToWorkWith 
                    << std::left << std::setw(width-2) << "-------------"
                    << std::left << std::setw(width) << "---------------"
                    << std::left << std::setw(width) << "----------------"
                    << std::left << std::setw(width-1) << "-------------"
                    << std::left << std::setw(width) << "---------------"
                    << std::left << std::setw(width) << "-------------" << std::endl;
    }
    for(int i = 0; i < r->total_spmvm_result.size(); ++i){
        relative_diff = ((*x)[i] - r->total_spmvm_result[i])/(*x)[i];
        absolute_diff = abs((*x)[i] - r->total_spmvm_result[i]);
        
        if(config -> verbose_validation == 1)
        {
            appendFileToWorkWith << std::left << std::setw(width) << (*x)[i]
                        << std::left << std::setw(width) << r->total_spmvm_result[i]
                        << std::left << std::setw(width) << 100 * relative_diff
                        << std::left  << std::setw(width) << absolute_diff;

            if((abs(relative_diff) > .01) || std::isnan(relative_diff) || std::isinf(relative_diff)){
                appendFileToWorkWith << std::left << std::setw(width) << "ERROR";
            }
            else if(abs(relative_diff) > .0001){
                appendFileToWorkWith << std::left << std::setw(width) << "WARNING";
            }

        appendFileToWorkWith << std::endl;
        }
        else if(config -> verbose_validation == 0)
        {
            if(relative_diff > max_relative_diff){
                max_relative_diff = relative_diff;
                max_relative_diff_elem_spmvm = r->total_spmvm_result[i];
                max_relative_diff_elem_mkl = (*x)[i];
            }
            if(absolute_diff > max_absolute_diff){
                max_absolute_diff = absolute_diff;
                max_absolute_diff_elem_spmvm = r->total_spmvm_result[i];
                max_absolute_diff_elem_mkl = (*x)[i];
            }
        }
        else
        {
            std::cout << "Validation verbose level not recognized" << std::endl;
            exit(1);
        }
    }
    if(config->verbose_validation == 0)
    {
        appendFileToWorkWith 
                    << std::left << std::setw(width) << max_relative_diff_elem_mkl
                    << std::left << std::setw(width) << max_relative_diff_elem_spmvm
                    << std::left << std::setw(width) << 100 * max_relative_diff
                    << std::left << std::setw(width) << max_absolute_diff_elem_mkl
                    << std::left << std::setw(width) << max_absolute_diff_elem_spmvm
                    << std::left << std::setw(width) << max_absolute_diff;
                         
        if(((abs(max_relative_diff) > .01) || std::isnan(max_relative_diff) || std::isinf(max_relative_diff))
        ||  (std::isnan(max_absolute_diff) || std::isinf(max_absolute_diff))){
        appendFileToWorkWith << std::left << std::setw(width) << "ERROR";       
        }
        else if(abs(max_relative_diff) > .0001){
        appendFileToWorkWith << std::left << std::setw(width) << "WARNING";
        }

        appendFileToWorkWith << std::endl;
    }
    appendFileToWorkWith << "\n";
    appendFileToWorkWith.close();
}


/**
    @brief Write the single precision comparison results to an external text file for validation
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *y_out : the vector declared to either hold the process local result, 
        or the global result if verification is selected as an option
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
    @param *x : the output from the mkl routine, against which we verify our spmvm result
*/
void write_sp_result_to_file(
    const std::string *file_name_str,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<float, int> *r,
    std::vector<float> *x
){
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    char filename[] = "spmv_mkl_compare.txt";
    int width;
    float relative_diff, max_relative_diff, max_relative_diff_elem_spmvm, max_relative_diff_elem_mkl;
    float absolute_diff, max_absolute_diff, max_absolute_diff_elem_spmvm, max_absolute_diff_elem_mkl;
    std::fstream appendFileToWorkWith;

    max_relative_diff = 0;
    max_relative_diff_elem_spmvm = r->total_spmvm_result[0];
    max_relative_diff_elem_mkl = (*x)[0];

    max_absolute_diff = 0;
    max_absolute_diff_elem_spmvm = r->total_spmvm_result[0];
    max_absolute_diff_elem_mkl = (*x)[0];

    std::cout.precision(8);

    // Print parameters
    appendFileToWorkWith.open(filename, std::fstream::in | std::fstream::out | std::fstream::app);
    appendFileToWorkWith << *file_name_str << " with " << comm_size << " MPI processes" << std::endl; 
    appendFileToWorkWith << "C: " << config->chunk_size << ", data_type: " <<
    'f' << ", revisions: " << config->n_repetitions << ", and seg_method: " << *seg_method << std::endl;
    appendFileToWorkWith << std::endl;

    // Print header
    if(config->verbose_validation == 1){
        width = 16;

        appendFileToWorkWith << std::left << std::setw(width) << "mkl results:"
                    << std::left << std::setw(width) << "spmv results:"
                    << std::left << std::setw(width) << "rel. diff(%):" 
                    << std::left << std::setw(width) << "abs. diff:" << std::endl;

        appendFileToWorkWith << std::left << std::setw(width) << "-----------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "---------" << std::endl;
    }
    else if(config->verbose_validation == 0){
        width = 18;
        appendFileToWorkWith 
                    << std::left << std::setw(width-2) << "mkl rel. elem:"
                    << std::left << std::setw(width) << "spmvm rel. elem:"
                    << std::left << std::setw(width) << "MAX rel. diff(%):" 
                    << std::left << std::setw(width-1) << "mkl abs. elem:"
                    << std::left << std::setw(width) << "spmvm abs. elem:"
                    << std::left << std::setw(width) << "MAX abs. diff:"
                    << std::endl;

        appendFileToWorkWith 
                    << std::left << std::setw(width-2) << "-------------"
                    << std::left << std::setw(width) << "---------------"
                    << std::left << std::setw(width) << "----------------"
                    << std::left << std::setw(width-1) << "-------------"
                    << std::left << std::setw(width) << "---------------"
                    << std::left << std::setw(width) << "-------------" << std::endl;
    }
    for(int i = 0; i < r->total_spmvm_result.size(); ++i){
        relative_diff = ((*x)[i] - r->total_spmvm_result[i])/(*x)[i];
        absolute_diff = abs((*x)[i] - r->total_spmvm_result[i]);
        
        if(config -> verbose_validation == 1)
        {
            appendFileToWorkWith << std::left << std::setw(width) << (*x)[i]
                        << std::left << std::setw(width) << r->total_spmvm_result[i]
                        << std::left << std::setw(width) << 100 * relative_diff
                        << std::left  << std::setw(width) << absolute_diff;

            if((abs(relative_diff) > .01) || std::isnan(relative_diff) || std::isinf(relative_diff)){
                appendFileToWorkWith << std::left << std::setw(width) << "ERROR";
            }
            else if(abs(relative_diff) > .0001){
                appendFileToWorkWith << std::left << std::setw(width) << "WARNING";
            }

        appendFileToWorkWith << std::endl;
        }
        else if(config -> verbose_validation == 0)
        {
            if(relative_diff > max_relative_diff){
                max_relative_diff = relative_diff;
                max_relative_diff_elem_spmvm = r->total_spmvm_result[i];
                max_relative_diff_elem_mkl = (*x)[i];
            }
            if(absolute_diff > max_absolute_diff){
                max_absolute_diff = absolute_diff;
                max_absolute_diff_elem_spmvm = r->total_spmvm_result[i];
                max_absolute_diff_elem_mkl = (*x)[i];
            }
        }
        else
        {
            std::cout << "Validation verbose level not recognized" << std::endl;
            exit(1);
        }
    }
    if(config->verbose_validation == 0)
    {
        appendFileToWorkWith 
                    << std::left << std::setw(width) << max_relative_diff_elem_mkl
                    << std::left << std::setw(width) << max_relative_diff_elem_spmvm
                    << std::left << std::setw(width) << 100 * max_relative_diff
                    << std::left << std::setw(width) << max_absolute_diff_elem_mkl
                    << std::left << std::setw(width) << max_absolute_diff_elem_spmvm
                    << std::left << std::setw(width) << max_absolute_diff;
                         
        if(((abs(max_relative_diff) > .01) || std::isnan(max_relative_diff) || std::isinf(max_relative_diff))
        ||  (std::isnan(max_absolute_diff) || std::isinf(max_absolute_diff))){
        appendFileToWorkWith << std::left << std::setw(width) << "ERROR";       
        }
        else if(abs(max_relative_diff) > .0001){
        appendFileToWorkWith << std::left << std::setw(width) << "WARNING";
        }

        appendFileToWorkWith << std::endl;
    }
    appendFileToWorkWith << "\n";
    appendFileToWorkWith.close();
}

/**
    @brief Read in the mtx struct to csr format, and use the mkl_dcsrmv to validate our double precision result against mkl
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
*/
void validate_dp_result(
    const std::string *file_name_str,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<double, int> *r,
    std::vector<double> *mkl_dp_result
){    
    // Root process reads in matrix again, for mkl
    MtxData<double, int> mtx = read_mtx_data<double, int>(file_name_str->c_str(), config->sort_matrix);

    int num_rows = mtx.n_rows;
    int num_cols = mtx.n_cols;

    mkl_dp_result->resize(num_rows, 0);
    std::vector<double> y(num_rows, 0);

    V<double, int> values;
    V<int, int> col_idxs;
    V<int, int> row_ptrs;

    convert_to_csr<double, int>(mtx, row_ptrs, col_idxs, values);

    // Promote values to doubles
    // std::vector<VT> values_vec(values.data(), values.data() + values.n_rows);
        
    for (int i = 0; i < num_rows; i++) {
        (*mkl_dp_result)[i] = r->total_x[i];
    }

    for (int i = 0; i < num_cols; i++) {
        y[i] = 0.0;
    }

    char transa = 'n';
    double alpha = 1.0;
    double beta = 0.0; 
    char matdescra [4] = {
        'G', // general matrix
        ' ', // ignored
        ' ', // ignored
        'C'}; // zero-based indexing (C-style)

    // Computes y := alpha*A*x + beta*y, for A -> m * k, 
    // mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    for(int i = 0; i < config->n_repetitions; ++i){
        mkl_dcsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], values.data(), col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &(*mkl_dp_result)[0], &beta, &y[0]);
        std::swap(*mkl_dp_result, y);
    }
}

/**
    @brief Read in the mtx struct to csr format, and use the mkl_scsrmv to validate our single precision result against mkl
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
*/
void validate_sp_result(
    const std::string *file_name_str,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<float, int> *r,
    std::vector<float> *mkl_sp_result
){    
    // Root process reads in matrix again, for mkl
    MtxData<float, int> mtx = read_mtx_data<float, int>(file_name_str->c_str(), config->sort_matrix);

    int num_rows = mtx.n_rows;
    int num_cols = mtx.n_cols;

    mkl_sp_result->resize(num_rows, 0);
    std::vector<float> y(num_rows, 0);

    V<float, int> values;
    V<int, int> col_idxs;
    V<int, int> row_ptrs;

    convert_to_csr<float, int>(mtx, row_ptrs, col_idxs, values);
        
    for (int i = 0; i < num_rows; i++) {
        (*mkl_sp_result)[i] = r->total_x[i];
    }

    for (int i = 0; i < num_cols; i++) {
        y[i] = 0.0;
    }

    char transa = 'n';
    float alpha = 1.0;
    float beta = 0.0; 
    char matdescra [4] = {
        'G', // general matrix
        ' ', // ignored
        ' ', // ignored
        'C'}; // zero-based indexing (C-style)

    // Computes y := alpha*A*x + beta*y, for A -> m * k, 
    // mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    for(int i = 0; i < config->n_repetitions; ++i){
        mkl_scsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], values.data(), col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &(*mkl_sp_result)[0], &beta, &y[0]);
        std::swap(*mkl_sp_result, y);
    }
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
    BenchmarkResult<VT, IT> *r)
{
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

    seg_and_send_data<VT, IT>(&local_mtx, config, seg_method, file_name, work_sharing_arr, my_rank, comm_size);

    // Each process must initially allocate space for total y vector, eventhough it is potentially resized later
    std::vector<VT> y_out(work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank], 0);

    // Each process needs to return the original local_x vector it uses in the multiplication, for optional validation 
    std::vector<VT> x_out(work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank], 0);

    DefaultValues<VT, IT> default_values;

    // Calculate y_out, which will either the global or local solution vector
    bench_spmv<VT, IT>(
        config,
        &local_mtx,
        work_sharing_arr,
        &y_out,
        &x_out,
        &default_values,
        r
    );

    // Assign proc local x and y to benchmark result object
    r->x_out = x_out;
    r->y_out = y_out;

    if (config->validate_result)
    {
        // TODO: is the size correct here?
        std::vector<VT> total_spmvm_result(work_sharing_arr[comm_size], 0);
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

        // Collect each of the process local x and y vectors to a global/total vector for validation
        if (typeid(VT) == typeid(double)){
            MPI_Allgatherv(&y_out[0],
                            y_out.size(),
                            MPI_DOUBLE,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            MPI_COMM_WORLD);

            MPI_Allgatherv(&x_out[0],
                            x_out.size(),
                            MPI_DOUBLE,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            MPI_COMM_WORLD);
        }
        else if (typeid(VT) == typeid(float)){
            MPI_Allgatherv(&y_out[0],
                            y_out.size(),
                            MPI_FLOAT,
                            &total_spmvm_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            MPI_COMM_WORLD);

            MPI_Allgatherv(&x_out[0],
                            x_out.size(),
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

/**
    @brief Scan user cli input to variables, and verify that the entered parameters are valid
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *value_type : either single precision (/float/-sp) or double precision (/double/-dp)
    @param *random_init_x : decides if our generated x-vector is randomly generated, 
        or made from the default value defined in the DefaultValues struct
    @param *config : struct to initialze default values and user input
*/
void verify_and_assign_inputs(
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
        fprintf(stderr, "Usage: %s martix-market-filename kernel_format [options]\n"
                        "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand-x [%i], -sp/dp [%s], -seg-nnz/seg-rows [%s], -validate [%i], -verbose [%i], -mode [%c]\n",
                argv[0], config->chunk_size, config->sigma, config->n_repetitions, *random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode);
        exit(1);
    }

    *file_name_str = argv[1];
    config->kernel_format = argv[2];

    int args_start_index = 3;
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
        else if (arg == "-verbose")
        {
            config->verbose_validation = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->verbose_validation != 0 && config->verbose_validation != 1)
            {
                fprintf(stderr, "ERROR: Only validation verbosity levels 0 and 1 are supported.\n");
                exit(1);
            }
        }
        else if (arg == "-validate")
        {
            config->validate_result = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->validate_result != 0 && config->validate_result != 1)
            {
                fprintf(stderr, "ERROR: You can only choose to validate result (1, i.e. yes) or not (0, i.e. no).\n");
                exit(1);
            }
        }
        else if (arg == "-mode")
        {
            config->mode = *argv[++i]; // i.e. grab the NEXT

            if (config->mode != 'b' && config->mode != 's')
            {
                fprintf(stderr, "ERROR: Only bench (b) and solve (s) modes are supported.\n");
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

    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MARKER_INIT();

    verify_and_assign_inputs(argc, argv, &file_name_str, &seg_method, &value_type, &random_init_x, &config);

    if (value_type == "dp")
    {
        BenchmarkResult<double, int> r;
        compute_result<double, int>(&file_name_str, &seg_method, &config, &r);

        if(my_rank == 0){
            std::cout << "SPMVM(s) completed. Mode: " << config.mode << std::endl;

            if(config.mode == 's'){
                if(config.validate_result){
                    std::cout << "Validating..." << std::endl;
                    std::vector<double> mkl_dp_result;
                    validate_dp_result(&file_name_str, &seg_method, &config, &r, &mkl_dp_result);
                    write_dp_result_to_file(&file_name_str, &seg_method, &config, &r, &mkl_dp_result);
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
        compute_result<float, int>(&file_name_str, &seg_method, &config, &r);

        if(my_rank == 0){
            std::cout << "SPMVM(s) completed. Mode: " << config.mode << std::endl;

            if(config.mode == 's'){
                if(config.validate_result){
                    std::cout << "Validating..." << std::endl;
                    std::vector<float> mkl_sp_result;
                    validate_sp_result(&file_name_str, &seg_method, &config, &r, &mkl_sp_result);
                    write_sp_result_to_file(&file_name_str, &seg_method, &config, &r, &mkl_sp_result);
                    std::cout << "See spmv_mkl_compare.txt" << std::endl;
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