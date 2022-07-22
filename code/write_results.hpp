#ifndef WRITE_RESULTS
#define WRITE_RESULTS

#include "mkl.h"

template<typename VT, typename IT>
void write_bench_to_file(
    const std::string *output_filename,
    const std::string *matrix_file_name,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<VT, IT> *r,
    const int *comm_size
){
    std::fstream working_file;
    int width = 16;

    std::cout.precision(16);

    // Print parameters
    working_file.open(*output_filename, std::fstream::in | std::fstream::out | std::fstream::app);
    working_file << *matrix_file_name << " with " << *comm_size << " MPI processes" << std::endl; 
    working_file << "C: " << config->chunk_size << ", data_type: " <<
    'd' << ", repetitions: " << r->n_calls << ", and seg_method: " << *seg_method << std::endl;
    working_file << std::endl;

    working_file << std::left << std::setw(width) << "Perf per MPI proc:" << std::endl;
    working_file << std::left << std::setw(width) << "------------------" << std::endl;

    // Print Flops per MPI process
    for(int proc = 0; proc < *comm_size; ++proc){
        working_file << "Proc " << proc << ": ";
        working_file << r->perfs_from_procs[proc] << " MF/s" << std::endl;
    }
    working_file << std::endl;

    working_file << std::left << std::setw(width) << "Average Mflops per proc:" << std::endl;
    working_file << std::left << std::setw(width) << "-----------------------" << std::endl;

    double sum_flops = std::accumulate(r->perfs_from_procs.begin(), r->perfs_from_procs.end(), 0.0);
    working_file << sum_flops << " MF/s" << std::endl;

    // working_file << sum_flops / (double)*comm_size << " GF/s" << std::endl;
    working_file << std::endl;
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
    const std::string *output_filename,
    const std::string *matrix_file_name,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<double, int> *r,
    std::vector<double> *x,
    const int *comm_size

){
    int width;
    double relative_diff, max_relative_diff, max_relative_diff_elem_spmvm, max_relative_diff_elem_mkl;
    double absolute_diff, max_absolute_diff, max_absolute_diff_elem_spmvm, max_absolute_diff_elem_mkl;
    std::fstream working_file;

    max_relative_diff = 0;
    max_relative_diff_elem_spmvm = r->total_spmvm_result[0];
    max_relative_diff_elem_mkl = (*x)[0];

    max_absolute_diff = 0;
    max_absolute_diff_elem_spmvm = r->total_spmvm_result[0];
    max_absolute_diff_elem_mkl = (*x)[0];

    std::cout.precision(16);

    // Print parameters
    working_file.open(*output_filename, std::fstream::in | std::fstream::out | std::fstream::app);
    working_file << *matrix_file_name << " with " << *comm_size << " MPI processes" << std::endl; 
    working_file << "C: " << config->chunk_size << ", data_type: " <<
    'd' << ", revisions: " << config->n_repetitions << ", and seg_method: " << *seg_method << std::endl;
    working_file << std::endl;

    // Print header
    if(config->verbose_validation == 1){
        width = 16;

        working_file << std::left << std::setw(width) << "mkl results:"
                    << std::left << std::setw(width) << "spmv results:"
                    << std::left << std::setw(width) << "rel. diff(%):" 
                    << std::left << std::setw(width) << "abs. diff:" << std::endl;

        working_file << std::left << std::setw(width) << "-----------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "---------" << std::endl;
    }
    else if(config->verbose_validation == 0){
        width = 18;
        working_file 
                    << std::left << std::setw(width-2) << "mkl rel. elem:"
                    << std::left << std::setw(width) << "spmvm rel. elem:"
                    << std::left << std::setw(width) << "MAX rel. diff(%):" 
                    << std::left << std::setw(width-1) << "mkl abs. elem:"
                    << std::left << std::setw(width) << "spmvm abs. elem:"
                    << std::left << std::setw(width) << "MAX abs. diff:"
                    << std::endl;

        working_file 
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
            working_file << std::left << std::setw(width) << (*x)[i]
                        << std::left << std::setw(width) << r->total_spmvm_result[i]
                        << std::left << std::setw(width) << 100 * relative_diff
                        << std::left  << std::setw(width) << absolute_diff;

            if((abs(relative_diff) > .01) || std::isnan(relative_diff) || std::isinf(relative_diff)){
                working_file << std::left << std::setw(width) << "ERROR";
            }
            else if(abs(relative_diff) > .0001){
                working_file << std::left << std::setw(width) << "WARNING";
            }

        working_file << std::endl;
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
        working_file 
                    << std::left << std::setw(width) << max_relative_diff_elem_mkl
                    << std::left << std::setw(width) << max_relative_diff_elem_spmvm
                    << std::left << std::setw(width) << 100 * max_relative_diff
                    << std::left << std::setw(width) << max_absolute_diff_elem_mkl
                    << std::left << std::setw(width) << max_absolute_diff_elem_spmvm
                    << std::left << std::setw(width) << max_absolute_diff;
                         
        if(((abs(max_relative_diff) > .01) || std::isnan(max_relative_diff) || std::isinf(max_relative_diff))
        ||  (std::isnan(max_absolute_diff) || std::isinf(max_absolute_diff))){
        working_file << std::left << std::setw(width) << "ERROR";       
        }
        else if(abs(max_relative_diff) > .0001){
        working_file << std::left << std::setw(width) << "WARNING";
        }

        working_file << std::endl;
    }
    working_file << "\n";
    working_file.close();
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
    const std::string *output_filename,
    const std::string *matrix_file_name,
    const std::string *seg_method,
    Config *config,
    BenchmarkResult<float, int> *r,
    std::vector<float> *x,
    const int *comm_size
){

    int width;
    float relative_diff, max_relative_diff, max_relative_diff_elem_spmvm, max_relative_diff_elem_mkl;
    float absolute_diff, max_absolute_diff, max_absolute_diff_elem_spmvm, max_absolute_diff_elem_mkl;
    std::fstream working_file;

    max_relative_diff = 0;
    max_relative_diff_elem_spmvm = r->total_spmvm_result[0];
    max_relative_diff_elem_mkl = (*x)[0];

    max_absolute_diff = 0;
    max_absolute_diff_elem_spmvm = r->total_spmvm_result[0];
    max_absolute_diff_elem_mkl = (*x)[0];

    std::cout.precision(8);

    // Print parameters
    working_file.open(*output_filename, std::fstream::in | std::fstream::out | std::fstream::app);
    working_file << *matrix_file_name << " with " << *comm_size << " MPI processes" << std::endl; 
    working_file << "C: " << config->chunk_size << ", data_type: " <<
    'f' << ", revisions: " << config->n_repetitions << ", and seg_method: " << *seg_method << std::endl;
    working_file << std::endl;

    // Print header
    if(config->verbose_validation == 1){
        width = 16;

        working_file << std::left << std::setw(width) << "mkl results:"
                    << std::left << std::setw(width) << "spmv results:"
                    << std::left << std::setw(width) << "rel. diff(%):" 
                    << std::left << std::setw(width) << "abs. diff:" << std::endl;

        working_file << std::left << std::setw(width) << "-----------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "---------" << std::endl;
    }
    else if(config->verbose_validation == 0){
        width = 18;
        working_file 
                    << std::left << std::setw(width-2) << "mkl rel. elem:"
                    << std::left << std::setw(width) << "spmvm rel. elem:"
                    << std::left << std::setw(width) << "MAX rel. diff(%):" 
                    << std::left << std::setw(width-1) << "mkl abs. elem:"
                    << std::left << std::setw(width) << "spmvm abs. elem:"
                    << std::left << std::setw(width) << "MAX abs. diff:"
                    << std::endl;

        working_file 
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
            working_file << std::left << std::setw(width) << (*x)[i]
                        << std::left << std::setw(width) << r->total_spmvm_result[i]
                        << std::left << std::setw(width) << 100 * relative_diff
                        << std::left  << std::setw(width) << absolute_diff;

            if((abs(relative_diff) > .01) || std::isnan(relative_diff) || std::isinf(relative_diff)){
                working_file << std::left << std::setw(width) << "ERROR";
            }
            else if(abs(relative_diff) > .0001){
                working_file << std::left << std::setw(width) << "WARNING";
            }

        working_file << std::endl;
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
        working_file 
                    << std::left << std::setw(width) << max_relative_diff_elem_mkl
                    << std::left << std::setw(width) << max_relative_diff_elem_spmvm
                    << std::left << std::setw(width) << 100 * max_relative_diff
                    << std::left << std::setw(width) << max_absolute_diff_elem_mkl
                    << std::left << std::setw(width) << max_absolute_diff_elem_spmvm
                    << std::left << std::setw(width) << max_absolute_diff;
                         
        if(((abs(max_relative_diff) > .01) || std::isnan(max_relative_diff) || std::isinf(max_relative_diff))
        ||  (std::isnan(max_absolute_diff) || std::isinf(max_absolute_diff))){
        working_file << std::left << std::setw(width) << "ERROR";       
        }
        else if(abs(max_relative_diff) > .0001){
        working_file << std::left << std::setw(width) << "WARNING";
        }

        working_file << std::endl;
    }
    working_file << "\n";
    working_file.close();
}

/**
    @brief Read in the mtx struct to csr format, and use the mkl_dcsrmv to validate our double precision result against mkl
    @param *matrix_file_name : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
*/
void validate_dp_result(
    MtxData<double, int> *total_mtx,
    Config *config,
    BenchmarkResult<double, int> *r,
    std::vector<double> *mkl_dp_result
){    
    int num_rows = total_mtx->n_rows;
    int num_cols = total_mtx->n_cols;

    mkl_dp_result->resize(num_rows, 0);
    std::vector<double> y(num_rows, 0);

    ScsData<double, int> scs;

    convert_to_scs<double, int>(total_mtx, 1, 1, &scs);

    V<int, int>row_ptrs(total_mtx->n_rows + 1);

    convert_idxs_to_ptrs(total_mtx->I, row_ptrs);
        
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
        // mkl_dcsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], values.data(), col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &(*mkl_dp_result)[0], &beta, &y[0]);
        mkl_dcsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], scs.values.data(), scs.col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &(*mkl_dp_result)[0], &beta, &y[0]);
        std::swap(*mkl_dp_result, y);
    }
}

/**
    @brief Read in the mtx struct to csr format, and use the mkl_scsrmv to validate our single precision result against mkl
    @param *matrix_file_name : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a BenchmarkResult struct, in which results of the benchmark are stored
*/
void validate_sp_result(
    MtxData<float, int> *total_mtx,
    Config *config,
    BenchmarkResult<float, int> *r,
    std::vector<float> *mkl_sp_result
){    
    int num_rows = total_mtx->n_rows;
    int num_cols = total_mtx->n_cols;

    mkl_sp_result->resize(num_rows, 0);
    std::vector<float> y(num_rows, 0);

    ScsData<float, int> scs;

    convert_to_scs<float, int>(total_mtx, 1, 1, &scs);

    V<int, int>row_ptrs(total_mtx->n_rows + 1);

    convert_idxs_to_ptrs(total_mtx->I, row_ptrs);
        
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
        mkl_scsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], scs.values.data(), scs.col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &(*mkl_sp_result)[0], &beta, &y[0]);
        std::swap(*mkl_sp_result, y);
    }
}
#endif