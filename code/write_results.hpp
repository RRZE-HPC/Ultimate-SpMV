#ifndef WRITE_RESULTS
#define WRITE_RESULTS

#include <mkl.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <math.h>

template<typename VT, typename IT>
void compute_euclid_dist(
    Result<VT, IT> *r,
    std::vector<VT> *x
){
    long double tmp = 0; // NOTE: does this need to be VT?

    #pragma omp parallel for reduction(+:tmp)
    for(int i = 0; i < r->total_rows; ++i){
        tmp += (r->total_uspmv_result[i] - (*x)[i]) * (r->total_uspmv_result[i] - (*x)[i]);
    }

    r->euclid_dist = std::sqrt(tmp);
}

template<typename VT, typename IT>
void compute_euclid_magnitude(
    Result<VT, IT> *r,
    std::vector<VT> *x
){
    long double tmp = 0; // NOTE: does this need to be VT?

    #pragma omp parallel for reduction(+:tmp)
    for(int i = 0; i < r->total_rows; ++i){
        tmp += (*x)[i] * (*x)[i];
    }
    r->mkl_magnitude = std::sqrt(tmp);
}

template<typename VT, typename IT>
void write_bench_to_file(
    const std::string *matrix_file_name,
    const std::string *seg_method,
    Config *config,
    Result<VT, IT> *r,
    int comm_size
){
    std::fstream working_file;
    int width = 32;

    int num_omp_threads;

    #pragma omp parallel
    {
        num_omp_threads = omp_get_num_threads();
    }

    // Print parameters
    working_file.open(config->output_filename_bench, std::fstream::in | std::fstream::out | std::fstream::app);
    working_file << *matrix_file_name << " with ";
#ifdef USE_MPI
    working_file << comm_size << " MPI processes, and ";
#endif 
#ifdef __CUDACC__
    working_file << config->num_blocks << " blocks, and " << config->tpb << " threads per block" << std::endl;
#else
    working_file << num_omp_threads << " thread(s) per proc" << std::endl;
#endif 
    working_file << "kernel: " << config->kernel_format; 
    if(config->kernel_format == "scs"){
        working_file << ", C: " << config->chunk_size << " sigma: " << config->sigma;
        if(config->value_type == "mp"){
            working_file << std::fixed << std::setprecision(2) << ", hp_beta: " << r->hp_beta << ", lp_beta: " << r->lp_beta;
        }
        else{
            working_file << std::fixed << std::setprecision(2) << ", beta: " << r->beta;
        }
    }
    if (config->value_type == "mp"){
        working_file << ", data_type: mp" << ", threshold: " << std::fixed << std::setprecision(2) << config->bucket_size << ", % hp elems: " << r->total_hp_percent << ", % lp elems: " << r->total_lp_percent;    
    }
    else{
        working_file << ", data_type: " << typeid(VT).name();
    }
#ifdef USE_MPI
    working_file << ", revisions: " << r->n_calls << ", seg_method: " << *seg_method << std::endl;
#else
    working_file << ", revisions: " << r->n_calls << std::endl;
#endif
    working_file << std::endl;

    working_file << std::left << std::setw(width) << "Total Gflops:" <<
                    std::left << std::setw(width) << "Total Walltime:" << std::endl;
    working_file << std::left << std::setw(width) << "-------------" << 
                    std::left << std::setw(width) << "-------------" << std::endl;

    // Since performance of slowest process is taken after a barrier, it doesn't matter which index we choose
    working_file << std::left << std::setprecision(16) <<
                    std::left << std::setw(width) << r->perfs_from_procs[0] <<
                    std::left << std::setw(width) << r->total_walltime <<  std::endl;

    working_file << std::endl;
}


/**
    @brief Write the comparison results to an external text file for validation
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *y_out : the vector declared to either hold the process local result, 
        or the global result if verification is selected as an option
    @param *r : a Result struct, in which results of the computations are stored
    @param *x : the output from the mkl routine, against which we verify our uspmv result
*/
template<typename VT, typename IT>
void write_result_to_file(
    const std::string *matrix_file_name,
    const std::string *seg_method,
    Config *config,
    Result<VT, IT> *r,
    std::vector<VT> *x,
    int comm_size

){
    // Used for assessing error
    compute_euclid_dist<VT, IT>(r, x);
    compute_euclid_magnitude(r,x);

    int width;
    long double relative_diff, max_relative_diff, max_relative_diff_elem_uspmv, max_relative_diff_elem_mkl;
    long double absolute_diff, max_absolute_diff, max_absolute_diff_elem_uspmv, max_absolute_diff_elem_mkl;
    std::fstream working_file;

    max_relative_diff = 0.0;
    max_relative_diff_elem_uspmv = r->total_uspmv_result[0];
    max_relative_diff_elem_mkl = (*x)[0];

    max_absolute_diff = 0.0;
    max_absolute_diff_elem_uspmv = r->total_uspmv_result[0];
    max_absolute_diff_elem_mkl = (*x)[0];

    std::cout.precision(16);
    int num_omp_threads;

    #pragma omp parallel
    {
        num_omp_threads = omp_get_num_threads();
    }

    // Print parameters
    std::string output_filename;
    if(config->value_type == "dp"){
        output_filename = config->output_filename_dp;
    }
    else if(config->value_type == "sp"){
        output_filename = config->output_filename_sp;
    }
    else if(config->value_type == "mp"){
        output_filename = config->output_filename_mp;
    }
    working_file.open(output_filename, std::fstream::in | std::fstream::out | std::fstream::app);
    working_file << *matrix_file_name << " with ";
#ifdef USE_MPI
    working_file << comm_size << " MPI processes, and ";
#endif 
    working_file << num_omp_threads << " thread(s) per proc" << std::endl; 
    working_file << "kernel: " << config->kernel_format; 
    if(config->kernel_format == "scs"){
        working_file << ", C: " << config->chunk_size << ", sigma: " << config->sigma;
        if(config->value_type == "mp"){
            working_file << std::fixed << std::setprecision(2) << ", hp_beta: " << r->hp_beta << ", lp_beta: " << r->lp_beta;
        }
        else{
            working_file << std::fixed << std::setprecision(2) << ", beta: " << r->beta;
        }
    }
    if(config->value_type == "dp"){
        working_file << ", data_type: " << "dp";
    }
    else if(config->value_type == "sp"){
        working_file << ", data_type: " << "sp";
    }
    else if(config->value_type == "mp"){
        working_file << ", data_type: " << "mp";
    }
#ifdef USE_MPI
    working_file << ", revisions: " << config->n_repetitions << ", seg_method: " << *seg_method << std::endl;
#else
    working_file << ", revisions: " << config->n_repetitions << std::endl;
#endif

    // Print header
    if(config->verbose_validation == 1){
        width = 24;

        working_file << std::left << std::setw(width) << "mkl results:"
                    << std::left << std::setw(width) << "uspmv results:"
                    << std::left << std::setw(width) << "rel. diff(%):" 
                    << std::left << std::setw(width) << "abs. diff:" 
                    << std::endl;

        working_file << std::left << std::setw(width) << "-----------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "---------" << std::endl;
    }
    else if(config->verbose_validation == 0){
        width = 18;
        working_file 
                    << std::left << std::setw(width-2) << "mkl rel. elem:"
                    << std::left << std::setw(width) << "uspmv rel. elem:"
                    << std::left << std::setw(width) << "MAX rel. diff(%):" 
                    << std::left << std::setw(width-1) << "mkl abs. elem:"
                    << std::left << std::setw(width) << "uspmv abs. elem:"
                    << std::left << std::setw(width) << "MAX abs. diff:"
                    << std::left << std::setw(width+4) << "||mkl - uspmv||_2"
                    << std::left << std::setw(width+4) << "||mkl - uspmv||/||mkl||_2"
                    << std::endl;

        working_file 
                    << std::left << std::setw(width-2) << "-------------"
                    << std::left << std::setw(width) << "---------------"
                    << std::left << std::setw(width) << "----------------"
                    << std::left << std::setw(width-1) << "-------------"
                    << std::left << std::setw(width) << "---------------"
                    << std::left << std::setw(width) << "-------------" 
                    << std::left << std::setw(width+4) << "-----------------"
                    << std::left << std::setw(width+4) << "-------------------------" << std::endl;
    }
    for(int i = 0; i < r->total_uspmv_result.size(); ++i){
        relative_diff = std::abs( ((*x)[i] - r->total_uspmv_result[i]) /(*x)[i]);
        absolute_diff = std::abs( (*x)[i] - r->total_uspmv_result[i]);

        // Protect against printing 'inf's
#ifdef DEBUG_MODE
        if (std::abs((*x)[i]) < 1e-25){
            printf("WARNING: At index %i, mkl_result = %f\n", i, (*x)[i]);
            relative_diff = r->total_uspmv_result[i];
        }
#endif
        
        if(config -> verbose_validation == 1)
        {
            working_file << std::left << std::setprecision(16) << std::setw(width) << (*x)[i]
                        << std::left << std::setw(width) << r->total_uspmv_result[i]
                        << std::left << std::setw(width) << 100 * relative_diff
                        << std::left  << std::setw(width) << absolute_diff;

            if((std::abs(relative_diff) > .01) || std::isinf(relative_diff)){
                working_file << std::left << std::setw(width) << "ERROR";
            }
            else if(std::abs(relative_diff) > .0001){
                working_file << std::left << std::setw(width) << "WARNING";
            }

        working_file << std::endl;
        // if(i > 5)
        //     exit(1);
        }
        else if(config -> verbose_validation == 0)
        {
            if(relative_diff > max_relative_diff){
                max_relative_diff = relative_diff;
                max_relative_diff_elem_uspmv = r->total_uspmv_result[i];
                max_relative_diff_elem_mkl = (*x)[i];
            }
            if(absolute_diff > max_absolute_diff){
                max_absolute_diff = absolute_diff;
                max_absolute_diff_elem_uspmv = r->total_uspmv_result[i];
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
                    << std::left << std::setw(width) << max_relative_diff_elem_uspmv
                    << std::left << std::setw(width) << 100 * max_relative_diff
                    << std::left << std::setw(width) << max_absolute_diff_elem_mkl
                    << std::left << std::setw(width) << max_absolute_diff_elem_uspmv
                    << std::left << std::setw(width) << max_absolute_diff
                    << std::left  << std::setw(width+6) << r->euclid_dist
                    << std::left  << std::setw(width+6) << r->euclid_dist / r->mkl_magnitude;
                         
        if(((std::abs(max_relative_diff) > .01) || std::isnan(max_relative_diff) || std::isinf(max_relative_diff))
        ||  (std::isnan(max_absolute_diff) || std::isinf(max_absolute_diff))){
        working_file << std::left << std::setw(width) << "ERROR";       
        }
        else if(std::abs(max_relative_diff) > .0001){
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
    @param *r : a Result struct, in which results of the comoputations are stored
*/
void validate_dp_result(
    MtxData<double, int> *total_mtx,
    Config *config,
    Result<double, int> *r,
    std::vector<double> *mkl_dp_result
){    
    int num_rows = total_mtx->n_rows;
    int num_cols = total_mtx->n_cols;

    mkl_dp_result->resize(num_rows, 0);
    std::vector<double> y(num_rows, 0);

    ScsData<double, int> scs;

    convert_to_scs<double, double, int>(total_mtx, 1, 1, &scs);

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
        mkl_dcsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], scs.values.data(), scs.col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &(*mkl_dp_result)[0], &beta, &y[0]);
        std::swap(*mkl_dp_result, y);
    }
}

/**
    @brief Read in the mtx struct to csr format, and use the mkl_scsrmv to validate our single precision result against mkl
    @param *matrix_file_name : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
    @param *r : a Result struct, in which results of the computations are stored
*/
void validate_sp_result(
    MtxData<float, int> *total_mtx,
    Config *config,
    Result<float, int> *r,
    std::vector<float> *mkl_sp_result
){    
    int num_rows = total_mtx->n_rows;
    int num_cols = total_mtx->n_cols;

    mkl_sp_result->resize(num_rows, 0);
    std::vector<float> y(num_rows, 0);

    ScsData<float, int> scs;

    convert_to_scs<float, float, int>(total_mtx, 1, 1, &scs);

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