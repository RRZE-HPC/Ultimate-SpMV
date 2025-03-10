#ifndef WRITE_RESULTS
#define WRITE_RESULTS

#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <math.h>

template<typename VT, typename IT>
void compute_euclid_dist(
    Result<VT, IT> *r,
    std::vector<double> *x
){
    double tmp = 0.0; // NOTE: does this need to be VT?

    #pragma omp parallel for reduction(+:tmp)
    for(int i = 0; i < x->size(); ++i){
        tmp += (double)((r->total_uspmv_result[i] - (*x)[i]) * (r->total_uspmv_result[i] - (*x)[i]));
    }

    r->euclid_dist = std::sqrt(tmp);
}

template<typename VT, typename IT>
void compute_euclid_magnitude(
    Result<VT, IT> *r,
    std::vector<double> *x
){
    double tmp = 0.0; // NOTE: does this need to be VT?

    #pragma omp parallel for reduction(+:tmp)
    for(int i = 0; i < x->size(); ++i){
        tmp += (double)((*x)[i] * (*x)[i]);
    }
    r->mkl_magnitude = std::sqrt(tmp);
}

template<typename VT, typename IT>
void write_bench_to_file(
    const std::string *seg_method,
    Config *config,
    Result<VT, IT> *r,
    int comm_size
){
    std::fstream working_file;
    int width = 32;

    int num_omp_threads;

#ifndef __CUDACC__
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_omp_threads = omp_get_num_threads();
    }
#else
    num_omp_threads = 1;
#endif
#endif
    // Print parameters
    working_file.open(config->output_filename_bench, std::fstream::in | std::fstream::out | std::fstream::app);
    working_file << config->matrix_file_name << " with ";
#ifdef USE_MPI
    working_file << comm_size << " MPI processes, and ";
#endif 
#ifdef __CUDACC__
    working_file << config->num_blocks << " block(s), and " << config->tpb << " thread(s) per block" << std::endl;
#else
    working_file << num_omp_threads << " thread(s) per proc" << std::endl;
#endif 
    working_file << "kernel: " << config->kernel_format; 
    working_file << ", block_vec_size: " << config->block_vec_size; 
    if(config->kernel_format == "scs"){
        working_file << ", C: " << config->chunk_size << " sigma: " << config->sigma;
        if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]"){
            std::cout << "TODO" << std::endl;
            exit(1);
            working_file << std::fixed << std::setprecision(2) << ", dp_beta: " << r->dp_beta << ", sp_beta: " << r->sp_beta;
        }
        else{
            working_file << std::fixed << std::setprecision(8) << ", beta: " << r->beta;
        }
    }
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
    working_file << ", block_vec_layout: " << "colwise";
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
    working_file << ", block_vec_layout: " << "rowwise";
#endif
    if (config->value_type == "ap[dp_sp]"){
        working_file << ", data_type: ap[dp_sp]" << ", threshold: " << std::fixed << std::setprecision(2) << config->ap_threshold_1 << ", % dp elems: " << r->total_dp_percent << ", % sp elems: " << r->total_sp_percent;    
    }
    else if(config->value_type == "ap[dp_hp]"){
        working_file << ", data_type: ap[dp_hp]" << ", threshold: " << std::fixed << std::setprecision(2) << config->ap_threshold_1 << ", % dp elems: " << r->total_dp_percent << ", % hp elems: " << r->total_hp_percent;    
    }
    else if(config->value_type == "ap[sp_hp]"){
        working_file << ", data_type: ap[sp_hp]" << ", threshold: " << std::fixed << std::setprecision(2) << config->ap_threshold_1 << ", % sp elems: " << r->total_sp_percent << ", % hp elems: " << r->total_hp_percent;    
    }
    else if(config->value_type == "ap[dp_sp_hp]"){
        working_file << ", data_type: ap[dp_sp_hp]" << ", threshold 1: " << std::fixed << std::setprecision(2) << config->ap_threshold_1 << ", threshold 2: " << std::fixed << std::setprecision(2) << config->ap_threshold_2 << ", % dp elems: " << r->total_dp_percent << ", % sp elems: " << r->total_sp_percent << ", % hp elems: " << r->total_hp_percent;    
    }
    else{
        if(config->value_type == "dp")
            working_file << ", data_type: double";
        else if(config->value_type == "sp")
            working_file << ", data_type: float";
        else if(config->value_type == "hp")
            working_file << ", data_type: half";
    }
#ifdef USE_MPI
#ifdef SINGLEVEC_MPI_MODE
    working_file << ", revisions: " << r->n_calls << ", seg_method: " << *seg_method << ", MPI_mode: singlevec" << std::endl;
#endif
#ifdef MULTIVEC_MPI_MODE
    working_file << ", revisions: " << r->n_calls << ", seg_method: " << *seg_method << ", MPI_mode: multivec" << std::endl;
#endif
#ifdef BULKVEC_MPI_MODE
    working_file << ", revisions: " << r->n_calls << ", seg_method: " << *seg_method << ", MPI_mode: bulkvec" << std::endl;
#endif
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

    if(config->verbose == 1){
#ifdef USE_MPI
        working_file << std::left << std::setw(width) << "Rank Idx:";
        working_file << std::left << std::setw(width) << "Per rank Elems Recvd:" << std::endl;
        working_file << std::left << std::setw(width) << "---------" << 
                        std::left << std::setw(width) << "-------------" << std::endl;
        
        for(int mpi_rank = 0; mpi_rank < comm_size; ++mpi_rank)
        working_file << std::left << std::setprecision(16) <<
            std::left << std::setw(width) << mpi_rank <<
            std::left << std::setw(width) << r->recvd_elems_per_procs[mpi_rank] << 
            std::endl;
#endif
    }

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
    const std::string *seg_method,
    Config *config,
    Result<VT, IT> *r,
    std::vector<double> *x,
    int comm_size

){
    // Used for computing error
    compute_euclid_dist<VT, IT>(r, x);
    compute_euclid_magnitude<VT, IT>(r,x);

    int width;
    double relative_diff, max_relative_diff, max_relative_diff_elem_uspmv, max_relative_diff_elem_mkl;
    double absolute_diff, max_absolute_diff, max_absolute_diff_elem_uspmv, max_absolute_diff_elem_mkl;
    std::fstream working_file;

    // IDK about all this "double" stuff
    max_relative_diff = 0.0;
    max_relative_diff_elem_uspmv = (double)(r->total_uspmv_result[0]);
    max_relative_diff_elem_mkl =(*x)[0];

    max_absolute_diff = 0.0;
    max_absolute_diff_elem_uspmv = (double)(r->total_uspmv_result[0]);
    max_absolute_diff_elem_mkl = (*x)[0];

    std::cout.precision(16);
    int num_omp_threads;

#ifndef __CUDACC__
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_omp_threads = omp_get_num_threads();
    }
#else
    num_omp_threads = 1;
#endif
#endif

    // Print parameters
    std::string output_filename;
    if(config->value_type == "dp"){
        output_filename = config->output_filename_dp;
    }
    else if(config->value_type == "sp"){
        output_filename = config->output_filename_sp;
    }
    else if(config->value_type == "hp"){
        output_filename = config->output_filename_hp;
    }
    else if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        output_filename = config->output_filename_ap;
    }
    working_file.open(output_filename, std::fstream::in | std::fstream::out | std::fstream::app);
    working_file << config->matrix_file_name << " with ";
#ifdef USE_MPI
    working_file << comm_size << " MPI processes, and ";
#endif 
#ifdef __CUDACC__
    working_file << config->num_blocks << " block(s), and " << config->tpb << " thread(s) per block" << std::endl;
#else
    working_file << num_omp_threads << " thread(s) per proc" << std::endl;
#endif 
    working_file << "kernel: " << config->kernel_format; 
    if(config->kernel_format == "scs"){
        working_file << ", C: " << config->chunk_size << ", sigma: " << config->sigma;
        if(config->value_type == "ap[dp_sp]"){
            working_file << std::fixed << std::setprecision(2) << ", dp_beta: " << r->dp_beta << ", sp_beta: " << r->sp_beta;
        }
        else if(config->value_type == "ap[dp_hp]"){
#ifdef HAVE_HALF_MATH
            working_file << std::fixed << std::setprecision(2) << ", dp_beta: " << r->dp_beta << ", hp_beta: " << r->hp_beta;
#else
            printf("ERROR: HAVE_HALF_MATH not defined.\n");
            exit(1);
#endif
        }
        else if(config->value_type == "ap[sp_hp]"){
#ifdef HAVE_HALF_MATH
            working_file << std::fixed << std::setprecision(2) << ", sp_beta: " << r->sp_beta << ", hp_beta: " << r->hp_beta;
#else
            printf("ERROR: HAVE_HALF_MATH not defined.\n");
            exit(1);
#endif
        }
        else if(config->value_type == "ap[dp_sp_hp]"){
#ifdef HAVE_HALF_MATH
            working_file << std::fixed << std::setprecision(2) << ", dp_beta: " << r->dp_beta << ", sp_beta: " << r->sp_beta << ", hp_beta: " << r->hp_beta;
#else
            printf("ERROR: HAVE_HALF_MATH not defined.\n");
            exit(1);
#endif
        }
        else{
            working_file << std::fixed << std::setprecision(8) << ", beta: " << r->beta;
        }
    }
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
    working_file << ", block_vec_layout: " << "colwise";
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
    working_file << ", block_vec_layout: " << "rowwise";
#endif
    if(config->value_type == "dp")
        working_file << ", data_type: " << "dp";
    else if(config->value_type == "sp")
        working_file << ", data_type: " << "sp";
    else if(config->value_type == "hp")
        working_file << ", data_type: " << "hp";
    else if(config->value_type == "ap[dp_sp]")
        working_file << ", data_type: " << "ap[dp_sp]";
#ifdef HAVE_HALF_MATH
    else if(config->value_type == "ap[dp_hp]")
        working_file << ", data_type: " << "ap[dp_hp]";
    else if(config->value_type == "ap[sp_hp]")
        working_file << ", data_type: " << "ap[sp_hp]";
    else if(config->value_type == "ap[dp_sp_hp]")
        working_file << ", data_type: " << "ap[dp_sp_hp]";
#endif
    

#ifdef USE_MPI
#ifdef SINGLEVEC_MPI_MODE
    working_file << ", revisions: " << config->n_repetitions << ", seg_method: " << *seg_method << ", MPI_mode: singlevec" << std::endl;
#endif
#ifdef MULTIVEC_MPI_MODE
    working_file << ", revisions: " << config->n_repetitions << ", seg_method: " << *seg_method << ", MPI_mode: multivec" << std::endl;
#endif
#ifdef BULKVEC_MPI_MODE
    working_file << ", revisions: " << config->n_repetitions << ", seg_method: " << *seg_method << ", MPI_mode: bulkvec" << std::endl;
#endif
#else
    working_file << ", revisions: " << config->n_repetitions << std::endl;
#endif

    int result_size = r->total_uspmv_result.size();
    int n_result_digits = result_size > 0 ? (int) log10 ((double) result_size) + 1 : 1;

    // Print header
    if(config->verbose == 1){
        width = 24;

        working_file<< std::left << std::setw(n_result_digits + 8) << "vec idx:" 
                    << std::left << std::setw(n_result_digits + 8) << "row idx:"
                    << std::left << std::setw(width) << "mkl results:"
                    << std::left << std::setw(width) << "uspmv results:"
                    << std::left << std::setw(width) << "rel. diff(%):" 
                    << std::left << std::setw(width) << "abs. diff:" 
                    << std::endl;

        working_file<< std::left << std::setw(n_result_digits + 8) << "--------" 
                    << std::left << std::setw(n_result_digits + 8) << "--------"
                    << std::left << std::setw(width) << "-----------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "------------"
                    << std::left << std::setw(width) << "---------" << std::endl;
    }
    else if(config->verbose == 0){
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

    int vec_count = 0;
    for(int i = 0; i < r->total_uspmv_result.size(); ++i){

        // TODO: static casting just to make it compile... 
        relative_diff = std::abs( (static_cast<double>((*x)[i]) - r->total_uspmv_result[i]) /static_cast<double>((*x)[i]));
        absolute_diff = std::abs( static_cast<double>((*x)[i]) - r->total_uspmv_result[i]);

        // Protect against printing 'inf's
#ifdef DEBUG_MODE
        if (std::abs((*x)[i]) < 1e-25){
            // printf("WARNING: At index %i, mkl_result = %f\n", i, (*x)[i]);
            relative_diff = r->total_uspmv_result[i];
        }
#endif
        
        if(config->verbose == 1)
        {
            // Setting the width of the index column to be the number of digits of the result vector size
            working_file<< std::left << std::setw(n_result_digits + 8) << vec_count
                        << std::left << std::setw(n_result_digits + 8) << (i - (r->total_rows * vec_count))
                        << std::left << std::setprecision(16) << std::scientific << std::setw(width) << (double)((*x)[i])
                        << std::left << std::setw(width) << static_cast<double>(r->total_uspmv_result[i])
                        << std::left << std::setw(width) << 100 * relative_diff
                        << std::left  << std::setw(width) << absolute_diff;

            if((std::abs(relative_diff) > .01) || std::isinf(relative_diff)){
                working_file << std::left << std::setw(width) << "ERROR";
            }
            else if(std::abs(relative_diff) > .0001){
                working_file << std::left << std::setw(width) << "WARNING";
            }

            working_file << std::endl;
        }
        else if(config->verbose == 0)
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

        // increments RHS vector counting for block x_vector
        if((i + 1) % (r->total_rows) == 0 && i > 0)
            ++vec_count;
    }
    if(config->verbose == 0)
    {
        working_file << std::scientific
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

#ifdef USE_MKL
/**
    @brief Read in the mtx struct to csr format, and use the mkl_dcsrmv to validate our double precision result against mkl
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *config : struct to initialze default values and user input
*/
void validate_result(
    MtxData<double, int> *total_mtx,
    Config *config,
    Result<double, int> *r_dp,
    Result<float, int> *r_sp,
#ifdef HAVE_HALF_MATH
    Result<_Float16, int> *r_hp,
#endif
    std::vector<double> *mkl_result,
    int *metis_perm = nullptr,
    int *metis_inv_perm = nullptr
){    
#ifdef __CUDACC__
    long num_rows = total_mtx->n_rows;
    long num_cols = total_mtx->n_cols;
    long chunk_size = 1;
#else
    int num_rows = total_mtx->n_rows;
    int num_cols = total_mtx->n_cols;
    int chunk_size = 1;
#endif

    mkl_result->resize(num_rows * config->block_vec_size, 0.0);
    std::vector<double> y(num_rows * config->block_vec_size, 0.0);
    std::vector<double> mkl_result_permuted(num_rows * config->block_vec_size, 0.0);

    ScsData<double, int> *scs = new ScsData<double, int>;

    convert_to_scs<double, double, int>(total_mtx, 1, 1, scs);

    std::vector<int> row_ptrs(total_mtx->n_rows + 1);

    convert_idxs_to_ptrs(total_mtx->I, row_ptrs);
        
    if(config->value_type == "dp" || config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        for (int i = 0; i < num_rows * config->block_vec_size; i++) {
            (*mkl_result)[i] = r_dp->total_x[i];
        }
    }
    else if(config->value_type == "sp" || config->value_type == "ap[sp_hp]"){
        for (int i = 0; i < num_rows * config->block_vec_size; i++) {
            (*mkl_result)[i] = r_sp->total_x[i];
        }
    }
    else if(config->value_type == "hp"){
#ifdef HAVE_HALF_MATH
        for (int i = 0; i < num_rows * config->block_vec_size; i++) {
            (*mkl_result)[i] = r_hp->total_x[i];
        }
#endif
    }

    // To account for global metis row reordering
    if(config->seg_method == "seg-metis"){
        for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
// #ifdef COLWISE_BLOCK_VECTOR_LAYOUT
            apply_permutation(&(mkl_result_permuted[vec_idx * num_rows]), &((*mkl_result)[vec_idx * num_rows]), metis_inv_perm, num_rows);
// #endif
// #ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
//             apply_strided_permutation(&(mkl_result_permuted[vec_idx]), &((*mkl_result)[vec_idx]), metis_inv_perm, num_rows, config->block_vec_size);
// #endif
        }
    }

    for (int i = 0; i < num_rows * config->block_vec_size; i++) {
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

    for(int i = 0; i < config->n_repetitions; ++i){
        for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
        // Computes y := alpha*A*x + beta*y, for A -> m * k, 
        // mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
        
#ifdef __CUDACC__
            // TODO: This is an ugly workaround, since I can't seem to get mkl to work with nvcc
            // spmv_omp_csr<double, int>(true, &chunk_size, &num_rows, scs->chunk_ptrs.data(), scs->chunk_lengths.data(), scs->col_idxs.data(), scs->values.data(), mkl_result->data(), y.data());
            #pragma omp for schedule(static)
            for (ST row = 0; row < num_rows; ++row) {
                double sum{};
    
                #pragma omp simd simdlen(SIMD_LENGTH) reduction(+:sum)
                for (int j = scs->chunk_ptrs[row]; j < scs->chunk_ptrs[row + 1]; ++j) {
                    sum += scs->values[j] * (*mkl_result)[scs->col_idxs[j]];
                }
                y[row] = sum;
            }
#else
            if(config->seg_method == "seg-metis"){
                mkl_dcsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], scs->values.data(), scs->col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &(mkl_result_permuted[num_rows * vec_idx]), &beta, &(y[num_rows * vec_idx]));
            }
            else{
                mkl_dcsrmv(&transa, &num_rows, &num_cols, &alpha, &matdescra[0], scs->values.data(), scs->col_idxs.data(), row_ptrs.data(), &(row_ptrs.data())[1], &((*mkl_result)[num_rows * vec_idx]), &beta, &(y[num_rows * vec_idx]));
            }
#endif
        }
        if(config->seg_method == "seg-metis"){
            std::swap(mkl_result_permuted, y);
            std::swap(*mkl_result, mkl_result_permuted);
        }
        else{
            std::swap(*mkl_result, y);
        }
    }

    delete scs;
}

#endif
#endif