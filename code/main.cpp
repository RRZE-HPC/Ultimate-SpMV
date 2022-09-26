#include "classes_structs.hpp"

#include "mmio.h"

#include "vectors.h"
#include "utilities.hpp"
#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "write_results.hpp"

#include <algorithm>
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

#define WARM_UP_REPS 15

#ifdef _OPENMP
#include <omp.h>
#endif

inline void sort_perm(int *arr, int *perm, int len, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] < arr[b]); });
    } else {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
    }
}

template <typename VT, typename IT>
void read_mtx(
    const std::string matrix_file_name,
    Config config,
    MtxData<VT, IT> *total_mtx,
    int my_rank)
{
    // (*total_mtx) = read_mtx_data<VT, IT>(matrix_file_name.c_str(), config.sort_matrix);

    // std::cout << "Ive read the matrix" << std::endl;
    // std::cout << "nnz: " << total_mtx.nnz << std::endl;
    // std::cout << "n_rows: " << total_mtx.n_rows << std::endl;
    // std::cout << "n_cols: " << total_mtx.n_cols << std::endl;
    // (const char *fname, int *M_, int *N_, int *nz_,
    //     double **val_, int **I_, int **J_)

    // std::cout <<  matrix_file_name.c_str() << " : " << typeid(VT).name() << ", C = " << config.chunk_size << std::endl;
    // exit(0);

    char* filename = const_cast<char*>(matrix_file_name.c_str());
    IT nrows, ncols, nnz;
    VT *val_ptr;
    IT *I_ptr;
    IT *J_ptr;

    MM_typecode matcode;
    FILE *f;

    if ((f = fopen(filename, "r")) == NULL) {printf("Unable to open file");}

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", filename);
        // return -1;
    }

    fclose(f);

    // bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode)||mm_is_integer(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool symm_flag = mm_is_symmetric(matcode);
    bool pattern_flag = mm_is_pattern(matcode);

    if(!compatible_flag)
    {
        printf("The matrix market file provided is not supported.\n Reason :\n");
        if(!mm_is_sparse(matcode))
        {
            printf(" * matrix has to be sparse\n");
        }

        if(!mm_is_real(matcode) && !(mm_is_pattern(matcode)))
        {
            printf(" * matrix has to be real or pattern\n");
        }

        if(!mm_is_symmetric(matcode) && !mm_is_general(matcode))
        {
            printf(" * matrix has to be either general or symmetric\n");
        }

        exit(0);
    }

    //int ncols;
    IT *row_unsorted;
    IT *col_unsorted;
    VT *val_unsorted;

    if(mm_read_unsymmetric_sparse<VT, IT>(filename, &nrows, &ncols, &nnz, &val_unsorted, &row_unsorted, &col_unsorted) < 0)
    {
        printf("Error in file reading\n");
        exit(1);
    }
    if(nrows != ncols)
    {
        printf("Matrix not square. Currently only square matrices are supported\n");
        exit(1);
    }

    //If matrix market file is symmetric; create a general one out of it
    if(symm_flag)
    {
        // printf("Creating a general matrix out of a symmetric one\n");

        int ctr = 0;

        //this is needed since diagonals might be missing in some cases
        for(int idx=0; idx<nnz; ++idx)
        {
            ++ctr;
            if(row_unsorted[idx]!=col_unsorted[idx])
            {
                ++ctr;
            }
        }

        int new_nnz = ctr;

        IT *row_general = new IT[new_nnz];
        IT *col_general = new IT[new_nnz];
        VT *val_general = new VT[new_nnz];

        int idx_gen=0;

        for(int idx=0; idx<nnz; ++idx)
        {
            row_general[idx_gen] = row_unsorted[idx];
            col_general[idx_gen] = col_unsorted[idx];
            val_general[idx_gen] = val_unsorted[idx];
            ++idx_gen;

            if(row_unsorted[idx] != col_unsorted[idx])
            {
                row_general[idx_gen] = col_unsorted[idx];
                col_general[idx_gen] = row_unsorted[idx];
                val_general[idx_gen] = val_unsorted[idx];
                ++idx_gen;
            }
        }

        free(row_unsorted);
        free(col_unsorted);
        free(val_unsorted);

        nnz = new_nnz;

        //assign right pointers for further proccesing
        row_unsorted = row_general;
        col_unsorted = col_general;
        val_unsorted = val_general;

        // delete[] row_general;
        // delete[] col_general;
        // delete[] val_general;
    }

    //permute the col and val according to row
    IT* perm = new IT[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(row_unsorted, perm, nnz);

    IT *col = new IT[nnz];
    IT *row = new IT[nnz];
    VT *val = new VT[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        col[idx] = col_unsorted[perm[idx]];
        val[idx] = val_unsorted[perm[idx]];
        row[idx] = row_unsorted[perm[idx]];
    }

    delete[] perm;
    delete[] col_unsorted;
    delete[] val_unsorted;
    delete[] row_unsorted;

    total_mtx->values = std::vector<VT>(val, val + nnz);
    total_mtx->I = std::vector<IT>(row, row + nnz);
    total_mtx->J = std::vector<IT>(col, col + nnz);
    total_mtx->n_rows = nrows;
    total_mtx->n_cols = ncols;
    total_mtx->nnz = nnz;
    total_mtx->is_sorted = 1; // TODO: not sure
    total_mtx->is_symmetric = 0; // TODO: not sure

    delete[] val;
    delete[] row;
    delete[] col;
}

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

        if(config->comm_halos){
            for(int k = 0; k < WARM_UP_REPS; ++k){
                    communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);

                    spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                        local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                        local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

                    std::swap(dummy_x, dummy_y);

                    // if(dummy_x[0]>1.0){ // prevent compiler from eliminating loop
                    //     printf("%lf", dummy_x[local_x->size() / 2]);
                    //     exit(0);
                    // }
                    // MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        else if(!config->comm_halos){
            for(int k = 0; k < WARM_UP_REPS; ++k){
                spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

                std::swap(dummy_x, dummy_y);

                // if(dummy_x[0]>1.0){ // prevent compiler from eliminating loop
                //     printf("%lf", dummy_x[local_x->size() / 2]);
                //     exit(0);
                // }
            }
        }
        end_warm_up_loop_time = MPI_Wtime();


        // Use warm-up to calculate n_iter for real benchmark
        int n_iter; // NOTE: is it correct for the root process' iteration count to be what is used?
        if(my_rank == 0){
            n_iter = static_cast<int>((double)WARM_UP_REPS / (end_warm_up_loop_time - begin_warm_up_loop_time));
        }

        MPI_Bcast(
            &n_iter,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        double begin_bench_loop_time, end_bench_loop_time;

        begin_bench_loop_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        
        if(config->comm_halos){
            for(int k = 0; k < n_iter; ++k){
                communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);

                spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
                

                std::swap(dummy_x, dummy_y);

                // if(dummy_x[0]>1.0){ // prevent compiler from eliminating loop
                //     printf("%lf", dummy_x[local_x->size() / 2]);
                //     exit(0);
                // }
                // MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        else if(!config->comm_halos){
            for(int k = 0; k < n_iter; ++k){
                spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);
                

                std::swap(dummy_x, dummy_y);

                // if(dummy_x[0]>1.0){ // prevent compiler from eliminating loop
                //     printf("%lf", dummy_x[local_x->size() / 2]);
                //     exit(0);
                // }
            }
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
        for (int i = 0; i < config->n_repetitions; ++i)
        {
            communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);

            spmv_omp_scs_adv<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

            std::swap(*local_x, *local_y);
        }
        std::swap(*local_x, *local_y);

        // for(int i = 0; i < 5; ++i){
        //     std::cout << (*local_y)[i] << std::endl;
        // }
        // exit(0);

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

    if(config->log_prof && my_rank == 0) {log("Begin mpi_init_local_structs");}
    double begin_mils_time = MPI_Wtime();
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
    if(config->log_prof && my_rank == 0) {log("Finish mpi_init_local_structs", begin_mils_time, MPI_Wtime());}


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

    // for debugging
    // std::cout << "Proc: " << my_rank << std::endl;

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
            read_mtx<double, int>(matrix_file_name, config, &total_mtx, my_rank);
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
            read_mtx<float, int>(matrix_file_name, config, &total_mtx, my_rank);
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