#ifndef CLASSES_STRUCTS
#define CLASSES_STRUCTS

#include "vectors.h"
#include <ctime>
#include <mpi.h>

#ifdef USE_METIS
    #include <metis.h>
#endif

template <typename VT, typename IT>
using V = Vector<VT, IT>;
using ST = long;

// Initialize all matrices and vectors the same.
// Use -rand to initialize randomly.
static bool g_same_seed_for_every_vector = true;

template <typename VT, typename IT>
struct MtxData
{
    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<IT> I;
    std::vector<IT> J;
    std::vector<VT> values;
};

template <typename VT, typename IT>
struct ContextData
{
    // std::vector<IT> to_send_heri;
    // std::vector<IT> local_needed_heri;

    // std::vector<IT> shift_vec; //how does this work, with these on the heap?
    // std::vector<IT> incidence_vec;

    std::vector<IT> non_zero_senders;
    std::vector<IT> non_zero_receivers;

    std::vector<std::vector<IT>> send_tags;
    std::vector<std::vector<IT>> recv_tags;

    // TODO: remove and not store, do calculations earlier
    std::vector<std::vector<IT>> comm_send_idxs;
    std::vector<std::vector<IT>> comm_recv_idxs;

    // TODO: I dont think context should be holding all elements needed to send...
    std::vector<std::vector<VT>> elems_to_send;

    std::vector<IT> recv_counts_cumsum;
    std::vector<IT> send_counts_cumsum;

    IT num_local_rows;
    IT scs_padding;
    IT total_nnz;

    // what else?
};

template <typename VT, typename IT>
struct ScsData
{
    ST C{};
    ST sigma{};

    ST n_rows{};
    ST n_cols{};
    ST n_rows_padded{};
    ST n_chunks{};
    ST n_elements{}; // No. of nz + padding.
    ST nnz{};        // No. of nz only.

    V<IT, IT> chunk_ptrs;    // Chunk start offsets into col_idxs & values.
    V<IT, IT> chunk_lengths; // Length of one row in a chunk.
    V<IT, IT> col_idxs;
    V<VT, IT> values;
    V<IT, IT> old_to_new_idx;
    std::vector<int> new_to_old_idx; //inverse of above
    // TODO: ^ make V object as well?

    void permute(int *_perm_, int*  _invPerm_);
};

// Need to make sure rows aren't being permuted, that happens in convert_to_scs
template <typename VT, typename IT>
void ScsData<VT, IT>::permute(int *_perm_, int*  _invPerm_){
    int nrows = n_chunks; // <- stupid

    // TODO: not efficient, but a workaround
    int *rowPtr = new int[nrows+1];
    int *col = new int[nnz];
    double *val = new double[nnz];

    for(int i = 0; i < nrows + 1; ++i){
        rowPtr[i] = (chunk_ptrs.data())[i];
    }
    for(int i = 0; i < nnz; ++i){
        col[i] = (col_idxs.data())[i];
        val[i] = (values.data())[i];
    } 
    

    double* newVal = (double*)malloc(sizeof(double)*nnz);
        //new double[block_size*block_size*nnz];
    int* newCol = (int*)malloc(sizeof(int)*nnz);
        //new int[nnz];
    int* newRowPtr = (int*)malloc(sizeof(int)*(nrows+1));
        //new int[nrows+1];
/*
    double *newVal = (double*) malloc(sizeof(double)*nnz);
    int *newCol = (int*) malloc(sizeof(int)*nnz);
    int *newRowPtr = (int*) malloc(sizeof(int)*(nrows+1));
*/

    newRowPtr[0] = 0;
    if(_perm_ != NULL)
    {
        //first find newRowPtr; therefore we can do proper NUMA init
        int _perm_Idx=0;
        printf("nchunks = %d\n", nrows);
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            int _perm_Row = _perm_[row];
            for(int idx=rowPtr[_perm_Row]; idx<rowPtr[_perm_Row+1]; ++idx)
            {
                ++_perm_Idx;
            }
            newRowPtr[row+1] = _perm_Idx;
        }
    }
    else
    {
        for(int row=0; row<nrows+1; ++row)
        {
            newRowPtr[row] = rowPtr[row];
        }
    }
printf("perm 1\n");
    if(_perm_ != NULL)
    {
        //with NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            int _perm_Row = _perm_[row];

            for(int _perm_Idx=newRowPtr[row],idx=rowPtr[_perm_Row]; _perm_Idx<newRowPtr[row+1]; ++idx,++_perm_Idx)
            {
                //_perm_ute column-wise also
                // guard added 22.12.22
                if(col[idx] < nrows){ // col[_perm_Idx] < nrows) ?
                    newCol[_perm_Idx] = _invPerm_[col[idx]]; // permute column of "local" elements
                }
                else{
                    newCol[_perm_Idx] = col[idx]; //do not permute columns of remote elements
                }
                
                // newCol[_perm_Idx] = _invPerm_[col[idx]]; // <- old
                // printf("permIdx = %d, idx = %d, col[permIdx] = %d, col[idx] = %d\n",_perm_Idx, idx, col[_perm_Idx],col[idx] );

                newVal[_perm_Idx] = val[idx]; // in both cases, value is permuted

                // if(newCol[_perm_Idx] >= n_rows && col[_perm_Idx] < nrows){
                //     printf("permute ERROR: local element from index %d and col %d was permuted out of it's bounds to %d.\n", idx, col[idx], newCol[_perm_Idx]);
                //     exit(1);
                // }
                if (newCol[_perm_Idx] >= n_cols){
                    printf("permute ERROR: Element at index %d has blow up column index: %d.\n", _perm_Idx,newCol[_perm_Idx]);
                    exit(1);     
                }
                if (newCol[_perm_Idx] < 0){
                    printf("permute ERROR: Element at index %d has negative column index: %d.\n", _perm_Idx, newCol[_perm_Idx]);
                    exit(1);
                }
            }
        }
    printf("perm 2.1\n");
    }
    else
    {
        printf("perm 2.2\n");
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            for(int idx=newRowPtr[row]; idx<newRowPtr[row+1]; ++idx)
            {
                newCol[idx] = col[idx];
                newVal[idx] = val[idx];
            }
        }
    }
printf("perm 3\n");
    // What if our chunk size is > 1? then there will be fewer than nrows chunks
    // for(int i = 0; i < nrows + 1; ++i){
    //     chunk_ptrs[i] = newRowPtr[i];
    // } 
    for(int i = 0; i < nnz; ++i){
        col_idxs[i] = newCol[i];
        values[i] = newVal[i];
    }
printf("perm 4\n");
    //free old _perm_utations
    delete[] val;
    delete[] rowPtr;
    delete[] col;
    delete[] newVal;
    delete[] newRowPtr;
    delete[] newCol;
}

struct Config
{
    long n_els_per_row{-1}; // ell
    long chunk_size{8};    // sell-c-sigma
    long sigma{1};         // sell-c-sigma

    // Initialize rhs vector with random numbers.
    bool random_init_x{true};

    // Override values of the matrix, read via mtx file, with random numbers.
    bool random_init_A{false};

    // No. of repetitions to perform. 0 for automatic detection.
    unsigned long n_repetitions{5};

    // Verify result of SpVM.
    int validate_result = 1;

    // Verify result against solution of COO kernel.
    int verify_result_with_coo = 0;

    // Print incorrect elements from solution.
    // bool verbose_verification{true};

    // Sort rows/columns of sparse matrix before
    // converting it to a specific format.
    int sort_matrix = 1;

    int verbose_validation = 0;

    // activate profile logs, only root process
    int log_prof = 0;

    // communicate the halo elements in benchmark loop
    int comm_halos = 1;

    // Configures if the code will be executed in bench mode (b) or solve mode (s)
    char mode = 'b'; 

    // Selects the default matrix storage format
    std::string kernel_format = "scs"; 

    // filename for single precision results printing
    std::string output_filename_sp = "spmv_mkl_compare_sp.txt";

    // filename for double precision results printing
    std::string output_filename_dp = "spmv_mkl_compare_dp.txt";

    // filename for benchmark results printing
    std::string output_filename_bench = "spmv_bench.txt";

};

template <typename VT, typename IT>
struct DefaultValues
{

    VT A{2.0};
    VT x{1.00};
    VT y{};

    VT *x_values{};
    ST n_x_values{};

    VT *y_values{};
    ST n_y_values{};
};

template <typename VT, typename IT>
struct Result
{
    double perf_gflops{};
    double mem_mb{};
    std::vector<double> perfs_from_procs; // used in Gather

    unsigned int size_value_type{};
    unsigned int size_index_type{};

    unsigned long n_calls{};
    double duration_total_s{};
    double duration_kernel_s{};

    bool is_result_valid{false};
    std::string notes;

    std::string value_type_str;
    std::string index_type_str;

    uint64_t value_type_size{};
    uint64_t index_type_size{};

    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    double fill_in_percent{};
    long C{};
    long sigma{};
    long nzr{};

    bool was_matrix_sorted{false};

    double mem_m_mb{};
    double mem_x_mb{};
    double mem_y_mb{};

    double beta{};

    double cb_a_0{};
    double cb_a_nzc{};

    std::vector<VT> y_out;
    std::vector<VT> x_out;
    std::vector<VT> total_spmvm_result;
    std::vector<VT> total_x;
};

// Honestly, probably not necessary
template <typename ST>
struct MtxDataBookkeeping
{
    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    bool is_sorted{};
    bool is_symmetric{};
};

#endif