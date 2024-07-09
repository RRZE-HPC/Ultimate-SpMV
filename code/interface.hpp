#ifndef INTERFACE_H
#define INTERFACE_H

#include "vectors.h"
#include <numeric>
#include <limits>
#include <algorithm>

#define RESTRICT				__restrict__


template <typename VT, typename IT>
using V = Vector<VT, IT>;
using ST = long;

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

    void permute(IT *_perm_, IT*  _invPerm_);
    void print(void);
};

template <typename VT, typename IT>
void ScsData<VT, IT>::permute(IT *_perm_, IT*  _invPerm_){
    int nrows = n_rows; // <- stupid

    // TODO: not efficient, but a workaround
    IT *rowPtr = new IT[nrows+1];
    IT *col = new IT[nnz];
    VT *val = new VT[nnz];

    for(int i = 0; i < nrows + 1; ++i){
        rowPtr[i] = (chunk_ptrs.data())[i];
    }
    for(int i = 0; i < nnz; ++i){
        col[i] = (col_idxs.data())[i];
        val[i] = (values.data())[i];
    } 
    

    VT* newVal = (VT*)malloc(sizeof(VT)*nnz);
        //new double[block_size*block_size*nnz];
    IT* newCol = (IT*)malloc(sizeof(IT)*nnz);
        //new int[nnz];
    IT* newRowPtr = (IT*)malloc(sizeof(IT)*(nrows+1));
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
        IT _perm_Idx=0;
#ifdef DEBUG_MODE
    // if(my_rank == 0){printf("nrows = %d\n", nrows);}
#endif
        
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            IT _perm_Row = _perm_[row];
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

    if(_perm_ != NULL)
    {
        //with NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            IT _perm_Row = _perm_[row];

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
                    printf("SCS permute ERROR: Element at index %d has blow up column index: %d.\n", _perm_Idx,newCol[_perm_Idx]);
                    exit(1);     
                }
                if (newCol[_perm_Idx] < 0){
                    printf("SCS permute ERROR: Element at index %d has negative column index: %d.\n", _perm_Idx, newCol[_perm_Idx]);
                    exit(1);
                }
            }
        }
    }
    else
    {
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

    for(int i = 0; i < nrows + 1; ++i){
        chunk_ptrs[i] = newRowPtr[i];
    } 
    for(int i = 0; i < nnz; ++i){
        col_idxs[i] = newCol[i];
        values[i] = newVal[i];
    }

    //free old _perm_utations
    delete[] val;
    delete[] rowPtr;
    delete[] col;
    delete[] newVal;
    delete[] newRowPtr;
    delete[] newCol;
}

template <typename VT, typename IT>
void ScsData<VT, IT>::print(void){

    std::cout << "C = " << C << std::endl;
    std::cout << "sigma = " << sigma << std::endl;
    std::cout << "n_rows = " << n_rows << std::endl;
    std::cout << "n_cols = " << n_cols << std::endl;
    std::cout << "n_rows_padded = " << n_rows_padded << std::endl;
    std::cout << "n_chunks = " << n_chunks << std::endl;
    std::cout << "n_elements = " << n_elements << std::endl;
    std::cout << "nnz = " << nnz << std::endl;

    std::cout << "chunk_ptrs = [";
    for(int i = 0; i < n_chunks + 1; ++i){
        std::cout << chunk_ptrs[i];
        if(i == n_chunks)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }

    std::cout << "chunk_lengths = [";
    for(int i = 0; i < n_chunks; ++i){
        std::cout << chunk_lengths[i];
        if(i == n_chunks - 1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }

    ST n_scs_elements = chunk_ptrs[n_chunks - 1] + chunk_lengths[n_chunks - 1] * C;

    std::cout << "col_idxs = [";
    for(int i = 0; i < n_scs_elements; ++i){
        std::cout << col_idxs[i];
        if(i == n_scs_elements - 1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }

    std::cout << "values = [";
    for(int i = 0; i < n_scs_elements; ++i){
        std::cout << values[i];
        if(i == n_scs_elements - 1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }

    std::cout << "old_to_new_idx = [";
    for(int i = 0; i < n_rows; ++i){
        std::cout << old_to_new_idx[i];
        if(i == n_rows - 1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }

    std::cout << "new_to_old_idx = [";
    for(int i = 0; i < n_rows; ++i){
        std::cout << new_to_old_idx[i];
        if(i == n_rows - 1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }
    printf("\n");
}

template <typename IT>
void generate_inv_perm(
    int *perm,
    int *inv_perm,
    int perm_len
){
    for(int i = 0; i < perm_len; ++i){
        inv_perm[perm[i]] = i;
    }
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
bool will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0)
    {
        return false;
    }
    else if (a < 0 && b > 0)
    {
        return std::numeric_limits<T>::min() / b > a;
    }
    else if (a > 0 && b < 0)
    {
        return std::numeric_limits<T>::min() / a > b;
    }
    else if (a > 0 && b > 0)
    {
        return std::numeric_limits<T>::max() / a < b;
    }
    else
    {
        T difference =
            std::numeric_limits<T>::max() + std::numeric_limits<T>::min();

        if (difference == 0)
        { // symmetric case
            return std::numeric_limits<T>::min() / a < b * T{-1};
        }
        else
        { // abs(min) > max
            T c = std::numeric_limits<T>::min() - difference;

            if (a < c || b < c)
                return true;

            T ap = a * T{-1};
            T bp = b * T{-1};

            return std::numeric_limits<T>::max() / ap < bp;
        }
    }
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_unsigned<T>::value,
              bool>::type = true>
bool will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0)
    {
        return false;
    }

    return std::numeric_limits<T>::max() / a < b;
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
bool will_add_overflow(T a, T b)
{
    if (a > 0 && b > 0)
    {
        return std::numeric_limits<T>::max() - a < b;
    }
    else if (a < 0 && b < 0)
    {
        return std::numeric_limits<T>::min() - a > b;
    }

    return false;
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_unsigned<T>::value,
              bool>::type = true>
bool will_add_overflow(T a, T b)
{
    return std::numeric_limits<T>::max() - a < b;
}

/**
    @brief Convert mtx struct to sell-c-sigma data structures.
    @param *local_mtx : process local mtx data structure, that was populated by  
    @param C : chunk height
    @param sigma : sorting scope
    @param *scs : The ScsData struct to populate with data
*/
template <typename VT, typename IT>
void convert_to_scs(
    const MtxData<VT, IT> *local_mtx,
    ST C,
    ST sigma,
    ScsData<VT, IT> *scs
    // int *work_sharing_arr = nullptr,
    // int my_rank = 0
    )
{
    scs->nnz    = local_mtx->nnz;
    scs->n_rows = local_mtx->n_rows;
    scs->n_cols = local_mtx->n_cols;

    scs->C = C;
    scs->sigma = sigma;

    if (scs->sigma % scs->C != 0 && scs->sigma != 1) {
#ifdef DEBUG_MODE
    // if(my_rank == 0){
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
        // }
#endif
    }

    if (will_add_overflow(scs->n_rows, scs->C)) {
#ifdef DEBUG_MODE
    // if(my_rank == 0){
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // }
    exit(1);
#endif        
        // return false;
    }
    scs->n_chunks      = (local_mtx->n_rows + scs->C - 1) / scs->C;

    if (will_mult_overflow(scs->n_chunks, scs->C)) {
#ifdef DEBUG_MODE
    // if(my_rank == 0){
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // }
    exit(1);
#endif   
        // return false;
    }
    scs->n_rows_padded = scs->n_chunks * scs->C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    std::vector<index_and_els_per_row> n_els_per_row(scs->n_rows_padded);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        n_els_per_row[i].first = i;
    }

    for (ST i = 0; i < local_mtx->nnz; ++i) {
        ++n_els_per_row[local_mtx->I[i]].second;
    }

    // sort rows in the scope of sigma
    if (will_add_overflow(scs->n_rows_padded, scs->sigma)) {
        fprintf(stderr, "ERROR: no. of padded rows + sigma exceeds size type.\n");
        // return false;
    }

    for (ST i = 0; i < scs->n_rows_padded; i += scs->sigma) {
        auto begin = &n_els_per_row[i];
        auto end   = (i + scs->sigma) < scs->n_rows_padded
                        ? &n_els_per_row[i + scs->sigma]
                        : &n_els_per_row[scs->n_rows_padded];

        std::sort(begin, end,
                  // sort longer rows first
                  [](const auto & a, const auto & b) {
                    return a.second > b.second;
                  });
    }

    // determine chunk_ptrs and chunk_lengths

    // TODO: check chunk_ptrs can overflow
    // std::cout << d.n_chunks << std::endl;
    scs->chunk_lengths = V<IT, IT>(scs->n_chunks); // init a vector of length d.n_chunks
    scs->chunk_ptrs    = V<IT, IT>(scs->n_chunks + 1);

    IT cur_chunk_ptr = 0;
    
    for (ST i = 0; i < scs->n_chunks; ++i) {
        auto begin = &n_els_per_row[i * scs->C];
        auto end   = &n_els_per_row[i * scs->C + scs->C];

        scs->chunk_lengths[i] =
                std::max_element(begin, end,
                    [](const auto & a, const auto & b) {
                        return a.second < b.second;
                    })->second;

        if (will_add_overflow(cur_chunk_ptr, scs->chunk_lengths[i] * (IT)scs->C)) {
            fprintf(stderr, "ERROR: chunck_ptrs exceed index type.\n");
            // return false;
        }

        scs->chunk_ptrs[i] = cur_chunk_ptr;
        cur_chunk_ptr += scs->chunk_lengths[i] * scs->C;
    }

    ST n_scs_elements = scs->chunk_ptrs[scs->n_chunks - 1]
                        + scs->chunk_lengths[scs->n_chunks - 1] * scs->C;

    // std::cout << "n_scs_elements = " << n_scs_elements << std::endl;
    scs->chunk_ptrs[scs->n_chunks] = n_scs_elements;

    // construct permutation vector

    scs->old_to_new_idx = V<IT, IT>(scs->n_rows);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < scs->n_rows) {
            scs->old_to_new_idx[old_row_idx] = i;
        }
    }
    

    scs->values   = V<VT, IT>(n_scs_elements);
    scs->col_idxs = V<IT, IT>(n_scs_elements);

    IT padded_col_idx = 0;

    // if(work_sharing_arr != nullptr){
    //     padded_col_idx = work_sharing_arr[my_rank];
    // }

    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->values[i]   = VT{};
        // scs->col_idxs[i] = IT{};
        scs->col_idxs[i] = padded_col_idx;
    }

    std::vector<IT> col_idx_in_row(scs->n_rows_padded);

    // fill values and col_idxs
    for (ST i = 0; i < scs->nnz; ++i) {
        IT row_old = local_mtx->I[i];

        IT row = scs->old_to_new_idx[row_old];

        ST chunk_index = row / scs->C;

        IT chunk_start = scs->chunk_ptrs[chunk_index];
        IT chunk_row   = row % scs->C;

        IT idx = chunk_start + col_idx_in_row[row] * scs->C + chunk_row;

        scs->col_idxs[idx] = local_mtx->J[i];
        scs->values[idx]   = local_mtx->values[i];

        col_idx_in_row[row]++;
    }

    // Sort inverse permutation vector, based on scs->old_to_new_idx
    std::vector<int> inv_perm(scs->n_rows);
    std::vector<int> inv_perm_temp(scs->n_rows);
    std::iota(std::begin(inv_perm_temp), std::end(inv_perm_temp), 0); // Fill with 0, 1, ..., scs->n_rows.

    generate_inv_perm<IT>(scs->old_to_new_idx.data(), &inv_perm[0],  scs->n_rows);

    scs->new_to_old_idx = inv_perm;

    scs->n_elements = n_scs_elements;

    // Experimental 2024_02_01, I do not want the rows permuted yet... so permute back
    // if sigma > C, I can see this being a problem
    // for (ST i = 0; i < scs->n_rows_padded; ++i) {
    //     IT old_row_idx = n_els_per_row[i].first;

    //     if (old_row_idx < scs->n_rows) {
    //         scs->old_to_new_idx[old_row_idx] = i;
    //     }
    // }    

    // return true;
}

template <typename VT, typename IT>
void apply_permutation(
    VT *permuted_vec,
    VT *vec_to_permute,
    IT *perm,
    int num_elems_to_permute
){
    #pragma omp parallel for
    for(int i = 0; i < num_elems_to_permute; ++i){
        permuted_vec[i] = vec_to_permute[perm[i]];
        // std::cout << "Permuting:" << vec_to_permute[i] <<  " to " << vec_to_permute[perm[i]] << std::endl;
    }
    // printf("\n");
}

// TODO: Validate
template<typename VT, typename IT>
void permute_scs_cols(
    ScsData<VT, IT> *scs,
    IT *perm
){
    ST n_scs_elements = scs->chunk_ptrs[scs->n_chunks - 1]
                    + scs->chunk_lengths[scs->n_chunks - 1] * scs->C;

    // std::vector<IT> col_idx_in_row(scs->n_rows_padded);

    V<IT, IT> col_perm_idxs(n_scs_elements);

    // TODO: parallelize

    for (ST i = 0; i < n_scs_elements; ++i) {
        if(scs->col_idxs[i] < scs->n_rows){
            // permuted version:
            col_perm_idxs[i] =  perm[scs->col_idxs[i]];
        }
        else{
            col_perm_idxs[i] = scs->col_idxs[i];
        }
    }

    // TODO (?): make col_perm_idx ptr, allocate on heap: parallelize
    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->col_idxs[i] = col_perm_idxs[i];
    }

}

/**
    @brief Matrix splitting routine, used for seperating a higher precision mtx coo struct into hp and lp sub-structs
    @param *local_mtx : Process-local coo matrix, received on this process from an earlier routine
    @param *hp_local_mtx : Process-local "higher precision" coo matrix
    @param *lp_local_mtx : Process-local "lower precision" coo matrix
*/
template <typename VT, typename IT>
void seperate_lp_from_hp( 
    Config *config,
    MtxData<VT, IT> *local_mtx, // <- should be scaled when entering this routine 
    MtxData<double, int> *hp_local_mtx, 
    MtxData<float, int> *lp_local_mtx,
    std::vector<VT> *largest_elems, // <- to adjust for jacobi scaling
    int my_rank = NULL)
{
    long double threshold = config->bucket_size;

    hp_local_mtx->is_sorted = local_mtx->is_sorted;
    hp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    hp_local_mtx->n_rows = local_mtx->n_rows;
    hp_local_mtx->n_cols = local_mtx->n_cols;

    lp_local_mtx->is_sorted = local_mtx->is_sorted;
    lp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    lp_local_mtx->n_rows = local_mtx->n_rows;
    lp_local_mtx->n_cols = local_mtx->n_cols;

    int lp_elem_ctr = 0;
    int hp_elem_ctr = 0;

    std::vector<int> hp_local_I;
    std::vector<int> hp_local_J;
    std::vector<double> hp_local_vals;
    hp_local_mtx->I = hp_local_I;
    hp_local_mtx->J = hp_local_J;
    hp_local_mtx->values = hp_local_vals;

    std::vector<int> lp_local_I;
    std::vector<int> lp_local_J;
    std::vector<float> lp_local_vals;
    lp_local_mtx->I = lp_local_I;
    lp_local_mtx->J = lp_local_J;
    lp_local_mtx->values = lp_local_vals;

    // Scan local_mtx
    // TODO: If this is a bottleneck:
    // 1. Scan in parallel 
    // 2. Allocate space
    // 3. Assign in parallel
    for(int i = 0; i < local_mtx->nnz; ++i){
        // If element value below threshold, place in hp_local_mtx
        if(config->jacobi_scale){
            if(std::abs(local_mtx->values[i]) >= std::abs((long double) threshold / (*largest_elems)[local_mtx->I[i]])){   
                hp_local_mtx->values.push_back(local_mtx->values[i]);
                hp_local_mtx->I.push_back(local_mtx->I[i]);
                hp_local_mtx->J.push_back(local_mtx->J[i]);
                ++hp_elem_ctr;
            }
            else{
                // else, place in lp_local_mtx 
                lp_local_mtx->values.push_back(local_mtx->values[i]);
                lp_local_mtx->I.push_back(local_mtx->I[i]);
                lp_local_mtx->J.push_back(local_mtx->J[i]);
                ++lp_elem_ctr;
            }
        }
        else{
            if(std::abs(local_mtx->values[i]) >= (long double) threshold){   
                hp_local_mtx->values.push_back(local_mtx->values[i]);
                hp_local_mtx->I.push_back(local_mtx->I[i]);
                hp_local_mtx->J.push_back(local_mtx->J[i]);
                ++hp_elem_ctr;
            }
            else{
                // else, place in lp_local_mtx 
                lp_local_mtx->values.push_back(local_mtx->values[i]);
                lp_local_mtx->I.push_back(local_mtx->I[i]);
                lp_local_mtx->J.push_back(local_mtx->J[i]);
                ++lp_elem_ctr;
            } 
        }

    }

    hp_local_mtx->nnz = hp_elem_ctr;
    lp_local_mtx->nnz = lp_elem_ctr;

    if(local_mtx->nnz != (hp_elem_ctr + lp_elem_ctr)){
        printf("seperate_lp_from_hp ERROR: %i Elements have been lost when seperating \
        into lp and hp structs on rank: %i.\n", local_mtx->nnz - (hp_elem_ctr + lp_elem_ctr), my_rank);
        exit(1);
    }
}

////////////////////////////// CPU Kernels ////////////////////////////// 

// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Kernel for CSR format.
 */
template <typename VT, typename IT>
static void
uspmv_omp_csr_cpu(const ST C, // 1
             const ST num_rows, // n_chunks
             const IT * RESTRICT row_ptrs, // chunk_ptrs
             const IT * RESTRICT row_lengths, // unused
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y)
{
    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for schedule(static)
    for (ST row = 0; row < num_rows; ++row) {
        VT sum{};

        #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sum)
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            sum += values[j] * x[col_idxs[j]];
        }
        y[row] = sum;
    }
}

/**
 * SpMV Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
uspmv_omp_scs_cpu(const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             VT * RESTRICT x,
             VT * RESTRICT y)
{

    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for schedule(static)
    for (ST c = 0; c < n_chunks; ++c) {
        VT tmp[C];
        for (ST i = 0; i < C; ++i) {
            tmp[i] = VT{};
        }

        IT cs = chunk_ptrs[c];

        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            for (IT i = 0; i < (IT)C; ++i) {
                tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
            }
        }

        for (ST i = 0; i < C; ++i) {
            y[c * C + i] = tmp[i];
        }
    }
}


/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename VT, typename IT>
static void
scs_impl_cpu(const ST n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{

    #pragma omp parallel for schedule(static)
    for (ST c = 0; c < n_chunks; ++c) {
        VT tmp[C]{};

        IT cs = chunk_ptrs[c];

        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            #pragma omp simd
            for (IT i = 0; i < C; ++i) {
                tmp[i] += values[cs + j * C + i] * x[col_idxs[cs + j * C + i]];
            }
        }

        #pragma omp simd
        for (IT i = 0; i < C; ++i) {
            y[c * C + i] = tmp[i];
        }
    }
}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename VT, typename IT>
static void
uspmv_omp_scs_c_cpu(
             const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{
    switch (C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64)

        #define X(CC) case CC: scs_impl_cpu<CC>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl_cpu<SCS_C>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
        break;
#endif
    default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(C));
        exit(1);
    }
}

template <typename IT>
static void
uspmv_omp_csr_ap(
    const ST hp_n_rows, // TODO: (same for both)
    const ST hp_C, // 1
    const IT * RESTRICT hp_row_ptrs, // hp_chunk_ptrs
    const IT * RESTRICT hp_row_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const double * RESTRICT hp_values,
    double * RESTRICT hp_x,
    double * RESTRICT hp_y, 
    const ST lp_n_rows, // TODO: (same for both)
    const ST lp_C, // 1
    const IT * RESTRICT lp_row_ptrs, // lp_chunk_ptrs
    const IT * RESTRICT lp_row_lengths, // unused for now
    const IT * RESTRICT lp_col_idxs,
    const float * RESTRICT lp_values,
    float * RESTRICT lp_x,
    float * RESTRICT lp_y, // unused
    int my_rank
    )
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < hp_n_rows; ++row) {
        double hp_sum{};
        #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:hp_sum)
        for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row+1]; ++j) {
            hp_sum += hp_values[j] * hp_x[hp_col_idxs[j]];
        }
        

        double lp_sum{};
        // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:lp_sum)
        #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:lp_sum)
        for (IT j = lp_row_ptrs[row]; j < lp_row_ptrs[row + 1]; ++j) {
            // lp_sum += lp_values[j] * lp_x[lp_col_idxs[j]];
            lp_sum += lp_values[j] * hp_x[lp_col_idxs[j]];

        }

        hp_y[row] = hp_sum + lp_sum; // implicit conversion to double
    }
}

#ifdef __CUDACC__
////////////////////////////// GPU Kernels //////////////////////////////
/**
 * Kernel for CSR format.
 */
template <typename VT, typename IT>
__global__ 
static void
uspmv_csr_gpu(const ST num_rows,
         const IT * RESTRICT row_ptrs,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < num_rows) {
        VT sum{};
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            sum += values[j] * x[col_idxs[j]];
        }
        y[row] = sum;
    }
}

/**
 * Kernel for ELL format, data structures use column major (CM) layout.
 */
template <typename VT, typename IT>
__global__
static void
uspmv_ell_cm_gpu(const ST num_rows,
         const ST nelems_per_row,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < num_rows) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; ++i) {
            VT val = values[row + i * num_rows];
            IT col = col_idxs[row + i * num_rows];

            sum += val * x[col];
        }
        y[row] = sum;
    }
}


/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
__global__
static void
uspmv_scs_gpu(const ST C,
         const ST n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;
    IT c   = row / C;  // the no. of the chunk
    IT idx = row % C;  // index inside the chunk

    if (row < n_chunks * C) {
        VT tmp{};
        IT cs = chunk_ptrs[c];

        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            tmp += values[cs + j * C + idx] * x[col_idxs[cs + j * C + idx]];
        }

        y[row] = tmp;
    }

}

// Advanced SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Sell-C-sigma implementation templated by C.
 */
template <ST C, typename VT, typename IT>
__device__
static void
scs_impl_gpu(const ST n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;
    ST c   = row / C;  // the no. of the chunk
    ST idx = row % C;  // index inside the chunk

    if (row < n_chunks * C) {
        VT tmp{};
        IT cs = chunk_ptrs[c];

        for (ST j = 0; j < chunk_lengths[c]; ++j) {
            tmp += values[cs + j * C + idx] * x[col_idxs[cs + j * C +idx]];
        }

        y[row] = tmp;
    }

}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename VT, typename IT>
__global__
static void
uspmv_scs_c_gpu(
             const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{
    switch (C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64) X(128)

        #define X(CC) case CC: scs_impl_gpu<CC>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl_gpu<SCS_C>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
        break;
#endif
    default:
        //fprintf(stderr,
        //        "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
        //        long(C));
        // exit(1);
    }
}
#endif

#endif /*INTERFACE_H*/