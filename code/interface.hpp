#ifndef INTERFACE_H
#define INTERFACE_H

#include "vectors.h"
#include <numeric>
#include <limits>
#include <algorithm>

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#define RESTRICT				__restrict__

// using ST = int;

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

    void print(void)
    {
            std::cout << "n_rows = " << n_rows << std::endl;
            std::cout << "n_cols = " << n_cols << std::endl;
            std::cout << "nnz = " << nnz << std::endl;
            std::cout << "is_sorted = " << is_sorted << std::endl;
            std::cout << "is_symmetric = " << is_symmetric << std::endl;

            std::cout << "I = [";
            for(int i = 0; i < nnz; ++i){
                std::cout << I[i] << ", ";
            }
            std::cout << "]" << std::endl;

            std::cout << "J = [";
            for(int i = 0; i < nnz; ++i){
                std::cout << J[i] << ", ";
            }
            std::cout << "]" << std::endl;

            std::cout << "values = [";
            for(int i = 0; i < nnz; ++i){
                std::cout << static_cast<double>(values[i]) << ", ";
            }
            std::cout << "]" << std::endl;
    }
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

    std::vector<IT> chunk_ptrs;    // Chunk start offsets into col_idxs & values.
    std::vector<IT> chunk_lengths; // Length of one row in a chunk.
    std::vector<IT> col_idxs;
    std::vector<VT> values;
    std::vector<IT> old_to_new_idx;
    IT * new_to_old_idx;

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
        std::cout << static_cast<double>(values[i]);
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

/**
    @brief Convert mtx struct to sell-c-sigma data structures.
    @param *local_mtx : process local mtx data structure, that was populated by  
    @param C : chunk height
    @param sigma : sorting scope
    @param *scs : The ScsData struct to populate with data
*/
template <typename MT, typename VT, typename IT>
void convert_to_scs(
    const MtxData<MT, IT> *local_mtx,
    ST C,
    ST sigma,
    ScsData<VT, IT> *scs,
    int *fixed_permutation = NULL
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

    // std::vector<index_and_els_per_row> n_els_per_row(scs->n_rows_padded);
    std::vector<index_and_els_per_row> n_els_per_row(scs->n_rows_padded + sigma);

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

    if(fixed_permutation != NULL){
        std::vector<index_and_els_per_row> n_els_per_row_tmp(scs->n_rows_padded);
        for(int i = 0; i < scs->n_rows_padded; ++i){
            if(i < scs->n_rows){
                n_els_per_row_tmp[i].first = n_els_per_row[i].first;
                // n_els_per_row_tmp[i].second = n_els_per_row[fixed_permutation[i]].second;
                n_els_per_row_tmp[fixed_permutation[i]].second = n_els_per_row[i].second;
            }
            else{
                n_els_per_row_tmp[i].first = n_els_per_row[i].first;
                n_els_per_row_tmp[i].second = n_els_per_row[i].second;
            }
        }
        // n_els_per_row = n_els_per_row_tmp;
        for(int i = 0; i < scs->n_rows_padded; ++i){
            n_els_per_row[i] = n_els_per_row_tmp[i];
        }
    }
    else{
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
    }

    scs->chunk_lengths = std::vector<IT>(scs->n_chunks + scs->sigma); // init a vector of length d.n_chunks
    scs->chunk_ptrs    = std::vector<IT>(scs->n_chunks + 1 + scs->sigma);

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

    scs->chunk_ptrs[scs->n_chunks] = n_scs_elements;

    // construct permutation vector
    scs->old_to_new_idx = std::vector<IT>(scs->n_rows + scs->sigma);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < scs->n_rows) {
            scs->old_to_new_idx[old_row_idx] = i;
        }
    }
    

    // scs->values   = V<VT, IT>(n_scs_elements + scs->sigma);
    // scs->col_idxs = V<IT, IT>(n_scs_elements + scs->sigma);
    scs->values   = std::vector<VT>(n_scs_elements + scs->sigma);
    scs->col_idxs = std::vector<IT>(n_scs_elements + scs->sigma);

    printf("n_scs_elements = %i.\n\n", n_scs_elements);
    // exit(1);

    IT padded_col_idx = 0;

    // if(work_sharing_arr != nullptr){
    //     padded_col_idx = work_sharing_arr[my_rank];
    // }

    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->values[i]   = VT{};
        // scs->col_idxs[i] = IT{};
        scs->col_idxs[i] = padded_col_idx;
    }

    // I don't know what this would help, but you can try it.
    // std::vector<IT> col_idx_in_row(scs->n_rows_padded);
    std::vector<IT> col_idx_in_row(scs->n_rows_padded + scs->sigma);
    // int *col_idx_in_row = new int [scs->n_rows_padded + scs->sigma];
    // for (int i = 0; i < scs->n_rows_padded + scs->sigma; ++i){
    //     col_idx_in_row[i] = 0;
    // }

    // fill values and col_idxs
    for (ST i = 0; i < scs->nnz; ++i) {
        IT row_old = local_mtx->I[i];
        IT row;

        if(fixed_permutation != NULL){
            row = fixed_permutation[row_old];
        }
        else{
            row = scs->old_to_new_idx[row_old];
        }

        ST chunk_index = row / scs->C;

        IT chunk_start = scs->chunk_ptrs[chunk_index];
        IT chunk_row   = row % scs->C;

        IT idx = chunk_start + col_idx_in_row[row] * scs->C + chunk_row;

        scs->col_idxs[idx] = local_mtx->J[i];
        scs->values[idx]   = local_mtx->values[i];

        col_idx_in_row[row]++;
    }

    // for (int i = 0; i < scs->n_rows_padded + scs->sigma; ++i){
    //     printf("col idx = %i\n", scs->col_idxs[i]);
    // }
    // exit(0);

    // printf("Problem row 16\n");
    // for (int i = 0; i < n_els_per_row[16].second; ++i){
    //     IT row = 16;
    //     ST chunk_index = row / scs->C;
    //     IT chunk_start = scs->chunk_ptrs[chunk_index];
    //     IT chunk_row   = row % scs->C;
    //     IT idx = chunk_start + col_idx_in_row[row] * scs->C + chunk_row;
    //     printf("val: %f, col: %i\n", scs->values[idx], scs->col_idxs[idx]);
    // }

    // Sort inverse permutation vector, based on scs->old_to_new_idx
    // std::vector<int> inv_perm(scs->n_rows);
    // std::vector<int> inv_perm_temp(scs->n_rows);
    // std::iota(std::begin(inv_perm_temp), std::end(inv_perm_temp), 0); // Fill with 0, 1, ..., scs->n_rows.
    // generate_inv_perm<IT>(scs->old_to_new_idx.data(), &(inv_perm)[0],  scs->n_rows);


    int *inv_perm = new int[scs->n_rows];
    int *inv_perm_temp = new int[scs->n_rows];
    for(int i = 0; i < scs->n_rows; ++i){
        inv_perm_temp[i] = i;
    }
    generate_inv_perm<IT>(scs->old_to_new_idx.data(), inv_perm,  scs->n_rows);

    scs->new_to_old_idx = inv_perm;

    scs->n_elements = n_scs_elements;

    // for(int i = 0; i < n_scs_elements; ++i){
    //     printf("col idx: %i\n", scs->col_idxs.data()[i]);
    // }

    // Experimental 2024_02_01, I do not want the rows permuted yet... so permute back
    // if sigma > C, I can see this being a problem
    // for (ST i = 0; i < scs->n_rows_padded; ++i) {
    //     IT old_row_idx = n_els_per_row[i].first;

    //     if (old_row_idx < scs->n_rows) {
    //         scs->old_to_new_idx[old_row_idx] = i;
    //     }
    // }    

    // return true;


    // printf("perm = [\n");
    // for (ST i = 0; i < scs->n_rows; ++i) {
    //     printf("perm idx %i\n", scs->old_to_new_idx.data()[i]);
    // }
    // printf("]\n");

    // printf("inv perm = [\n");
    // for (ST i = 0; i < scs->n_rows; ++i) {
    //     printf("inv perm idx %i\n", scs->new_to_old_idx[i]);
    // }
    // printf("]\n");
    // exit(1);

    // delete col_idx_in_row; <- uhh why does this cause a seg fault?
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

    std::vector<IT, IT> col_perm_idxs(n_scs_elements);

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

template <typename VT, typename IT>
void partition_precisions( 
    MtxData<VT, IT> *local_mtx, // <- should be scaled when entering this routine 
    MtxData<double, int> *dp_local_mtx, 
    MtxData<float, int> *sp_local_mtx,
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *hp_local_mtx,
#endif
    std::vector<VT> *largest_row_elems, 
    std::vector<VT> *largest_col_elems,
    double ap_threshold_1,
    double ap_threshold_2,
    char *ap_value_type,
    bool is_equilibrated
)
{
    dp_local_mtx->is_sorted = local_mtx->is_sorted;
    dp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    dp_local_mtx->n_rows = local_mtx->n_rows;
    dp_local_mtx->n_cols = local_mtx->n_cols;

    sp_local_mtx->is_sorted = local_mtx->is_sorted;
    sp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    sp_local_mtx->n_rows = local_mtx->n_rows;
    sp_local_mtx->n_cols = local_mtx->n_cols;

#ifdef HAVE_HALF_MATH
    hp_local_mtx->is_sorted = local_mtx->is_sorted;
    hp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    hp_local_mtx->n_rows = local_mtx->n_rows;
    hp_local_mtx->n_cols = local_mtx->n_cols;
#endif

    int dp_elem_ctr = 0;
    int sp_elem_ctr = 0;
    int hp_elem_ctr = 0;

    // TODO: This practice of assigning pointers to vectors is dangerous...
    std::vector<IT> dp_local_I;
    std::vector<IT> dp_local_J;
    std::vector<double> dp_local_vals;
    dp_local_mtx->I = dp_local_I;
    dp_local_mtx->J = dp_local_J;
    dp_local_mtx->values = dp_local_vals;

    std::vector<IT> sp_local_I;
    std::vector<IT> sp_local_J;
    std::vector<float> sp_local_vals;
    sp_local_mtx->I = sp_local_I;
    sp_local_mtx->J = sp_local_J;
    sp_local_mtx->values = sp_local_vals;

#ifdef HAVE_HALF_MATH
    std::vector<IT> hp_local_I;
    std::vector<IT> hp_local_J;
    std::vector<_Float16> hp_local_vals;
    hp_local_mtx->I = hp_local_I;
    hp_local_mtx->J = hp_local_J;
    hp_local_mtx->values = hp_local_vals;
#endif

    // Scan local_mtx
    // TODO: If this is a bottleneck:
    // 1. Scan in parallel 
    // 2. Allocate space
    // 3. Assign in parallel
    if(ap_value_type == "ap[dp_sp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(is_equilibrated){
                // TODO: static casting just to make it compile... 
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1 / \
                    ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) {   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else{
                    // else, place in sp_local_mtx 
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
            }
            else{
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1){   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if (std::abs(static_cast<double>(local_mtx->values[i])) < ap_threshold_1){
                    // else, place in sp_local_mtx 
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else{
                    printf("partition_precisions ERROR: Element %i does not fit into either struct.\n", i);
                    exit(1);
                }
            }
        }

        dp_local_mtx->nnz = dp_elem_ctr;
        sp_local_mtx->nnz = sp_elem_ctr;

        if(local_mtx->nnz != (dp_elem_ctr + sp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into dp and sp structs.\n", local_mtx->nnz - (dp_elem_ctr + sp_elem_ctr));
            exit(1);
        }
    }
#ifdef HAVE_HALF_MATH
    else if(ap_value_type == "ap[dp_hp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(is_equilibrated){
                // TODO: static casting just to make it compile... 
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1 / ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) {   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else{
                    hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
            else{
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1){   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if (std::abs(static_cast<double>(local_mtx->values[i])) < ap_threshold_1){
                    hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
                else{
                    printf("partition_precisions ERROR: Element %i does not fit into either struct.\n", i);
                    exit(1);
                }
            }
        }

        dp_local_mtx->nnz = dp_elem_ctr;
        hp_local_mtx->nnz = hp_elem_ctr;


        if(local_mtx->nnz != (dp_elem_ctr + hp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into dp and hp structs: %i.\n", local_mtx->nnz - (dp_elem_ctr + hp_elem_ctr));
            exit(1);
        }
    }
    else if(ap_value_type == "ap[sp_hp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(is_equilibrated){
                // TODO: static casting just to make it compile... 
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1 / \
                    ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) {   
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else{
                    // else, place in sp_local_mtx 
                    hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
            else{
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1){   
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else if (std::abs(static_cast<double>(local_mtx->values[i])) < ap_threshold_1){
                    // else, place in sp_local_mtx 
                    hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
                else{
                    printf("partition_precisions ERROR: Element %i does not fit into either struct.\n", i);
                    exit(1);
                }
            }
        }

        sp_local_mtx->nnz = sp_elem_ctr;
        hp_local_mtx->nnz = hp_elem_ctr;

        if(local_mtx->nnz != (sp_elem_ctr + hp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into sp and hp structs: %i.\n", local_mtx->nnz - (sp_elem_ctr + hp_elem_ctr));
            exit(1);
        }
    }
    else if(ap_value_type == "ap[dp_sp_hp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(is_equilibrated){
                // Element is larger than the largest threshold
                if(
                    (std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1 / \
                    (static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]])))
                    ) {   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if(
                    // Element is between thresholds
                    (std::abs(static_cast<double>(local_mtx->values[i])) <= ap_threshold_1 / ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) &&
                    (std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_2 / ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]])))
                ){
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else
                {
                    // else, element is between 0 and lowest threshold
                    hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
            else{
                // Element is larger than the largest threshold
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_1){
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if(
                    // Element is between thresholds
                    (std::abs(static_cast<double>(local_mtx->values[i])) <= ap_threshold_1) &&
                    (std::abs(static_cast<double>(local_mtx->values[i])) >= ap_threshold_2)
                ){
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else
                {
                    // else, element is between 0 and lowest threshold
                    hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
        }

        dp_local_mtx->nnz = dp_elem_ctr;
        sp_local_mtx->nnz = sp_elem_ctr;
        hp_local_mtx->nnz = hp_elem_ctr;

        if(local_mtx->nnz != (dp_elem_ctr + sp_elem_ctr + hp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into dp, sp, and hp structs.\n", local_mtx->nnz - (dp_elem_ctr + sp_elem_ctr + hp_elem_ctr));
            exit(1);
        }
    }
#endif
}

////////////////////////////// CPU Kernels ////////////////////////////// 

// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Kernel for CSR format.
 */
template <typename FT, typename VT, typename IT>
static void
uspmv_csr_cpu(const ST C, // 1
             const ST num_rows, // n_chunks
             const IT * RESTRICT row_ptrs, // chunk_ptrs
             const IT * RESTRICT row_lengths, // unused
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             FT * RESTRICT x,
             FT * RESTRICT y)
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("uspmv_crs_benchmark");
#endif
        #pragma omp for schedule(static)
        for (ST row = 0; row < num_rows; ++row) {
            VT sum{};
            // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sum)
            #pragma omp simd reduction(+:sum)
            for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
                sum += values[j] * x[col_idxs[j]];
            }
            y[row] = sum;
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("uspmv_crs_benchmark");
#endif
    }
}

/**
 * SpMV Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename FT, typename IT>
static void
uspmv_scs_cpu(const IT C,
             const IT n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             FT * RESTRICT x,
             FT * RESTRICT y)
{
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < n_chunks; ++c) {
        VT tmp[C];
        for (ST i = 0; i < C; ++i) {
            tmp[i] = VT{};
        }

        IT cs = chunk_ptrs[c];

        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            for (IT i = 0; i < C; ++i) {
                tmp[i] += values[cs + j * C + i] * x[col_idxs[cs + j * C + i]];
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
uspmv_scs_c_cpu(
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
uspmv_csr_apdpsp_cpu(
    const ST * dp_C,
    const ST * dp_n_rows,
    const IT * RESTRICT dp_row_ptrs,
    const IT * RESTRICT dp_row_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_rows,
    const IT * RESTRICT sp_row_ptrs,
    const IT * RESTRICT sp_row_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
#ifdef HAVE_HALF_MATH
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_rows,
    const IT * RESTRICT hp_row_ptrs,
    const IT * RESTRICT hp_row_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
#else
    float * RESTRICT sp_y
#endif
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apdpsp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *dp_n_rows; ++row) {
                double dp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:dp_sum)
                for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                    dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
    #endif
                }

                double sp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                #pragma omp simd reduction(+:sp_sum)
                for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row + 1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    sp_sum += sp_values[j] * dp_x[sp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], dp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], dp_x[sp_col_idxs[j]]);
    #endif
                }

                dp_y[row] = dp_sum + sp_sum; // implicit conversion to VTU?
            }

    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apdpsp_crs_benchmark");
    #endif
        }
}

#ifdef HAVE_HALF_MATH
template <typename IT>
static void
uspmv_csr_apdphp_cpu(
    const ST * dp_C,
    const ST * dp_n_rows,
    const IT * RESTRICT dp_row_ptrs,
    const IT * RESTRICT dp_row_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_rows,
    const IT * RESTRICT sp_row_ptrs,
    const IT * RESTRICT sp_row_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_rows,
    const IT * RESTRICT hp_row_ptrs,
    const IT * RESTRICT hp_row_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apdphp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *dp_n_rows; ++row) {
                double dp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:dp_sum)
                for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                    dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
    #endif
                }

                double hp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                #pragma omp simd reduction(+:hp_sum)
                for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row + 1]; ++j) {
                    // hp_sum += hp_values[j] * hp_x[hp_col_idxs[j]];
                    hp_sum += hp_values[j] * dp_x[hp_col_idxs[j]]; // Conversion how???

    #ifdef DEBUG_MODE_FINE
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);
    #endif
                }

                dp_y[row] = dp_sum + hp_sum;
            }

    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apdphp_crs_benchmark");
    #endif
        }
}
#endif

#ifdef HAVE_HALF_MATH
template <typename IT>
static void
uspmv_csr_apsphp_cpu(
    const ST * dp_C,
    const ST * dp_n_rows,
    const IT * RESTRICT dp_row_ptrs,
    const IT * RESTRICT dp_row_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_rows,
    const IT * RESTRICT sp_row_ptrs,
    const IT * RESTRICT sp_row_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_rows,
    const IT * RESTRICT hp_row_ptrs,
    const IT * RESTRICT hp_row_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apsphp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *sp_n_rows; ++row) {
                float sp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:sp_sum)
                for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row+1]; ++j) {
                    sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                    printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
    #endif
                }

                float hp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                #pragma omp simd reduction(+:hp_sum)
                for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row + 1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    hp_sum += hp_values[j] * sp_x[hp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                    printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], sp_x[sp_col_idxs[j]]);
                    printf("sp_sum += %5.16f * %f\n", sp_values[j], sp_x[sp_col_idxs[j]]);
    #endif
                }

                sp_y[row] = sp_sum + hp_sum; // implicit conversion to VTU?
            }
    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apsphp_crs_benchmark");
    #endif
        }
}
#endif

#ifdef HAVE_HALF_MATH
template <typename IT>
static void
uspmv_csr_apdpsphp_cpu(
    const ST * dp_C,
    const ST * dp_n_rows,
    const IT * RESTRICT dp_row_ptrs,
    const IT * RESTRICT dp_row_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_rows,
    const IT * RESTRICT sp_row_ptrs,
    const IT * RESTRICT sp_row_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_rows,
    const IT * RESTRICT hp_row_ptrs,
    const IT * RESTRICT hp_row_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
    )
{
    #pragma omp parallel
    {
    #ifdef USE_LIKWID
            LIKWID_MARKER_START("spmv_apdpsphp_crs_benchmark");
    #endif
            #pragma omp for schedule(static)
            for (ST row = 0; row < *sp_n_rows; ++row) {

                double dp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:dp_sum)
                for (IT j = dp_row_ptrs[row]; j < dp_row_ptrs[row+1]; ++j) {
                    dp_sum += dp_values[j] * dp_x[dp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                    // printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    // printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
    #endif
                }

                double sp_sum{};
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:dp_sum)
                #pragma omp simd reduction(+:sp_sum)
                for (IT j = sp_row_ptrs[row]; j < sp_row_ptrs[row+1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    sp_sum += sp_values[j] * dp_x[sp_col_idxs[j]];
    #ifdef DEBUG_MODE_FINE
                    // printf("j = %i, dp_col_idxs[j] = %i, dp_x[dp_col_idxs[j]] = %f\n", j, dp_col_idxs[j], dp_x[dp_col_idxs[j]]);
                    // printf("dp_sum += %f * %f\n", dp_values[j], dp_x[dp_col_idxs[j]]);
    #endif
                }

                double hp_sum{};
                // #pragma omp simd simdlen(2*VECTOR_LENGTH) reduction(+:sp_sum)
                // #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:sp_sum)
                #pragma omp simd reduction(+:hp_sum)
                for (IT j = hp_row_ptrs[row]; j < hp_row_ptrs[row + 1]; ++j) {
                    // sp_sum += sp_values[j] * sp_x[sp_col_idxs[j]];
                    hp_sum += hp_values[j] * dp_x[hp_col_idxs[j]];

    #ifdef DEBUG_MODE_FINE
                    // printf("j = %i, sp_col_idxs[j] = %i, sp_x[sp_col_idxs[j]] = %f\n", j, sp_col_idxs[j], dp_x[sp_col_idxs[j]]);
                    // printf("sp_sum += %5.16f * %f\n", sp_values[j], dp_x[sp_col_idxs[j]]);
    #endif
                }

                dp_y[row] = dp_sum + sp_sum + hp_sum;
            }

    #ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv_apdpsphp_crs_benchmark");
    #endif
        }
}
#endif

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
uspmv_scs_apdpsp_cpu(
    const ST * dp_C,
    const ST * dp_n_chunks,
    const IT * RESTRICT dp_chunk_ptrs,
    const IT * RESTRICT dp_chunk_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_chunks,
    const IT * RESTRICT sp_chunk_ptrs,
    const IT * RESTRICT sp_chunk_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
#ifdef HAVE_HALF_MATH
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_chunks,
    const IT * RESTRICT hp_chunk_ptrs,
    const IT * RESTRICT hp_chunk_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
#else
    float * RESTRICT sp_y
#endif
)
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("uspmv_apdpsp_scs_benchmark");
#endif
        #pragma omp for
        for (ST c = 0; c < *dp_n_chunks; ++c) {
            double dp_tmp[*dp_C];
            double sp_tmp[*dp_C];

            for (ST i = 0; i < *dp_C; ++i) {
                dp_tmp[i] = 0.0;
            }
            for (ST i = 0; i < *dp_C; ++i) {
                sp_tmp[i] = 0.0;
            }

            IT dp_cs = dp_chunk_ptrs[c];
            IT sp_cs = sp_chunk_ptrs[c];

            for (IT j = 0; j < dp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    dp_tmp[i] += dp_values[dp_cs + j * *dp_C + i] * dp_x[dp_col_idxs[dp_cs + j * *dp_C + i]];
                }
            }
            for (IT j = 0; j < sp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    sp_tmp[i] += sp_values[sp_cs + j * *dp_C + i] * dp_x[sp_col_idxs[sp_cs + j * *dp_C + i]];
                }
            }

            for (IT i = 0; i < *dp_C; ++i) {
                dp_y[c * *dp_C + i] = dp_tmp[i] + sp_tmp[i];
            }
#ifdef USE_LIKWID
            LIKWID_MARKER_STOP("uspmv_apdpsp_scs_benchmark");
#endif
        }
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
#ifdef HAVE_HALF_MATH
template <typename VT, typename IT>
static void
uspmv_scs_apdphp_cpu(
    const ST * dp_C,
    const ST * dp_n_chunks,
    const IT * RESTRICT dp_chunk_ptrs,
    const IT * RESTRICT dp_chunk_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_chunks,
    const IT * RESTRICT sp_chunk_ptrs,
    const IT * RESTRICT sp_chunk_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_chunks,
    const IT * RESTRICT hp_chunk_ptrs,
    const IT * RESTRICT hp_chunk_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
)
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("uspmv_apdphp_scs_benchmark");
#endif
        #pragma omp for
        for (ST c = 0; c < *dp_n_chunks; ++c) {
            double dp_tmp[*dp_C];
            double hp_tmp[*dp_C];

            for (ST i = 0; i < *dp_C; ++i) {
                dp_tmp[i] = 0.0;
                hp_tmp[i] = 0.0;
            }

            IT dp_cs = dp_chunk_ptrs[c];
            IT hp_cs = hp_chunk_ptrs[c];

            for (IT j = 0; j < dp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    dp_tmp[i] += dp_values[dp_cs + j * *dp_C + i] * dp_x[dp_col_idxs[dp_cs + j * *dp_C + i]];
                }
            }
            for (IT j = 0; j < hp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    hp_tmp[i] += hp_values[hp_cs + j * *dp_C + i] * dp_x[hp_col_idxs[hp_cs + j * *dp_C + i]];
                }
            }

            for (IT i = 0; i < *dp_C; ++i) {
                dp_y[c * *dp_C + i] = dp_tmp[i] + hp_tmp[i];
            }
#ifdef USE_LIKWID
            LIKWID_MARKER_STOP("uspmv_apdphp_scs_benchmark");
#endif
        }
    }
}
#endif

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
#ifdef HAVE_HALF_MATH
template <typename VT, typename IT>
static void
uspmv_scs_apsphp_cpu(
    const ST * dp_C,
    const ST * dp_n_chunks,
    const IT * RESTRICT dp_chunk_ptrs,
    const IT * RESTRICT dp_chunk_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_chunks,
    const IT * RESTRICT sp_chunk_ptrs,
    const IT * RESTRICT sp_chunk_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_chunks,
    const IT * RESTRICT hp_chunk_ptrs,
    const IT * RESTRICT hp_chunk_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
)
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("uspmv_apsphp_scs_benchmark");
#endif
        #pragma omp for
        for (ST c = 0; c < *sp_n_chunks; ++c) {
            double sp_tmp[*sp_C];
            double hp_tmp[*sp_C];

            for (ST i = 0; i < *sp_C; ++i) {
                sp_tmp[i] = 0.0f;
                hp_tmp[i] = 0.0;
            }

            IT sp_cs = sp_chunk_ptrs[c];
            IT hp_cs = hp_chunk_ptrs[c];

            for (IT j = 0; j < sp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *sp_C; ++i) {
                    sp_tmp[i] += sp_values[sp_cs + j * *sp_C + i] * sp_x[sp_col_idxs[sp_cs + j * *sp_C + i]];
                }
            }
            for (IT j = 0; j < hp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *sp_C; ++i) {
                    hp_tmp[i] += hp_values[hp_cs + j * *sp_C + i] * sp_x[hp_col_idxs[hp_cs + j * *sp_C + i]];
                }
            }

            for (IT i = 0; i < *sp_C; ++i) {
                sp_y[c * *sp_C + i] = sp_tmp[i] + hp_tmp[i];
            }
#ifdef USE_LIKWID
            LIKWID_MARKER_STOP("uspmv_apsphp_scs_benchmark");
#endif
        }
    }
}
#endif

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
#ifdef HAVE_HALF_MATH
template <typename VT, typename IT>
static void
uspmv_scs_apdpsphp_cpu(
    const ST * dp_C,
    const ST * dp_n_chunks,
    const IT * RESTRICT dp_chunk_ptrs,
    const IT * RESTRICT dp_chunk_lengths,
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y, 
    const ST * sp_C,
    const ST * sp_n_chunks,
    const IT * RESTRICT sp_chunk_ptrs,
    const IT * RESTRICT sp_chunk_lengths,
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
    const ST * hp_C,
    const ST * hp_n_chunks,
    const IT * RESTRICT hp_chunk_ptrs,
    const IT * RESTRICT hp_chunk_lengths,
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y
)
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("uspmv_apdpsphp_scs_benchmark");
#endif
        #pragma omp for
        for (ST c = 0; c < *dp_n_chunks; ++c) {
            double dp_tmp[*dp_C];
            double sp_tmp[*dp_C];
            double hp_tmp[*dp_C];

            // Just fuse probably
            for (ST i = 0; i < *dp_C; ++i) {
                dp_tmp[i] = 0.0;
                sp_tmp[i] = 0.0;
                hp_tmp[i] = 0.0;
            }

            IT dp_cs = dp_chunk_ptrs[c];
            IT sp_cs = sp_chunk_ptrs[c];
            IT hp_cs = hp_chunk_ptrs[c];

            for (IT j = 0; j < dp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    dp_tmp[i] += dp_values[dp_cs + j * *dp_C + i] * dp_x[dp_col_idxs[dp_cs + j * *dp_C + i]];
                }
            }
            for (IT j = 0; j < sp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *dp_C; ++i) {
                    sp_tmp[i] += sp_values[sp_cs + j * *dp_C + i] * dp_x[sp_col_idxs[sp_cs + j * *dp_C + i]];
                }
            }
            for (IT j = 0; j < hp_chunk_lengths[c]; ++j) {
                for (IT i = 0; i < *sp_C; ++i) {
                    hp_tmp[i] += hp_values[hp_cs + j * *dp_C + i] * dp_x[hp_col_idxs[hp_cs + j * *dp_C + i]];
                }
            }

            for (IT i = 0; i < *dp_C; ++i) {
                dp_y[c * *dp_C + i] = dp_tmp[i] + sp_tmp[i] + hp_tmp[i];
            }
#ifdef USE_LIKWID
            LIKWID_MARKER_STOP("uspmv_apdpsphp_scs_benchmark");
#endif
        }
    }
}
#endif

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


template <typename VT, typename IT>
void execute_uspmv(
    const ST * C, // 1
    const ST * n_chunks, // TODO: (same for both)
    const IT * RESTRICT chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT chunk_lengths, // unused for now
    const IT * RESTRICT col_idxs,
    const VT * RESTRICT values,
    VT * RESTRICT x,
    VT * RESTRICT y,
#ifdef USE_AP
    const ST * dp_C, // 1
    const ST * dp_n_chunks, // TODO: (same for both)
    const IT * RESTRICT dp_chunk_ptrs, // dp_chunk_ptrs
    const IT * RESTRICT dp_chunk_lengths, // unused for now
    const IT * RESTRICT dp_col_idxs,
    const double * RESTRICT dp_values,
    double * RESTRICT dp_x,
    double * RESTRICT dp_y,
    const ST * sp_C, // 1
    const ST * sp_n_chunks, // TODO: (same for both)
    const IT * RESTRICT sp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT sp_chunk_lengths, // unused for now
    const IT * RESTRICT sp_col_idxs,
    const float * RESTRICT sp_values,
    float * RESTRICT sp_x,
    float * RESTRICT sp_y,
#ifdef HAVE_HALF_MATH
    const ST * hp_C, // 1
    const ST * hp_n_chunks, // TODO: (same for both)
    const IT * RESTRICT hp_chunk_ptrs, // sp_chunk_ptrs
    const IT * RESTRICT hp_chunk_lengths, // unused for now
    const IT * RESTRICT hp_col_idxs,
    const _Float16 * RESTRICT hp_values,
    _Float16 * RESTRICT hp_x,
    _Float16 * RESTRICT hp_y,
#endif
#endif
    char *ap_value_type
){
    if(CHUNK_SIZE > 1 || SIGMA > 1){
        // Use SELL-C-sigma kernels
#ifdef USE_AP
        if(ap_value_type == "ap[dp_sp]"){
            uspmv_scs_apdpsp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_x,
                dp_y,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                nullptr,
#ifdef HAVE_HALF_MATH
                nullptr,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
#else
                nullptr
#endif
            );
        }
        else if(ap_value_type == "ap[dp_hp]"){
#ifdef HAVE_HALF_MATH
            uspmv_scs_apdphp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_x,
                dp_y,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                nullptr,
                nullptr,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
            );
#endif
        }
        else if(ap_value_type == "ap[sp_hp]"){
#ifdef HAVE_HALF_MATH
            uspmv_scs_apsphp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                nullptr,
                nullptr,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                sp_x,
                sp_y,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
            );
#endif
        }
        else if(ap_value_type == "ap[dp_sp_hp]"){
#ifdef HAVE_HALF_MATH
            uspmv_scs_apdpsphp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_x,
                dp_y,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                nullptr,
                nullptr,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
            );
#endif
        }
#else
        uspmv_scs_cpu<VT, VT, int>(
            *C,
            *n_chunks,
            chunk_ptrs,
            chunk_lengths,
            col_idxs,
            values,
            x,
            y
        );
#endif
    }
    else{
        // Else, default to CRS kernels
#ifdef USE_AP
        if(ap_value_type == "ap[dp_sp]"){
            uspmv_csr_apdpsp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_x,
                dp_y,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                nullptr,
#ifdef HAVE_HALF_MATH
                nullptr,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
#else
                nullptr
#endif
            );
        }
        else if(ap_value_type == "ap[dp_hp]"){
#ifdef HAVE_HALF_MATH
            uspmv_csr_apdphp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_x,
                dp_y,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                nullptr,
                nullptr,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
            );
#endif
        }
        else if(ap_value_type == "ap[sp_hp]"){
#ifdef HAVE_HALF_MATH
            uspmv_csr_apsphp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                nullptr,
                nullptr,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                sp_x,
                sp_y,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
            );
#endif
        }
        else if(ap_value_type == "ap[dp_sp_hp]"){
#ifdef HAVE_HALF_MATH
            uspmv_csr_apdpsphp_cpu<int>(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_x,
                dp_y,
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                nullptr,
                nullptr,
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                nullptr,
                nullptr
            );
#endif
        }
#else
        uspmv_csr_cpu<VT, VT, int>(
            *C,
            *n_chunks,
            chunk_ptrs,
            chunk_lengths,
            col_idxs,
            values,
            x,
            y
        );
#endif
    }
}

#endif /*INTERFACE_H*/