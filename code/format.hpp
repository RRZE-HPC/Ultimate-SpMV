#ifndef FORMAT
#define FORMAT

#include <mpi.h>
#include <algorithm>

#include "mpi_funcs.hpp"
#include "structs.hpp"

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
    ScsData<VT, IT> *scs)
{
    scs->nnz    = local_mtx->nnz;
    scs->n_rows = local_mtx->n_rows;
    scs->n_cols = local_mtx->n_cols;

    scs->C = C;
    scs->sigma = sigma;

    if (scs->sigma % scs->C != 0 && scs->sigma != 1) {
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
    }

    if (will_add_overflow(scs->n_rows, scs->C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // return false;
    }
    scs->n_chunks      = (local_mtx->n_rows + scs->C - 1) / scs->C;

    if (will_mult_overflow(scs->n_chunks, scs->C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
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

    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->values[i]   = VT{};
        scs->col_idxs[i] = IT{};
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

    scs->n_elements = n_scs_elements;

    // return true;
}




// TODO: include functionality for other 2 storage formats
// /////////////////////////////////////////////////////////////////////////////////////////////////
// /**
//  * Create data structures for ELL format from \p mtx.
//  *
//  * \param col_major If true, column major layout for data structures will
//  *                  be used.  If false, row major layout will be used.
//  */
// template <typename VT, typename IT>
// static bool
// convert_to_ell(const MtxData<VT, IT> &mtx,
//                bool col_major,
//                ST &n_els_per_row,
//                V<IT, IT> &col_idxs,
//                V<VT, IT> &values)
// {
//     const ST max_els_per_row = calculate_max_nnz_per_row(
//                 mtx.n_rows, mtx.nnz, mtx.I.data());

//     if (n_els_per_row == -1) {
//         n_els_per_row = max_els_per_row;
//     }
//     else {
//         if (n_els_per_row < max_els_per_row) {
//             fprintf(stderr,
//                     "ERROR: ell format: number of elements per row must be >= %ld.\n",
//                     (long)max_els_per_row);
//             exit(1);
//         }
//     }

//     if (will_mult_overflow(mtx.n_rows, n_els_per_row)) {
//         fprintf(stderr, "ERROR: for ELL format no. of padded elements will exceed size type.\n");
//         return false;
//     }

//     const ST n_ell_elements = mtx.n_rows * n_els_per_row;

//     values   = V<VT, IT>(n_ell_elements);
//     col_idxs = V<IT, IT>(n_ell_elements);

//     for (ST i = 0; i < n_ell_elements; ++i) {
//         values[i]   = VT{};
//         col_idxs[i] = IT{};
//     }

//     std::vector<IT> col_idx_in_row(mtx.n_rows);

//     if (col_major) {
//         for (ST i = 0; i < mtx.nnz; ++i) {
//             IT row = mtx.I[i];
//             IT idx = col_idx_in_row[row] * mtx.n_rows + row;

//             col_idxs[idx] = mtx.J[i];
//             values[idx]   = mtx.values[i];

//             col_idx_in_row[row]++;
//         }
//     }
//     else { /* row major */
//         for (ST i = 0; i < mtx.nnz; ++i) {
//             IT row = mtx.I[i];
//             IT idx = row * max_els_per_row + col_idx_in_row[row];

//             col_idxs[idx] = mtx.J[i];
//             values[idx]   = mtx.values[i];

//             col_idx_in_row[row]++;
//         }
//     }
//     return true;
// }



// //////////////////////////////////////////////////////////////////////////////////////////////////
template <typename VT, typename IT>
static void
convert_to_csr(const MtxData<VT, IT> &mtx,
               V<IT, IT> &row_ptrs,
               V<IT, IT> &col_idxs,
               V<VT, IT> &values)
{
    values = V<VT, IT>(mtx.nnz);
    col_idxs = V<IT, IT>(mtx.nnz);
    row_ptrs = V<IT, IT>(mtx.n_rows + 1);

    std::vector<IT> col_offset_in_row(mtx.n_rows);

    convert_idxs_to_ptrs(mtx.I, row_ptrs);

    for (ST i = 0; i < mtx.nnz; ++i) {
        IT row = mtx.I[i];

        IT idx = row_ptrs[row] + col_offset_in_row[row];

        col_idxs[idx] = mtx.J[i];
        values[idx]   = mtx.values[i];

        col_offset_in_row[row]++;
    }
}
#endif