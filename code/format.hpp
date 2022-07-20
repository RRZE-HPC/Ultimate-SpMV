#ifndef KERNELS
#define KERNELS

#include <mpi.h>
#include <algorithm>

#include "mpi_funcs.hpp"
#include "structs.hpp"

/**
    @brief Convert mtx struct to sell-c-sigma data structures.
    @param *mtx : data structure that was populated by the matrix market format reader mtx-reader.h 
    @param C : chunk height
    @param sigma : sorting scope
    @param *d : The ScsData struct to populate with data
*/
template <typename VT, typename IT>
void convert_to_scs(
    const MtxData<VT, IT> * mtx,
    ST C,
    ST sigma,
    ScsData<VT, IT> *d)
{
    d->nnz    = mtx->nnz;
    d->n_rows = mtx->n_rows;
    d->n_cols = mtx->n_cols;

    d->C = C;
    d->sigma = sigma;

    if (d->sigma % d->C != 0 && d->sigma != 1) {
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
    }

    if (will_add_overflow(d->n_rows, d->C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // return false;
    }
    d->n_chunks      = (mtx->n_rows + d->C - 1) / d->C;

    if (will_mult_overflow(d->n_chunks, d->C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // return false;
    }
    d->n_rows_padded = d->n_chunks * d->C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    std::vector<index_and_els_per_row> n_els_per_row(d->n_rows_padded);

    for (ST i = 0; i < d->n_rows_padded; ++i) {
        n_els_per_row[i].first = i;
    }

    for (ST i = 0; i < mtx->nnz; ++i) {
        ++n_els_per_row[mtx->I[i]].second;
    }

    // sort rows in the scope of sigma
    if (will_add_overflow(d->n_rows_padded, d->sigma)) {
        fprintf(stderr, "ERROR: no. of padded rows + sigma exceeds size type.\n");
        // return false;
    }

    for (ST i = 0; i < d->n_rows_padded; i += d->sigma) {
        auto begin = &n_els_per_row[i];
        auto end   = (i + d->sigma) < d->n_rows_padded
                        ? &n_els_per_row[i + d->sigma]
                        : &n_els_per_row[d->n_rows_padded];

        std::sort(begin, end,
                  // sort longer rows first
                  [](const auto & a, const auto & b) {
                    return a.second > b.second;
                  });
    }

    // determine chunk_ptrs and chunk_lengths

    // TODO: check chunk_ptrs can overflow
    // std::cout << d.n_chunks << std::endl;
    d->chunk_lengths = V<IT, IT>(d->n_chunks); // init a vector of length d.n_chunks
    d->chunk_ptrs    = V<IT, IT>(d->n_chunks + 1);

    IT cur_chunk_ptr = 0;
    
    for (ST i = 0; i < d->n_chunks; ++i) {
        auto begin = &n_els_per_row[i * d->C];
        auto end   = &n_els_per_row[i * d->C + d->C];

        d->chunk_lengths[i] =
                std::max_element(begin, end,
                    [](const auto & a, const auto & b) {
                        return a.second < b.second;
                    })->second;

        if (will_add_overflow(cur_chunk_ptr, d->chunk_lengths[i] * (IT)d->C)) {
            fprintf(stderr, "ERROR: chunck_ptrs exceed index type.\n");
            // return false;
        }

        d->chunk_ptrs[i] = cur_chunk_ptr;
        cur_chunk_ptr += d->chunk_lengths[i] * d->C;
    }

    

    ST n_scs_elements = d->chunk_ptrs[d->n_chunks - 1]
                        + d->chunk_lengths[d->n_chunks - 1] * d->C;
    d->chunk_ptrs[d->n_chunks] = n_scs_elements;

    // construct permutation vector

    d->old_to_new_idx = V<IT, IT>(d->n_rows);

    for (ST i = 0; i < d->n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < d->n_rows) {
            d->old_to_new_idx[old_row_idx] = i;
        }
    }
    

    d->values   = V<VT, IT>(n_scs_elements);
    d->col_idxs = V<IT, IT>(n_scs_elements);

    for (ST i = 0; i < n_scs_elements; ++i) {
        d->values[i]   = VT{};
        d->col_idxs[i] = IT{};
    }

    std::vector<IT> col_idx_in_row(d->n_rows_padded);

    // fill values and col_idxs
    for (ST i = 0; i < d->nnz; ++i) {
        IT row_old = mtx->I[i];

        IT row = d->old_to_new_idx[row_old];

        ST chunk_index = row / d->C;

        IT chunk_start = d->chunk_ptrs[chunk_index];
        IT chunk_row   = row % d->C;

        IT idx = chunk_start + col_idx_in_row[row] * d->C + chunk_row;

        d->col_idxs[idx] = mtx->J[i];
        d->values[idx]   = mtx->values[i];

        col_idx_in_row[row]++;
    }

    d->n_elements = n_scs_elements;

    // return true;
}


// TODO: what do I return for this?
// NOTE: every process will return something...
/**
    @brief Collect halo element row indices for each local x-vector, and perform SPMVM
    @param *config : struct to initialze default values and user input
    @param *local_mtx : pointer to local mtx struct
    @param *work_sharing_arr : the array describing the partitioning of the rows of the global mtx struct
    @param *y_out : the vector declared to either hold the process local result, 
        or the global result if verification is selected as an option
    @param *x_out : the local x-vector, which is collect to the root process for mkl validation later
    @param *defaults : a DefaultValues struct, in which default values of x and y can be defined
    @param *x_in : optional input vector for x
*/
template <typename VT, typename IT>
void bench_spmv_scs(
    Config *config,
    MtxData<VT, IT> *local_mtx,
    const IT *work_sharing_arr,
    std::vector<VT> *y_out,
    std::vector<VT> *x_out,
    DefaultValues<VT, IT> *defaults,
    BenchmarkResult<VT, IT> *r,
    const std::vector<VT> *x_in)
{
    // TODO: More efficient just to deduce this from worksharingArr size?
    IT comm_size;
    IT my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    ScsData<VT, IT> scs;

    convert_to_scs<VT, IT>(local_mtx, config->chunk_size, config->sigma, &scs);




    // TODO: How bad is this for scalability? Is there a way around this?
    std::vector<VT> local_y_scs_vec(scs.n_rows_padded, 0); //necessary ^^?

    // IT updated_col_idx, initial_col_idx = scs.col_idxs[0];

    IT amnt_local_elements = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

    adjust_halo_col_idxs<VT, IT>(local_mtx, &scs, &amnt_local_elements, work_sharing_arr);

    std::vector<VT> local_x(amnt_local_elements, 0);
    std::vector<VT> dummy_x(amnt_local_elements, 0);

    init_std_vec_with_ptr_or_value(local_x, local_x.size(), x_in,
                                   defaults->x, config->random_init_x);

    // Copy local_x to x_out for (optional) validation against mkl later
    for(IT i = 0; i < local_x.size(); ++i){
        (*x_out)[i] = local_x[i];
    }

    // heri := halo element row indices
    // "local_needed_heri" is all the halo elements that this process needs
    std::vector<IT> local_needed_heri;

    collect_local_needed_heri<VT, IT>(&local_needed_heri, local_mtx, work_sharing_arr);

    // "to_send_heri" are all halo elements that this process is to send
    std::vector<IT> to_send_heri;

    IT local_needed_heri_size = local_needed_heri.size();
    IT global_needed_heri_size;

    // TODO: Is this actually necessary?
    MPI_Allreduce(&local_needed_heri_size,
                  &global_needed_heri_size,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    IT *global_needed_heri = new IT[global_needed_heri_size];

    for (IT i = 0; i < global_needed_heri_size; ++i)
    {
        global_needed_heri[i] = IT{};
    }

    collect_to_send_heri<IT>(
        &to_send_heri,
        &local_needed_heri,
        global_needed_heri);

    // The shift array is used in the tag-generation scheme in halo communication.
    // the row idx is the "from_proc", the column is the "to_proc", and the element is the shift
    // after the local element index to make for the incoming halo elements
    IT *shift_arr = new IT[comm_size * comm_size];
    IT *incidence_arr = new IT[comm_size * comm_size];

    for (IT i = 0; i < comm_size * comm_size; ++i)
    {
        shift_arr[i] = IT{};
        incidence_arr[i] = IT{};
    }

    calc_heri_shifts<IT>(global_needed_heri, &global_needed_heri_size, shift_arr, incidence_arr); // NOTE: always symmetric?

    // NOTE: should always be a multiple of 3
    IT local_x_needed_padding = local_needed_heri.size() / 3;
    IT padding_for_x_and_y = std::max(local_x_needed_padding, (int)config->chunk_size);

    // Prepare buffers for communication
    dummy_x.resize(dummy_x.size() + padding_for_x_and_y);
    local_x.resize(local_x.size() + padding_for_x_and_y);
    local_y_scs_vec.resize(local_y_scs_vec.size() + padding_for_x_and_y);

    if(config->mode == 'b'){ // Enter main COMM-SPMVM-SWAP loop, bench mode
        // int NITER = 2;
        // do
        // {
        //     // get start time
        //     // TODO
        //     for(int k = 0; k < NITER; ++k){
        //         communicate_halo_elements<VT, IT>(&local_needed_heri, &to_send_heri, &local_x, shift_arr, work_sharing_arr);

        //         spmv_omp_scs<VT, IT>(scs.C, scs.n_chunks, scs.chunk_ptrs.data(),
        //                             scs.chunk_lengths.data(), scs.col_idxs.data(),
        //                             scs.values.data(), &(local_x)[0], &(local_y_scs_vec)[0]);

        //         std::swap(dummy_x, local_y_scs_vec);
        //     }

        // } while (end - start < .1)
        // NITER = NITER / 2;
    }
    else if(config->mode == 's'){ // Enter main COMM-SPMVM-SWAP loop, solve mode
        for (IT i = 0; i < config->n_repetitions; ++i)
        {
            communicate_halo_elements<VT, IT>(&local_needed_heri, &to_send_heri, &local_x, shift_arr, work_sharing_arr);

            spmv_omp_scs<VT, IT>(scs.C, scs.n_chunks, scs.chunk_ptrs.data(),
                                scs.chunk_lengths.data(), scs.col_idxs.data(),
                                scs.values.data(), &(local_x)[0], &(local_y_scs_vec)[0]);

            std::swap(local_x, local_y_scs_vec);
        }

        for (IT i = 0; i < scs.old_to_new_idx.n_rows; ++i)
        {
            (*y_out)[i] = local_x[scs.old_to_new_idx[i]];
        }
    }

    delete[] shift_arr;
    delete[] incidence_arr;
    delete[] global_needed_heri;

    double mem_matrix_b =
            (double)sizeof(VT) * scs.n_elements     // values
        + (double)sizeof(IT) * scs.n_chunks       // chunk_ptrs
        + (double)sizeof(IT) * scs.n_chunks       // chunk_lengths
        + (double)sizeof(IT) * scs.n_elements;    // col_idxs

    double mem_x_b = (double)sizeof(VT) * scs.n_cols;
    double mem_y_b = (double)sizeof(VT) * scs.n_rows_padded;
    double mem_b   = mem_matrix_b + mem_x_b + mem_y_b;

    r->mem_mb   = mem_b / 1e6;
    r->mem_m_mb = mem_matrix_b / 1e6;
    r->mem_x_mb  = mem_x_b / 1e6;
    r->mem_y_mb  = mem_y_b / 1e6;

    r->n_rows = local_mtx->n_rows;
    r->n_cols = local_mtx->n_cols;
    r->nnz    = scs.nnz;

    r->duration_kernel_s = r->duration_total_s/ r->n_calls;
    r->perf_gflops       = (double)scs.nnz * 2.0
                          / r->duration_kernel_s
                          / 1e9;                   // Only count usefull flops

    r->value_type_str = type_name_from_type<VT>();
    r->index_type_str = type_name_from_type<IT>();
    r->value_type_size = sizeof(VT);
    r->index_type_size = sizeof(IT);

    r->was_matrix_sorted = local_mtx->is_sorted;

    r->fill_in_percent = ((double)scs.n_elements / scs.nnz - 1.0) * 100.0;
    r->C               = scs.C;
    r->sigma           = scs.sigma;

    // TODO: compute code balances
}

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