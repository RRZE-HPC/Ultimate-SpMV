#ifndef BENCHMARK
#define BENCHMARK

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

// TODO: what do I return for this?
// NOTE: every process will return something...
/**
    @brief Collect halo element row indices for each local x-vector, and perform SPMVM
    @param *config : struct to initialze default values and user input
    @param *local_mtx : pointer to local mtx struct
    @param *work_sharing_arr : the array describing the partitioning of the rows of the total mtx struct
    @param *y_out : the vector declared to either hold the process local result, 
        or the global result if verification is selected as an option
    @param *x_out : the local x-vector, which is collect to the root process for mkl validation later
    @param *defaults : a DefaultValues struct, in which default values of x and y can be defined
    @param *x_in : optional input vector for x
*/
template <typename VT, typename IT>
void bench_spmv(
    Config *config,
    MtxData<VT, IT> *local_mtx,
    ContextData<IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<VT> *local_x,
    BenchmarkResult<VT, IT> *r,
    const int *my_rank,
    const int *comm_size
    )
{
    ScsData<VT, IT> scs;

    convert_to_scs<VT, IT>(local_mtx, config->chunk_size, config->sigma, &scs);

    // clock_t begin_ahci_time = std::clock();
    adjust_halo_col_idxs<VT, IT>(local_mtx, &scs, work_sharing_arr, my_rank, comm_size); // scs specific

    // Enter main COMM-SPMVM-SWAP loop, bench mode
    if(config->mode == 'b'){
        int local_x_size = local_x->size();
        std::vector<VT> dummy_x(local_x_size, 1);

        clock_t begin_bench_loop_time, end_bench_loop_time;
        int n_iter = 1;
        do
        {
            begin_bench_loop_time = std::clock();
            MPI_Barrier(MPI_COMM_WORLD);
            for(int k = 0; k < n_iter; ++k){
                communicate_halo_elements<VT, IT>(local_context, &dummy_x, work_sharing_arr, my_rank, comm_size);

                spmv_omp_scs<VT, IT>(scs.C, scs.n_chunks, scs.chunk_ptrs.data(),
                                    scs.chunk_lengths.data(), scs.col_idxs.data(),
                                    scs.values.data(), &(*local_x)[0], &(*local_y)[0]);

                std::swap(dummy_x, *local_y);

                if(dummy_x[local_x_size]<0.) // prevent compiler from eliminating loop
                    printf("%lf", dummy_x[local_x_size/2]);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            end_bench_loop_time = std::clock();
            n_iter *= 2;
        } while ((end_bench_loop_time - begin_bench_loop_time) / CLOCKS_PER_SEC < 1);

        n_iter /= 2;

        r->n_calls = n_iter;
        r->duration_total_s = (end_bench_loop_time - begin_bench_loop_time) / CLOCKS_PER_SEC;
        r->duration_kernel_s = r->duration_total_s/ r->n_calls;
        r->perf_mflops = (double)scs.nnz * 2.0
                            / r->duration_kernel_s
                            / 1e6;                   // Only count usefull flops
    }
    else if(config->mode == 's'){ // Enter main COMM-SPMVM-SWAP loop, solve mode
        for (IT i = 0; i < config->n_repetitions; ++i)
        {
            communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);

            spmv_omp_scs<VT, IT>(scs.C, scs.n_chunks, scs.chunk_ptrs.data(),
                                scs.chunk_lengths.data(), scs.col_idxs.data(),
                                scs.values.data(), &(*local_x)[0], &(*local_y)[0]);

            std::swap(*local_x, *local_y);
        }
        std::swap(*local_x, *local_y);
    }

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



    r->value_type_str = type_name_from_type<VT>();
    r->index_type_str = type_name_from_type<IT>();
    r->value_type_size = sizeof(VT);
    r->index_type_size = sizeof(IT);

    r->was_matrix_sorted = local_mtx->is_sorted;

    r->fill_in_percent = ((double)scs.n_elements / scs.nnz - 1.0) * 100.0;
    r->C               = scs.C;
    r->sigma           = scs.sigma;
}
#endif