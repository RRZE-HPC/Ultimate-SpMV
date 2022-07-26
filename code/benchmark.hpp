#ifndef BENCHMARK
#define BENCHMARK

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
    ScsData<VT, IT> *local_scs,
    ContextData<IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<VT> *local_x,
    BenchmarkResult<VT, IT> *r,
    const int *my_rank,
    const int *comm_size
    )
{
    if(config->log_prof && *my_rank == 0) {log("Begin adjust_halo_col_idxs");}
    clock_t begin_ahci_time = std::clock();
    adjust_halo_col_idxs<VT, IT>(local_scs, work_sharing_arr, my_rank, comm_size);
    if(config->log_prof && *my_rank == 0) {log("Finish adjust_halo_col_idxs", begin_ahci_time, std::clock());}

    // Enter main COMM-SPMVM-SWAP loop, bench mode
    if(config->mode == 'b'){
        if(config->log_prof && *my_rank == 0) {log("Begin COMM-SPMVM-SWAP loop, bench mode");}
        clock_t begin_csslbm_time = std::clock();

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

                spmv_omp_scs<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                    local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                    local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

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
        r->perf_mflops = (double)local_scs->nnz * 2.0
                            / r->duration_kernel_s
                            / 1e6;                   // Only count usefull flops

        if(config->log_prof && *my_rank == 0) {log("Finish COMM-SPMVM-SWAP loop, bench mode", begin_csslbm_time, std::clock());}
    }
    else if(config->mode == 's'){ // Enter main COMM-SPMVM-SWAP loop, solve mode
        if(config->log_prof && *my_rank == 0) {log("Begin COMM-SPMVM-SWAP loop, solve mode");}
        clock_t begin_csslsm_time = std::clock();
        for (IT i = 0; i < config->n_repetitions; ++i)
        {
            communicate_halo_elements<VT, IT>(local_context, local_x, work_sharing_arr, my_rank, comm_size);

            spmv_omp_scs<VT, IT>(local_scs->C, local_scs->n_chunks, local_scs->chunk_ptrs.data(),
                                local_scs->chunk_lengths.data(), local_scs->col_idxs.data(),
                                local_scs->values.data(), &(*local_x)[0], &(*local_y)[0]);

            std::swap(*local_x, *local_y);
        }
        std::swap(*local_x, *local_y);

        if(config->log_prof && *my_rank == 0) {log("Finish COMM-SPMVM-SWAP loop, solve mode", begin_csslsm_time, std::clock());}
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



    r->value_type_str = type_name_from_type<VT>();
    r->index_type_str = type_name_from_type<IT>();
    r->value_type_size = sizeof(VT);
    r->index_type_size = sizeof(IT);

    // r->was_matrix_sorted = local_scs->is_sorted;
    r->was_matrix_sorted = 1;

    r->fill_in_percent = ((double)local_scs->n_elements / local_scs->nnz - 1.0) * 100.0;
    r->C               = local_scs->C;
    r->sigma           = local_scs->sigma;
}
#endif