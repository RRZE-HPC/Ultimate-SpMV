#ifndef KERNELS
#define KERNELS

#include <mpi.h>
// #include <set>

#include "mpi_funcs.hpp"
#include "structs.hpp"

/**
    Convert mtx struct to sell-c-sigma data structures.
    @param *mtx : 
    @param C : 
    @param sigma : 
    @param *d : 
    @return
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
        // std::cout << i << std::endl;
    }

    // for(int i = 0; i < d.n_rows; ++i){
    //     std::cout << d.old_to_new_idx[i] << std::endl;
    // }
    // exit(0);
    

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

        // std::cout << "new row: " << row << ", old row: " << row_old << std::endl;
        // std::cout << d.C << std::endl;

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
    Description...
    @param *config : 
    @param *local_mtx : 
    @param *work_sharing_arr : 
    @param *y_out : 
    @param *x_out : 
    @param *defaults : 
    @param *x_in : 
    @return
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
    // const int c_comm_size = comm_size;

    log("allocate and place CPU matrices start\n");

    // TODO: what to do with the benchmark object?
    // BenchmarkResult r;

    ScsData<VT, IT> scs;

    log("allocate and place CPU matrices end\n");
    log("converting to scs format start\n");

    // std::cout << "Do I get here?1" << std::endl;

    // TODO: fuse with x idx adjustments potentially
    convert_to_scs<VT, IT>(local_mtx, config->chunk_size, config->sigma, &scs);

    // Print scs formatted values to verify padding at the end is done correctly
    // if(my_rank == 1){
    //     for(int i = 0; i < scs.n_elements; ++i){
    //         std::cout << scs.values[i] << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);


    log("converting to scs format end\n");

    // This y is only process local. Need an Allgather for each proc to have
    // all of the solution segments
    V<VT, IT> local_y_scs = V<VT, IT>(scs.n_rows_padded);

    std::uninitialized_fill_n(local_y_scs.data(), local_y_scs.n_rows, defaults->y);

    IT updated_col_idx, initial_col_idx = scs.col_idxs[0];

    IT amnt_local_elements = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

    adjust_halo_col_idxs<VT, IT>(local_mtx, &scs, &amnt_local_elements, work_sharing_arr);

    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);

    std::vector<VT> local_x(amnt_local_elements, 0);
    std::vector<VT> temp_vec(amnt_local_elements, 0);

    // Did I write this function?
    init_std_vec_with_ptr_or_value(local_x, local_x.size(), x_in,
                                   defaults->x, config->random_init_x);

    // init_std_vec_with_ptr_or_value(local_x, local_x.size(), x_in,
    //                                defaults->x, false);
    
    
    // Test with 2 proc
    // if(my_rank == 0){
    //     local_x[0] = 1;
    //     local_x[1] = 2;
    //     // local_x[2] = 3;
    // }
    // if(my_rank == 1){
    //     local_x[0] = 3;
    //     local_x[1] = 4;
    //     local_x[2] = 5;
    // }

    // Test with 2 procs, matrix1.mtx. hold on
    // if(my_rank == 0){
    //     local_x[0] = 1;
    //     local_x[1] = 2;
    //     local_x[2] = 3;
    //     local_x[3] = 4;
    //     local_x[4] = 5;
    //     local_x[5] = 6;
    //     local_x[6] = 7;
    //     local_x[7] = 8;
    //     local_x[8] = 9;
    //     local_x[9] = 10;
    //     local_x[10] = 11;
    //     local_x[11] = 12;
    //     local_x[12] = 13;
    //     local_x[13] = 14;
    //     local_x[14] = 15;
    //     local_x[15] = 16;
    //     local_x[16] = 17;
    //     local_x[17] = 18;
    //     local_x[18] = 19;
    //     local_x[19] = 20;
    //     local_x[20] = 21;
    //     local_x[21] = 22;
    //     local_x[22] = 23;
    // }
    // if(my_rank == 1){
    //     local_x[0] = 3;
    //     local_x[1] = 4;
    //     local_x[2] = 5;
    // }

    // Test with 3 procs
    // if(my_rank == 0){
    //     local_x[0] = 1;
    // }
    // if(my_rank == 1){
    //     local_x[0] = 2;
    // }
    // if(my_rank == 2){
    //     local_x[0] = 3;
    //     local_x[1] = 4;
    //     local_x[2] = 5;
    // }

    // Test with 4 procs
    // if(my_rank == 0){
    //     local_x[0] = 1;
    // }
    // if(my_rank == 1){
    //     local_x[0] = 2;
    // }
    // if(my_rank == 2){
    //     local_x[0] = 3;
    // }
    // if(my_rank == 3){
    //     local_x[0] = 4;
    //     local_x[1] = 5;
    // }
    // local_x[0] = 1;
    // local_x[1] = 2;
    // local_x[2] = 3;
    // local_x[3] = 4;
    // local_x[4] = 5;
    // local_x[5] = 6;
    // local_x[6] = 7;
    // local_x[7] = 8;
    // local_x[8] = 9;
    // local_x[9] = 10;


    // Copy local_x to x_out for (optional) validation against mkl later
    for(IT i = 0; i < local_x.size(); ++i){
        (*x_out)[i] = local_x[i];
    }

    // if(config->mode == "solver"){
    //     // TODO: make modifications for solver mode
    //     // the object "returned" will be the gathered results vector
    // }
    // else if(config->mode == "bench"){
    //     // TODO: make modifications for bench mode
    //     // the object returned will be the benchmark results
    // }

    // TODO: How bad is this for scalability? Is there a way around this?
    std::vector<VT> local_y_scs_vec(local_y_scs.data(), local_y_scs.data() + local_y_scs.n_rows);

    // heri := halo element row indices
    // "local_needed_heri" is all the halo elements that this process needs
    std::vector<IT> local_needed_heri;

    collect_local_needed_heri<VT, IT>(&local_needed_heri, local_mtx, work_sharing_arr);

    // int test_rank = 0;
    // if(my_rank == test_rank){
    //     for(int i = 0; i < local_needed_heri.size(); ++i){
    //         std::cout << local_needed_heri[i] << std::endl;
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);

    // "to_send_heri" are all halo elements that this process is to send
    std::vector<IT> to_send_heri;

    IT local_needed_heri_size = local_needed_heri.size();
    IT global_needed_heri_size;

    // MPI_Barrier(MPI_COMM_WORLD);
    // if(my_rank == 0){
    //     std::cout << "I'm rank: " << my_rank << " and I need: " << local_needed_heri_size/3 << " halo elements." << std::endl;
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    // if(my_rank == 1){
    //     std::cout << "I'm rank: " << my_rank << " and I need: " << local_needed_heri_size/3 << " halo elements." << std::endl;
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);


    // TODO: Is this actually necessary?
    MPI_Allreduce(&local_needed_heri_size,
                  &global_needed_heri_size,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    IT *global_needed_heri = new IT[global_needed_heri_size];
    // IT global_needed_heri[global_needed_heri_size];

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
    // IT *shift_arr = new IT[comm_size * comm_size];
    // IT *incidence_arr = new IT[comm_size * comm_size];
    IT shift_arr[comm_size * comm_size];
    IT incidence_arr[comm_size * comm_size];

    for (IT i = 0; i < comm_size * comm_size; ++i)
    {
        shift_arr[i] = IT{};
        incidence_arr[i] = IT{};
    }

    calc_heri_shifts<IT>(global_needed_heri, &global_needed_heri_size, shift_arr, incidence_arr); // NOTE: always symmetric?

    // NOTE: should always be a multiple of 3
    IT local_x_padding = local_needed_heri.size() / 3;

    // Prepare buffers for communication
    int local_x_original_size = local_x.size();
    local_x.resize(local_x.size() + local_x_padding);
    int local_x_padded_size = local_x.size();

    // std::cout << "Do I get here?2" << std::endl;
    // std::cout.precision(17);

    int show_steps = 0;
    // Enter main loop
    for (IT i = 0; i < config->n_repetitions; ++i)
    {
        int test_rank = 0;
        // if(my_rank == test_rank){
        //     for(int i = 0; i < local_needed_heri.size(); ++i){
        //         std::cout << local_needed_heri[i] << std::endl;
        //     }
        // }

        // MPI_Barrier(MPI_COMM_WORLD);
        // exit(0);
        if(show_steps){
            if(my_rank == test_rank){
                std::cout << "local_x before comm, before SPMV, before swap: " << std::endl;
                for(int i = 0; i < local_x.size(); ++i){
                    std::cout << local_x[i] << std::endl;
                }
                printf("\n");
                std::cout << "local_y before comm, before SPMV, before swap: " << std::endl;
                for(int i = 0; i < local_y_scs_vec.size(); ++i){
                    std::cout << local_y_scs_vec[i] << std::endl;
                }
                printf("\n");
            }
        }

        communicate_halo_elements<VT, IT>(&local_needed_heri, &to_send_heri, &local_x, shift_arr, work_sharing_arr);
        // printf("\n");
        MPI_Barrier(MPI_COMM_WORLD); //necessary?

        if(show_steps){
            if(my_rank == test_rank){
                std::cout << "local_x AFTER comm, before SPMV, before swap: " << std::endl;
                for(int i = 0; i < local_x.size(); ++i){
                    std::cout << local_x[i] << std::endl;
                }
                printf("\n");
                std::cout << "local_y AFTER comm, before SPMV, before swap: " << std::endl;
                for(int i = 0; i < local_y_scs_vec.size(); ++i){
                    std::cout << local_y_scs_vec[i] << std::endl;
                }
                printf("\n");
            }
        }
        // MPI_Barrier(MPI_COMM_WORLD); // temporary
        // exit(0);

        // std::cout << scs.n_elements << std::endl;
        // std::cout << scs.C << std::endl;
        // std::cout << scs.n_chunks << std::endl;
        // printf("\n");
        // // exit(0);
        // if(my_rank == test_rank){
        spmv_omp_scs<VT, IT>(scs.C, scs.n_chunks, scs.chunk_ptrs.data(),
                             scs.chunk_lengths.data(), scs.col_idxs.data(),
                             scs.values.data(), &(local_x)[0], &(local_y_scs_vec)[0]);
        // }

        // MPI_Barrier(MPI_COMM_WORLD);

        if(show_steps){
            if(my_rank == test_rank){
                std::cout << "local_x AFTER comm, AFTER SPMV, before swap: " << std::endl;
                for(int i = 0; i < local_x.size(); ++i){
                    std::cout << local_x[i] << std::endl;
                }
                printf("\n");
                std::cout << "local_y AFTER comm, AFTER SPMV, before swap: " << std::endl;
                for(int i = 0; i < local_y_scs_vec.size(); ++i){
                    std::cout << local_y_scs_vec[i] << std::endl;
                }
                printf("\n");
            }
        }
        // exit(0);

        std::swap(local_x, local_y_scs_vec);

        // MPI_Barrier(MPI_COMM_WORLD);

        if(show_steps){
            if(my_rank == test_rank){
                std::cout << "local_x AFTER comm, AFTER SPMV, AFTER swap: " << std::endl;
                for(int i = 0; i < local_x.size(); ++i){
                    std::cout << local_x[i] << std::endl;
                }
                printf("\n");
                std::cout << "local_y AFTER comm, AFTER SPMV, AFTER swap: " << std::endl;
                for(int i = 0; i < local_y_scs_vec.size(); ++i){
                    std::cout << local_y_scs_vec[i] << std::endl;
                }
                printf("\n");
            }
        }
        // TODO: I think theres a way around this...
        // if(i != config->n_repetitions - 1){
        // MPI_Barrier(MPI_COMM_WORLD);

        local_y_scs_vec.resize(local_x.size(), 0);
        local_x.resize(local_x_padded_size);

        if(show_steps){
            if(my_rank == test_rank){
                std::cout << "local_x AFTER comm, AFTER SPMV, AFTER swap, AFTER size adjust: " << std::endl;
                for(int i = 0; i < local_x.size(); ++i){
                    std::cout << local_x[i] << std::endl;
                }
                printf("\n");
                std::cout << "local_y AFTER comm, AFTER SPMV, AFTER swap, AFTER size adjust: " << std::endl;
                for(int i = 0; i < local_y_scs_vec.size(); ++i){
                    std::cout << local_y_scs_vec[i] << std::endl;
                }
                printf("\n");
            }
        }
    }

    // exit(0);

    // TODO: use a pragma parallel for?
    // Reformat proc-local result vectors. Only take the useful (non-padded) elements
    // from the scs formatted local_y_scs, and assign to local_y
    // std::vector<VT> local_y(scs.n_rows, 0);

    for (IT i = 0; i < scs.old_to_new_idx.n_rows; ++i)
    {
        (*y_out)[i] = local_x[scs.old_to_new_idx[i]];
    }

    // TODO: move to the heap
    // delete[] shift_arr;
    // delete[] incidence_arr;
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


// template <typename VT, typename IT>
// static BenchmarkResult
// bench_spmv_ell(
//                 const Config & config,
//                 const MtxData<VT, IT> & mtx,
//                 const Kernel::entry_t & k_entry,
//                 DefaultValues<VT, IT> & defaults,
//                 std::vector<VT> &x_out,
//                 std::vector<VT> &y_out,

//                 const std::vector<VT> * x_in = nullptr)
// {
//     BenchmarkResult r;

//     const ST nnz    = mtx.nnz;
//     const ST n_rows = mtx.n_rows;
//     // const ST n_cols = mtx.n_cols;

//     ST n_els_per_row = config.n_els_per_row;

//     V<VT, IT> values;
//     V<IT, IT> col_idxs;

//     log("converting to ell format start\n");

//     bool col_majro = k_entry.format == MatrixFormat::EllCm;

//     if (!convert_to_ell<VT, IT>(mtx, col_majro, n_els_per_row, col_idxs, values)) {
//         r.is_result_valid = false;
//         return r;
//     }

//     if (   n_els_per_row * n_rows != col_idxs.n_rows
//         && n_els_per_row * n_rows != values.n_rows) {
//         fprintf(stderr, "ERROR: converting matrix to ell format failed.\n");
//         r.is_result_valid = false;
//         return r;
//     }

//     log("converting to ell format end\n");

//     V<VT, IT> x_ell = V<VT, IT>(mtx.n_cols);
//     init_with_ptr_or_value(x_ell, x_ell.n_rows, x_in,
//                            defaults.x, config.random_init_x);

//     V<VT, IT> y_ell = V<VT, IT>(mtx.n_rows);
//     std::uninitialized_fill_n(y_ell.data(), y_ell.n_rows, defaults.y);

//     Kernel::fn_ell_t<VT, IT> kernel = k_entry.as_ell_kernel<VT, IT>();

//     if (k_entry.is_gpu_kernel) {
// #ifdef __NVCC__
//         log("init GPU matrices start\n");
//         VG<VT, IT> values_gpu(values);
//         VG<IT, IT> col_idxs_gpu(col_idxs);
//         VG<VT, IT> x_gpu(x_ell);
//         VG<VT, IT> y_gpu(y_ell);
//         log("init GPU matrices end\n");

//         r = spmv<VT, IT>([&]() {
//                 const int num_blocks = (n_rows + default_block_size - 1) \
//                                         / default_block_size;

//                 kernel<<<num_blocks, default_block_size>>>(n_rows,
//                        n_els_per_row, col_idxs_gpu.data(), values_gpu.data(),
//                        x_gpu.data(), y_gpu.data());
//                 },
//                 /* is_gpu_kernel */ true,
//                 config);

//         y_ell = y_gpu.copy_from_device();
// #endif
//     }
//     else {
//         r = spmv<VT, IT>([&]() {
//                 kernel(n_rows, n_els_per_row,
//                        col_idxs.data(), values.data(),
//                        x_ell.data(), y_ell.data());
//                 },
//                 /* is_gpu_kernel */ false,
//                 config);
//     }

//     if (config.verify_result) {
//         V<VT, IT> y_ref(y_ell.n_rows);
//         std::uninitialized_fill_n(y_ref.data(), y_ref.n_rows, defaults.y);

//         spmv_ell_reference(n_rows, n_els_per_row,
//                        col_idxs.data(), values.data(),
//                        x_ell.data(), y_ref.data());

//         r.is_result_valid &= spmv_verify(y_ref.data(), y_ell.data(),
//                                          y_ell.n_rows, config.verbose_verification);
//     }

//     double mem_matrix_b =
//               (double)sizeof(VT) * values.n_rows     // values
//             + (double)sizeof(IT) * col_idxs.n_rows;  // col idxs

//     double mem_x_b = (double)sizeof(VT) * x_ell.n_rows;
//     double mem_y_b = (double)sizeof(VT) * y_ell.n_rows;

//     double mem_b   = mem_matrix_b + mem_x_b + mem_y_b;

//     r.mem_mb   = mem_b / 1e6;
//     r.mem_m_mb = mem_matrix_b / 1e6;
//     r.mem_x_mb  = mem_x_b / 1e6;
//     r.mem_y_mb  = mem_y_b / 1e6;

//     r.n_rows = mtx.n_rows;
//     r.n_cols = mtx.n_cols;
//     r.nnz    = nnz;

//     r.duration_kernel_s = r.duration_total_s / r.n_calls;
//     r.perf_gflops       = (double)nnz * 2.0
//                           / r.duration_kernel_s
//                           / 1e9; // Only count usefull flops

//     r.value_type_str = type_name_from_type<VT>();
//     r.index_type_str = type_name_from_type<IT>();
//     r.value_type_size = sizeof(VT);
//     r.index_type_size = sizeof(IT);

//     r.was_matrix_sorted = mtx.is_sorted;

//     r.fill_in_percent = ((double)(n_els_per_row * n_rows) / nnz - 1.0) * 100.0;
//     r.nzr             = n_els_per_row;

//     compute_code_balances(k_entry.format, k_entry.is_gpu_kernel, false, r);

//     x_out = std::move(x_ell);
//     y_out = std::move(y_ell);

//     return r;
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


// template <typename VT, typename IT>
// static BenchmarkResult
// bench_spmv_csr(
//                 const Config & config,
//                 const MtxData<VT, IT> &mtx,

//                 const Kernel::entry_t & k_entry,

//                 DefaultValues<VT, IT> & defaults,
//                 std::vector<VT> &x_out,
//                 std::vector<VT> &y_out,

//                 const std::vector<VT> * x_in = nullptr)
// {
//     BenchmarkResult r;

//     const ST nnz    = mtx.nnz;
//     const ST n_rows = mtx.n_rows;
//     const ST n_cols = mtx.n_cols;

//     V<VT, IT> values;
//     V<IT, IT> col_idxs;
//     V<IT, IT> row_ptrs;

//     log("converting to csr format start\n");

//     convert_to_csr<VT, IT>(mtx, row_ptrs, col_idxs, values);

//     log("converting to csr format end\n");

//     V<VT, IT> x_csr = V<VT, IT>(mtx.n_cols);
//     init_with_ptr_or_value(x_csr, x_csr.n_rows, x_in,
//                            defaults.x, config.random_init_x);

//     V<VT, IT> y_csr = V<VT, IT>(mtx.n_rows);
//     std::uninitialized_fill_n(y_csr.data(), y_csr.n_rows, defaults.y);

//     Kernel::fn_csr_t<VT, IT> kernel = k_entry.as_csr_kernel<VT, IT>();

//     // print_vector("x", x_csr.data(), x_csr.data() + x_csr.n_rows);
//     // print_vector("y(pre)", y_csr.data(), y_csr.data() + y_csr.n_rows);
//     //
//     // print_vector("row_ptrs", row_ptrs.data(), row_ptrs.data() + row_ptrs.n_rows);
//     // print_vector("col_idxs", col_idxs.data(), col_idxs.data() + col_idxs.n_rows);
//     // print_vector("values",   values.data(),   values.data()   + values.n_rows);

//     if (k_entry.is_gpu_kernel) {
// #ifdef __NVCC__
//         log("init GPU matrices start\n");
//         VG<VT, IT> values_gpu(values);
//         VG<IT, IT> col_idxs_gpu(col_idxs);
//         VG<IT, IT> row_ptrs_gpu(row_ptrs);
//         VG<VT, IT> x_gpu(x_csr);
//         VG<VT, IT> y_gpu(y_csr);
//         log("init GPU matrices end\n");

//         r = spmv<VT, IT>([&]() {
//                 const int num_blocks = (n_rows + default_block_size - 1) \
//                                         / default_block_size;

//                 kernel<<<num_blocks, default_block_size>>>(n_rows,
//                        row_ptrs_gpu.data(), col_idxs_gpu.data(), values_gpu.data(),
//                        x_gpu.data(), y_gpu.data());
//             },
//             /* is_gpu_kernel */ true,
//             config);

//         y_csr = y_gpu.copy_from_device();
// #endif
//     }
//     else {
//         r = spmv<VT, IT>([&]() {
//                 kernel(n_rows,
//                        row_ptrs.data(), col_idxs.data(), values.data(),
//                        x_csr.data(), y_csr.data());
//             },
//             /* is_gpu_kernel */ false,
//             config);
//     }

//     // print_vector("y", y_csr.data(), y_csr.data() + y_csr.n_rows);

//     if (config.verify_result) {
//         V<VT, IT> y_ref(y_csr.n_rows);
//         std::uninitialized_fill_n(y_ref.data(), y_ref.n_rows, defaults.y);

//         spmv_csr_reference(n_rows, row_ptrs.data(), col_idxs.data(),
//                            values.data(), x_csr.data(), y_ref.data());

//         r.is_result_valid &= spmv_verify(y_csr.data(), y_ref.data(),
//                                          y_csr.n_rows, config.verbose_verification);
//     }

//     double mem_matrix_b =
//               (double)sizeof(VT) * nnz           // values
//             + (double)sizeof(IT) * nnz           // col idxs
//             + (double)sizeof(IT) * (n_rows + 1); // row ptrs
//     double mem_x_b = (double)sizeof(VT) * n_cols;
//     double mem_y_b = (double)sizeof(VT) * n_rows;
//     double mem_b   = mem_matrix_b + mem_x_b + mem_y_b;

//     r.mem_mb   = mem_b / 1e6;
//     r.mem_m_mb = mem_matrix_b / 1e6;
//     r.mem_x_mb  = mem_x_b / 1e6;
//     r.mem_y_mb  = mem_y_b / 1e6;

//     r.n_rows = mtx.n_rows;
//     r.n_cols = mtx.n_cols;
//     r.nnz    = nnz;

//     r.duration_kernel_s = r.duration_total_s / r.n_calls;
//     r.perf_gflops       = (double)nnz * 2.0
//                           / r.duration_kernel_s
//                           / 1e9;                 // Only count usefull flops

//     r.value_type_str = type_name_from_type<VT>();
//     r.index_type_str = type_name_from_type<IT>();
//     r.value_type_size = sizeof(VT);
//     r.index_type_size = sizeof(IT);

//     r.was_matrix_sorted = mtx.is_sorted;

//     compute_code_balances(k_entry.format, k_entry.is_gpu_kernel, false, r);

//     x_out = std::move(x_csr);
//     y_out = std::move(y_csr);

//     return r;
// }
#endif