#ifndef KERNELS
#define KERNELS



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
};

template <ST C, typename VT, typename IT>
static void
scs_impl(const ST n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    // int comm_size;
    // int my_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    // if(my_rank == 2){
    //     for(int i = 0; i < x.size(); ++i){
    //         std::cout << x[i] << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD); // temporary
    // exit(0);

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
spmv_omp_scs_c(
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

        #define X(CC) case CC: scs_impl<CC>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl<SCS_C>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
        break;
#endif
    default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(C));
        exit(1);
    }
}

/**
 * Convert \p mtx to sell-c-sigma data structures.
 *
 * If \p C is < 1 then SCS_DEFAULT_C is used.
 * If \p sigma is < 1 then 1 is used.
 *
 * Note: the matrix entries in \p mtx don't need to be sorted.
 */
template <typename VT, typename IT>
static bool
convert_to_scs(const MtxData<VT, IT> & mtx,
               ST C, ST sigma,
               ScsData<VT, IT> & d)
{
    d.nnz    = mtx.nnz;
    d.n_rows = mtx.n_rows;
    d.n_cols = mtx.n_cols;

    d.C = C;
    d.sigma = sigma;

    if (d.sigma % d.C != 0 && d.sigma != 1) {
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
    }

    if (will_add_overflow(d.n_rows, d.C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        return false;
    }
    d.n_chunks      = (mtx.n_rows + d.C - 1) / d.C;

    if (will_mult_overflow(d.n_chunks, d.C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        return false;
    }
    d.n_rows_padded = d.n_chunks * d.C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    std::vector<index_and_els_per_row> n_els_per_row(d.n_rows_padded);

    for (ST i = 0; i < d.n_rows_padded; ++i) {
        n_els_per_row[i].first = i;
    }

    for (ST i = 0; i < mtx.nnz; ++i) {
        ++n_els_per_row[mtx.I[i]].second;
    }

    // sort rows in the scope of sigma
    if (will_add_overflow(d.n_rows_padded, d.sigma)) {
        fprintf(stderr, "ERROR: no. of padded rows + sigma exceeds size type.\n");
        return false;
    }

    for (ST i = 0; i < d.n_rows_padded; i += d.sigma) {
        auto begin = &n_els_per_row[i];
        auto end   = (i + d.sigma) < d.n_rows_padded
                        ? &n_els_per_row[i + d.sigma]
                        : &n_els_per_row[d.n_rows_padded];

        std::sort(begin, end,
                  // sort longer rows first
                  [](const auto & a, const auto & b) {
                    return a.second > b.second;
                  });
    }

    // determine chunk_ptrs and chunk_lengths

    // TODO: check chunk_ptrs can overflow
    // std::cout << d.n_chunks << std::endl;
    d.chunk_lengths = V<IT, IT>(d.n_chunks); // init a vector of length d.n_chunks
    d.chunk_ptrs    = V<IT, IT>(d.n_chunks + 1);

    IT cur_chunk_ptr = 0;
    
    for (ST i = 0; i < d.n_chunks; ++i) {
        auto begin = &n_els_per_row[i * d.C];
        auto end   = &n_els_per_row[i * d.C + d.C];

        d.chunk_lengths[i] =
                std::max_element(begin, end,
                    [](const auto & a, const auto & b) {
                        return a.second < b.second;
                    })->second;

        if (will_add_overflow(cur_chunk_ptr, d.chunk_lengths[i] * (IT)d.C)) {
            fprintf(stderr, "ERROR: chunck_ptrs exceed index type.\n");
            return false;
        }

        d.chunk_ptrs[i] = cur_chunk_ptr;
        cur_chunk_ptr += d.chunk_lengths[i] * d.C;
    }

    

    ST n_scs_elements = d.chunk_ptrs[d.n_chunks - 1]
                        + d.chunk_lengths[d.n_chunks - 1] * d.C;
    d.chunk_ptrs[d.n_chunks] = n_scs_elements;

    // construct permutation vector

    d.old_to_new_idx = V<IT, IT>(d.n_rows);

    for (ST i = 0; i < d.n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < d.n_rows) {
            d.old_to_new_idx[old_row_idx] = i;
        }
    }

    

    d.values   = V<VT, IT>(n_scs_elements);
    d.col_idxs = V<IT, IT>(n_scs_elements);

    for (ST i = 0; i < n_scs_elements; ++i) {
        d.values[i]   = VT{};
        d.col_idxs[i] = IT{};
    }

    std::vector<IT> col_idx_in_row(d.n_rows_padded);

    // fill values and col_idxs
    for (ST i = 0; i < d.nnz; ++i) {
        IT row_old = mtx.I[i];
        IT row = d.old_to_new_idx[row_old];

        ST chunk_index = row / d.C;

        IT chunk_start = d.chunk_ptrs[chunk_index];
        IT chunk_row   = row % d.C;

        IT idx = chunk_start + col_idx_in_row[row] * d.C + chunk_row;

        d.col_idxs[idx] = mtx.J[i];
        d.values[idx]   = mtx.values[i];

        col_idx_in_row[row]++;
    }

    d.n_elements = n_scs_elements;

    return true;
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
// template <typename VT, typename IT>
// static void
// convert_to_csr(const MtxData<VT, IT> &mtx,
//                V<IT, IT> &row_ptrs,
//                V<IT, IT> &col_idxs,
//                V<VT, IT> &values)
// {
//     values = V<VT, IT>(mtx.nnz);
//     col_idxs = V<IT, IT>(mtx.nnz);
//     row_ptrs = V<IT, IT>(mtx.n_rows + 1);

//     std::vector<IT> col_offset_in_row(mtx.n_rows);

//     convert_idxs_to_ptrs(mtx.I, row_ptrs);

//     for (ST i = 0; i < mtx.nnz; ++i) {
//         IT row = mtx.I[i];

//         IT idx = row_ptrs[row] + col_offset_in_row[row];

//         col_idxs[idx] = mtx.J[i];
//         values[idx]   = mtx.values[i];

//         col_offset_in_row[row]++;
//     }
// }


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


// /**
//  * Benchmark the OpenMP/GPU spmv kernel with a matrix of dimensions \p n_rows x
//  * \p n_cols.  If \p mmr is not NULL, then the matrix is read via the provided
//  * MatrixMarketReader from file.
//  */
// template <typename VT, typename IT>
// static BenchmarkResult
// bench_spmv(const std::string & kernel_name,
//            const Config & config,
//            const Kernel::entry_t & k_entry,
//            const MtxData<VT, IT> & mtx,
//            DefaultValues<VT, IT> * defaults = nullptr,
//            const std::vector<VT> * x_in = nullptr,
//            std::vector<VT> * y_out_opt = nullptr
// )
// {
//     BenchmarkResult r;

//     std::vector<VT> y_out;
//     std::vector<VT> x_out;

//     DefaultValues<VT, IT> default_values;

//     if (!defaults) {
//         defaults = &default_values;
//     }

//     switch (k_entry.format) {
//     case MatrixFormat::Csr:
//         r = bench_spmv_csr<VT, IT>(config,
//                                    mtx,
//                                    k_entry, *defaults,
//                                    x_out, y_out, x_in);
//         break;
//     case MatrixFormat::EllRm:
//     case MatrixFormat::EllCm:
//         r = bench_spmv_ell<VT, IT>(config,
//                                    mtx,
//                                    k_entry, *defaults,
//                                    x_out, y_out, x_in);
//         break;
//     case MatrixFormat::SellCSigma:
//         r = bench_spmv_scs<VT, IT>(config,
//                                    mtx,
//                                    k_entry, *defaults,
//                                    x_out, y_out, x_in);
//         break;    default:
//         fprintf(stderr, "ERROR: SpMV format for kernel %s is not implemented.\n", kernel_name.c_str());
//         return r;
//     }


//     if (config.verify_result_with_coo) {
//         log("verify begin\n");

//         bool ok = spmv_verify(kernel_name, mtx, x_out, y_out);

//         r.is_result_valid = ok;

//         log("verify end\n");
//     }

//     if (y_out_opt) *y_out_opt = std::move(y_out);

//     // if (print_matrices) {
//     //     printf("Matrices for kernel: %s\n", kernel_name.c_str());
//     //     printf("A, is_col_major: %d\n", A.is_col_major);
//     //     print(A);
//     //     printf("b\n");
//     //     print(b);
//     //     printf("x\n");
//     //     print(x);
//     // }

//     return r;
// }
#endif