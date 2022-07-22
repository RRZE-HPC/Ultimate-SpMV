#ifndef BENCHMARKS
#define BENCHMARKS

#include "spmv.h"
#include "mtx-reader.h"
#include "vectors.h"

// #include "utilities.hpp"
#include "structs.hpp"
// #include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "format.hpp"
#include "write_results.hpp"


#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include <set>
#include <mpi.h>
#include <fstream>





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
void bench_spmv_scs(
    Config *config,
    MtxData<VT, IT> *local_mtx,
    ContextData<IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<VT> *dummy_x,
    BenchmarkResult<VT, IT> *r,
    const int *my_rank,
    const int *comm_size
    )
{
    ScsData<VT, IT> scs;

    convert_to_scs<VT, IT>(local_mtx, config->chunk_size, config->sigma, &scs);

    //init


    // TODO: How bad is this for scalability? Is there a way around this?
    // std::vector<VT> local_y_scs_vec(scs.n_rows_padded, 0); //necessary ^^?

    // IT updated_col_idx, initial_col_idx = scs.col_idxs[0];

    // clock_t begin_ahci_time = std::clock();
    adjust_halo_col_idxs<VT, IT>(local_mtx, &scs, work_sharing_arr, my_rank, comm_size); // scs specific
    // if(config->log_prof)
    //     log("adjust_halo_col_idxs", begin_ahci_time, std::clock());
    // Copy local_x to x_out for (optional) validation against mkl later
    // for(IT i = 0; i < local_x.size(); ++i){
    //     (*x_out)[i] = local_x[i];
    // }

    // clock_t begin_main_loop_time = std::clock();
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

        // } while (end - start < 1)
        // NITER = NITER / 2;
    }
    else if(config->mode == 's'){ // Enter main COMM-SPMVM-SWAP loop, solve mode
        for (IT i = 0; i < config->n_repetitions; ++i)
        {
            communicate_halo_elements<VT, IT>(local_context, dummy_x, work_sharing_arr, my_rank, comm_size);

            spmv_omp_scs<VT, IT>(scs.C, scs.n_chunks, scs.chunk_ptrs.data(),
                                scs.chunk_lengths.data(), scs.col_idxs.data(),
                                scs.values.data(), &(*dummy_x)[0], &(*local_y)[0]);

            // if (i != config->n_repetitions - 1){
            std::swap(*dummy_x, *local_y);
            // }
        }

        // NOTE: Is it better just to do the "if i != rev - 1" in the main loop?
        std::swap(*dummy_x, *local_y);

        // for (IT i = 0; i < scs.old_to_new_idx.n_rows; ++i)
        // {
        //     (*y_out)[i] = local_x[scs.old_to_new_idx[i]];
        // }
    }
    // if(config->log_prof)
    //     log("main_loop", begin_main_loop_time, std::clock());

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

/**
    @brief Select the matrix format, and benchmark the spmvm kernal
    @param *config : struct to initialze default values and user input
    @param *mtx : mtx data struct, read in from matrix market format reader mtx-reader.h
    @param *work_sharing_arr : the array describing the partitioning of the rows of the global mtx struct
    @param *y_out : the vector declared to either hold the process local result, 
        or the global result if verification is selected as an option
    @param *defaults : a DefaultValues struct, in which default values of x and y can be defined
    @param *x_in : if one wishes to start with a pre-defined x vector
*/
template <typename VT, typename IT>
static void
bench_spmv(
    Config *config,
    MtxData<VT, IT> *local_mtx,
    ContextData<IT> *local_context,
    const int *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<VT> *local_x,
    BenchmarkResult<VT, IT> *r,
    const int *my_rank,
    const int *comm_size
)
{
    if(config->kernel_format == "csr"){
        // bench_spmv_csr<VT, IT>(
        //     config,
        //     local_mtx,
        //     work_sharing_arr,
        //     y_out,
        //     x_out,
        //     defaults,
        //     r,
        //     x_in
        // );
    }
    else if(config->kernel_format == "ellrm"){}
    else if(config->kernel_format == "ellcm"){}
    else if(config->kernel_format == "ell"){
        // bench_spmv_ell<VT, IT>(
        //     config,
        //     local_mtx,
        //     work_sharing_arr,
        //     y_out,
        //     x_out,
        //     defaults,
        //     r,
        //     x_in
        // );
    }
    else if(config->kernel_format == "scs"){
        bench_spmv_scs<VT, IT>(
            config,
            local_mtx,
            local_context,
            work_sharing_arr,
            local_y,
            local_x,
            r,
            my_rank,
            comm_size
        );
    }
    else{
        fprintf(stderr, "ERROR: SpMV format for kernel %s is not implemented.\n", config->kernel_format.c_str());
    }
}

#endif