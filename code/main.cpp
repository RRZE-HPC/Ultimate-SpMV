#include "mmio.h"
#include "vectors.h"
#include "utilities.hpp"
#include "kernels.hpp"
#include "mpi_funcs.hpp"
#include "write_results.hpp"
#include "timing.h"

#include <cinttypes>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>


#define WARM_UP_REPS 100
#define MILLI_TO_SEC 0.001

#ifdef _OPENMP
#include <omp.h>
#endif

/**
    @brief Perform SPMV kernel, either in "solve" mode or "bench" mode
    @param *config : struct to initialze default values and user input
    @param *local_scs : pointer to process-local scs struct 
    @param *local_context : struct containing communication information
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_y : Process-local results vector, instance of SimpleDenseMatrix class
    @param *local_x : local RHS vector, instance of SimpleDenseMatrix class
    @param *r : a Result struct, in which results of the benchmark are stored
*/
template <typename VT, typename IT>
void bench_spmv(
    Config *config,
    ScsData<VT, IT> *local_scs,
    ScsData<double, IT> *dp_local_scs,
    ScsData<float, IT> *sp_local_scs,
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> *hp_local_scs,
#endif
    ContextData<IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<double> *dp_local_y,
    std::vector<float> *sp_local_y,
#ifdef HAVE_HALF_MATH
    std::vector<_Float16> *hp_local_y,
#endif
    std::vector<VT> *local_x,
    std::vector<double> *dp_local_x,
    std::vector<float> *sp_local_x,
#ifdef HAVE_HALF_MATH
    std::vector<_Float16> *hp_local_x,
#endif
    Result<VT, IT> *r,
    int my_rank,
    int comm_size)
{
    // Permute x, in order to match the permutation which was done to the columns
    std::vector<VT> local_x_permuted(local_x->size(), 0);
    std::vector<double> dp_local_x_permuted(dp_local_x->size(), 0);
    std::vector<float> sp_local_x_permuted(sp_local_x->size(), 0);
#ifdef HAVE_HALF_MATH
    std::vector<_Float16> hp_local_x_permuted(hp_local_x->size(), 0);
#endif

    apply_permutation<VT, IT>(&(local_x_permuted)[0], &(*local_x)[0], &(local_scs->new_to_old_idx)[0], local_scs->n_rows);

    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        // Currently, we fix one sigma. That is, we permute dp and sp exactly the same
        apply_permutation<double, IT>(&(dp_local_x_permuted)[0], &(*dp_local_x)[0], &(dp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
        apply_permutation<float, IT>(&(sp_local_x_permuted)[0], &(*sp_local_x)[0], &(sp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
#ifdef HAVE_HALF_MATH
        apply_permutation<_Float16, IT>(&(hp_local_x_permuted)[0], &(*hp_local_x)[0], &(hp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
#endif
    }

    OnePrecKernelArgs<VT, IT> *one_prec_kernel_args_encoded = new OnePrecKernelArgs<VT, IT>;
    MultiPrecKernelArgs<IT> *multi_prec_kernel_args_encoded = new MultiPrecKernelArgs<IT>;
    void *comm_args_void_ptr;
    void *kernel_args_void_ptr;
    void *cusparse_args_void_ptr;
#ifdef USE_CUSPARSE
    CuSparseArgs *cusparse_args_encoded = new CuSparseArgs;
#endif
    CommArgs<VT, IT> *comm_args_encoded = new CommArgs<VT, IT>;

///////////////////////////////////////////////////////////////////////////////////////////
//     allocate_data<VT, IT>(
//         config,
//         local_scs,
//         hp_local_scs,
//         lp_local_scs,
//         local_context,
//         local_y,
//         hp_local_y,
//         lp_local_y,
//         local_x,
//         &local_x_permuted,
//         hp_local_x,
//         &hp_local_x_permuted,
//         lp_local_x,
//         &lp_local_x_permuted,
//         one_prec_kernel_args_encoded,
//         two_prec_kernel_args_encoded,
// #ifdef USE_CUSPARSE
//         cusparse_args_encoded,
//         // cusparse_args_void_ptr,
// #endif
//         comm_args_encoded,
//         // comm_args_void_ptr,
//         // kernel_args_void_ptr,
//         comm_size,
//         my_rank
//     );

    // void *comm_args_void_ptr;
//     comm_args_void_ptr = (void*) comm_args_encoded;

//     void *kernel_args_void_ptr;
//     if(config->value_type == "ap"){
//         kernel_args_void_ptr = (void*) two_prec_kernel_args_encoded;
//     }
//     else{
//         kernel_args_void_ptr = (void*) one_prec_kernel_args_encoded;
//     }
    
//     void *cusparse_args_void_ptr;
// #ifdef USE_CUSPARSE
//     cusparse_args_void_ptr = (void*) cusparse_args_encoded;
// #endif

// TODO: move allocation into subroutine
///////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_MPI
    // Allocate a send buffer for each process we're sending a message to
    int nz_comms = local_context->non_zero_receivers.size();
    int nz_recver;

    VT *to_send_elems[nz_comms];
    for(int i = 0; i < nz_comms; ++i){
        nz_recver = local_context->non_zero_receivers[i];
        to_send_elems[i] = new VT[local_context->comm_send_idxs[nz_recver].size()];
    }

    int nzr_size = local_context->non_zero_receivers.size();
    int nzs_size = local_context->non_zero_senders.size();

    // Delare MPI requests for non-blocking communication
    MPI_Request *recv_requests = new MPI_Request[local_context->non_zero_senders.size()];
    MPI_Request *send_requests = new MPI_Request[local_context->non_zero_receivers.size()];
#endif

#ifdef __CUDACC__
    // If using cuda compiler, move data to device and assign device pointers
    printf("Moving data to device...\n");
    long n_blocks = (local_scs->n_rows_padded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    config->num_blocks = n_blocks; // Just for ease of results printing later
    config->tpb = THREADS_PER_BLOCK;
    
    VT *d_x = new VT;
    VT *d_y = new VT;
    ST *d_C = new ST;
    ST *d_n_chunks = new ST;
    IT *d_chunk_ptrs = new IT;
    IT *d_chunk_lengths = new IT;
    IT *d_col_idxs = new IT;
    VT *d_values = new VT;
    ST *d_n_blocks = new ST;

    double *d_x_dp = new double;
    double *d_y_dp = new double;
    ST *d_C_dp = new ST;
    ST *d_n_chunks_dp = new ST;
    IT *d_chunk_ptrs_dp = new IT;
    IT *d_chunk_lengths_dp = new IT;
    IT *d_col_idxs_dp = new IT;
    double *d_values_dp = new double;
    float *d_x_sp = new float;
    float *d_y_sp = new float;
    ST *d_C_sp = new ST;
    ST *d_n_chunks_sp = new ST;
    IT *d_chunk_ptrs_sp = new IT;
    IT *d_chunk_lengths_sp = new IT;
    IT *d_col_idxs_sp = new IT;
    float *d_values_sp = new float;
#ifdef HAVE_HALF_MATH
    _Float16 *d_x_sp = new _Float16;
    _Float16 *d_y_sp = new _Float16;
    ST *d_C_sp = new ST;
    ST *d_n_chunks_sp = new ST;
    IT *d_chunk_ptrs_sp = new IT;
    IT *d_chunk_lengths_sp = new IT;
    IT *d_col_idxs_sp = new IT;
    _Float16  *d_values_sp = new _Float16;
#endif

    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        long n_scs_elements_dp = dp_local_scs->chunk_ptrs[dp_local_scs->n_chunks - 1]
                    + dp_local_scs->chunk_lengths[dp_local_scs->n_chunks - 1] * dp_local_scs->C;
        long n_scs_elements_sp = sp_local_scs->chunk_ptrs[sp_local_scs->n_chunks - 1]
                    + sp_local_scs->chunk_lengths[sp_local_scs->n_chunks - 1] * sp_local_scs->C;
        long n_scs_elements_hp = hp_local_scs->chunk_ptrs[hp_local_scs->n_chunks - 1]
                    + hp_local_scs->chunk_lengths[hp_local_scs->n_chunks - 1] * hp_local_scs->C;

        // TODO: temporary way to get around memory courruption problem
        for(int i = 0; i < n_scs_elements_dp; ++i){
            if(dp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad dp element %i found at idx %i\n", dp_local_scs->col_idxs[i], i);
#endif
                dp_local_scs->col_idxs[i] = 0;
            }
        }
        for(int i = 0; i < n_scs_elements_sp; ++i){
            if(sp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad sp %i element found at idx %i\n", sp_local_scs->col_idxs[i], i);
#endif
                sp_local_scs->col_idxs[i] = 0;
            }
        }
        for(int i = 0; i < n_scs_elements_hp; ++i){
            if(hp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad sp %i element found at idx %i\n", hp_local_scs->col_idxs[i], i);
#endif
                hp_local_scs->col_idxs[i] = 0;
            }
        }


        // Allocate space for MP structs on device
        cudaMalloc(&d_values_dp, n_scs_elements_dp*sizeof(double));
        cudaMemcpy(d_values_dp, &(dp_local_scs->values)[0], n_scs_elements_dp*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_values_sp, n_scs_elements_sp*sizeof(float));
        cudaMemcpy(d_values_sp, &(sp_local_scs->values)[0], n_scs_elements_sp*sizeof(float), cudaMemcpyHostToDevice);
#ifdef HAVE_HALF_MATH
        cudaMalloc(&d_values_hp, n_scs_elements_hp*sizeof(_Float16));
        cudaMemcpy(d_values_hp, &(hp_local_scs->values)[0], n_scs_elements_hp*sizeof(_Float16), cudaMemcpyHostToDevice);
#endif

        cudaMalloc(&d_C_dp, sizeof(long));
        cudaMalloc(&d_n_chunks_dp, sizeof(long));
        cudaMalloc(&d_chunk_ptrs_dp, (dp_local_scs->n_chunks + 1)*sizeof(int));
        cudaMalloc(&d_chunk_lengths_dp, dp_local_scs->n_chunks*sizeof(int));
        cudaMalloc(&d_col_idxs_dp, n_scs_elements_dp*sizeof(int));
        cudaMalloc(&d_C_sp, sizeof(long));
        cudaMalloc(&d_n_chunks_sp, sizeof(long));
        cudaMalloc(&d_chunk_ptrs_sp, (sp_local_scs->n_chunks + 1)*sizeof(int));
        cudaMalloc(&d_chunk_lengths_sp, sp_local_scs->n_chunks*sizeof(int));
        cudaMalloc(&d_col_idxs_sp, n_scs_elements_sp*sizeof(int));
#ifdef HAVE_HALF_MATH
        cudaMalloc(&d_C_hp, sizeof(long));
        cudaMalloc(&d_n_chunks_hp, sizeof(long));
        cudaMalloc(&d_chunk_ptrs_hp, (hp_local_scs->n_chunks + 1)*sizeof(int));
        cudaMalloc(&d_chunk_lengths_hp, hp_local_scs->n_chunks*sizeof(int));
        cudaMalloc(&d_col_idxs_hp, n_scs_elements_hp*sizeof(int));
#endif

        // Copy matrix data to device
        cudaMemcpy(d_chunk_ptrs_dp, &(dp_local_scs->chunk_ptrs)[0], (dp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_lengths_dp, &(dp_local_scs->chunk_lengths)[0], dp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idxs_dp, &(dp_local_scs->col_idxs)[0], n_scs_elements_dp*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_dp, &dp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_chunks_dp, &dp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice);

        cudaMemcpy(d_chunk_ptrs_sp, &(sp_local_scs->chunk_ptrs)[0], (sp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_lengths_sp, &(sp_local_scs->chunk_lengths)[0], sp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idxs_sp, &(sp_local_scs->col_idxs)[0], n_scs_elements_sp*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_sp, &sp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_chunks_sp, &sp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice);

#ifdef HAVE_HALF_MATH
        cudaMemcpy(d_chunk_ptrs_hp, &(hp_local_scs->chunk_ptrs)[0], (hp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_lengths_hp, &(hp_local_scs->chunk_lengths)[0], hp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idxs_hp, &(hp_local_scs->col_idxs)[0], n_scs_elements_hp*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_hp, &hp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_chunks_hp, &hp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice);
#endif

        // Copy x and y data to device
        double *local_x_dp_hardcopy = new double[local_scs->n_rows_padded];
        double *local_y_dp_hardcopy = new double[local_scs->n_rows_padded];
        float *local_x_sp_hardcopy = new float[local_scs->n_rows_padded];
        float *local_y_sp_hardcopy = new float[local_scs->n_rows_padded];
#ifdef HAVE_HALF_MATH
        _Float16 *local_x_hp_hardcopy = new _Float16[local_scs->n_rows_padded];
        _Float16 *local_y_hp_hardcopy = new _Float16[local_scs->n_rows_padded];
#endif

        #pragma omp parallel for
        for(int i = 0; i < local_scs->n_rows_padded; ++i){
            local_x_dp_hardcopy[i] = dp_local_x_permuted[i];
            local_y_dp_hardcopy[i] = (*dp_local_y)[i];
            local_x_sp_hardcopy[i] = sp_local_x_permuted[i];
            local_y_sp_hardcopy[i] = (*sp_local_y)[i];
#ifdef HAVE_HALF_MATH
            local_x_hp_hardcopy[i] = hp_local_x_permuted[i];
            local_y_hp_hardcopy[i] = (*hp_local_y)[i];
#endif
        }

        cudaMalloc(&d_x_dp, local_scs->n_rows_padded*sizeof(double));
        cudaMalloc(&d_y_dp, local_scs->n_rows_padded*sizeof(double));
        cudaMalloc(&d_x_sp, local_scs->n_rows_padded*sizeof(float));
        cudaMalloc(&d_y_sp, local_scs->n_rows_padded*sizeof(float));
#ifdef HAVE_HALF_MATH
        cudaMalloc(&d_x_sp, local_scs->n_rows_padded*sizeof(_Float16));
        cudaMalloc(&d_y_sp, local_scs->n_rows_padded*sizeof(_Float16));
#endif

        cudaMemcpy(d_x_dp, local_x_dp_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_dp, local_y_dp_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_sp, local_x_sp_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_sp, local_y_sp_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice);
#ifdef HAVE_HALF_MATH
        cudaMemcpy(d_x_hp, local_x_hp_hardcopy, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_hp, local_y_hp_hardcopy, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyHostToDevice);
#endif

        delete local_x_dp_hardcopy;
        delete local_y_dp_hardcopy;
        delete local_x_sp_hardcopy;
        delete local_y_sp_hardcopy;
        delete local_x_hp_hardcopy;
        delete local_y_hp_hardcopy;

        // Pack pointers into struct 
        // TODO: allow for each struct to have it's own C
        multi_prec_kernel_args_encoded->dp_C =             d_C_dp;
        multi_prec_kernel_args_encoded->dp_n_chunks =      d_n_chunks_dp; //shared for now
        multi_prec_kernel_args_encoded->dp_chunk_ptrs =    d_chunk_ptrs_dp;
        multi_prec_kernel_args_encoded->dp_chunk_lengths = d_chunk_lengths_dp;
        multi_prec_kernel_args_encoded->dp_col_idxs =      d_col_idxs_dp;
        multi_prec_kernel_args_encoded->dp_values =        d_values_dp;
        multi_prec_kernel_args_encoded->dp_local_x =       d_x_dp;
        multi_prec_kernel_args_encoded->dp_local_y =       d_y_dp;
        multi_prec_kernel_args_encoded->sp_C =             d_C_dp; // shared maybe don't?
        multi_prec_kernel_args_encoded->sp_n_chunks =      d_n_chunks_dp; //shared for now
        multi_prec_kernel_args_encoded->sp_chunk_ptrs =    d_chunk_ptrs_sp;
        multi_prec_kernel_args_encoded->sp_chunk_lengths = d_chunk_lengths_sp;
        multi_prec_kernel_args_encoded->sp_col_idxs =      d_col_idxs_sp;
        multi_prec_kernel_args_encoded->sp_values =        d_values_sp;
        multi_prec_kernel_args_encoded->sp_local_x =       d_x_sp;
        multi_prec_kernel_args_encoded->sp_local_y =       d_y_sp;
        multi_prec_kernel_args_encoded->hp_C =             d_C_dp; // shared maybe don't?
        multi_prec_kernel_args_encoded->hp_n_chunks =      d_n_chunks_dp; //shared for now
        multi_prec_kernel_args_encoded->hp_chunk_ptrs =    d_chunk_ptrs_hp;
        multi_prec_kernel_args_encoded->hp_chunk_lengths = d_chunk_lengths_hp;
        multi_prec_kernel_args_encoded->hp_col_idxs =      d_col_idxs_hp;
        multi_prec_kernel_args_encoded->hp_values =        d_values_hp;
        multi_prec_kernel_args_encoded->hp_local_x =       d_x_hp;
        multi_prec_kernel_args_encoded->hp_local_y =       d_y_hp;
        multi_prec_kernel_args_encoded->n_blocks =         &n_blocks;
        kernel_args_void_ptr = (void*) multi_prec_kernel_args_encoded;

    }
    else{
        long n_scs_elements = local_scs->chunk_ptrs[local_scs->n_chunks - 1]
                    + local_scs->chunk_lengths[local_scs->n_chunks - 1] * local_scs->C;

        if(config->value_type == "dp"){
            cudaMalloc(&d_values, n_scs_elements*sizeof(double));
            cudaMemcpy(d_values, &(local_scs->values)[0], n_scs_elements*sizeof(double), cudaMemcpyHostToDevice);
        }
        else if(config->value_type == "sp"){
            cudaMalloc(&d_values, n_scs_elements*sizeof(float));
            cudaMemcpy(d_values, &(local_scs->values)[0], n_scs_elements*sizeof(float), cudaMemcpyHostToDevice);
        }
        else if(config->value_type == "hp"){
#ifdef HAVE_HALF_MATH
            cudaMalloc(&d_values, n_scs_elements*sizeof(_Float16));
            cudaMemcpy(d_values, &(local_scs->values)[0], n_scs_elements*sizeof(_Float16), cudaMemcpyHostToDevice);
#endif
        }
        
        cudaMalloc(&d_C, sizeof(long));
        cudaMalloc(&d_n_chunks, sizeof(long));
        cudaMalloc(&d_chunk_ptrs, (local_scs->n_chunks + 1)*sizeof(int));
        cudaMalloc(&d_chunk_lengths, local_scs->n_chunks*sizeof(int));
        cudaMalloc(&d_col_idxs, n_scs_elements*sizeof(int));

        cudaMemcpy(d_chunk_ptrs, &(local_scs->chunk_ptrs)[0], (local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_lengths, &(local_scs->chunk_lengths)[0], local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idxs, &(local_scs->col_idxs)[0], n_scs_elements*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, &local_scs->C, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_chunks, &local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice);

        if(config->value_type == "dp"){
            // Make type-specific copy to send to device
            double *local_x_hardcopy = new double[local_scs->n_rows_padded];
            double *local_y_hardcopy = new double[local_scs->n_rows_padded];

            #pragma omp parallel for
            for(int i = 0; i < local_scs->n_rows_padded; ++i){
                local_x_hardcopy[i] = local_x_permuted[i];
                local_y_hardcopy[i] = (*local_y)[i];
            }

            cudaMalloc(&d_x, local_scs->n_rows_padded*sizeof(double));
            cudaMalloc(&d_y, local_scs->n_rows_padded*sizeof(double));

            cudaMemcpy(d_x, local_x_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, local_y_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice);

            delete local_x_hardcopy;
            delete local_y_hardcopy;
        }
        else if (config->value_type == "sp"){
            // Make type-specific copy to send to device
            float *local_x_hardcopy = new float[local_scs->n_rows_padded];
            float *local_y_hardcopy = new float[local_scs->n_rows_padded];

            #pragma omp parallel for
            for(int i = 0; i < local_scs->n_rows_padded; ++i){
                local_x_hardcopy[i] = local_x_permuted[i];
                local_y_hardcopy[i] = (*local_y)[i];
            }

            cudaMalloc(&d_x, local_scs->n_rows_padded*sizeof(float));
            cudaMalloc(&d_y, local_scs->n_rows_padded*sizeof(float));

            cudaMemcpy(d_x, local_x_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, local_y_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice);   

            delete local_x_hardcopy;
            delete local_y_hardcopy;
        }
        else if (config->value_type == "hp"){
#ifdef HAVE_HALF_MATH
            // Make type-specific copy to send to device
            _Float16 *local_x_hardcopy = new _Float16[local_scs->n_rows_padded];
            _Float16 *local_y_hardcopy = new _Float16[local_scs->n_rows_padded];

            #pragma omp parallel for
            for(int i = 0; i < local_scs->n_rows_padded; ++i){
                local_x_hardcopy[i] = local_x_permuted[i];
                local_y_hardcopy[i] = (*local_y)[i];
            }

            cudaMalloc(&d_x, local_scs->n_rows_padded*sizeof(_Float16));
            cudaMalloc(&d_y, local_scs->n_rows_padded*sizeof(_Float16));

            cudaMemcpy(d_x, local_x_hardcopy, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, local_y_hardcopy, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyHostToDevice);   

            delete local_x_hardcopy;
            delete local_y_hardcopy;
#endif
        }


        // All args for kernel reside on the device
        one_prec_kernel_args_encoded->C =             d_C;
        one_prec_kernel_args_encoded->n_chunks =      d_n_chunks;
        one_prec_kernel_args_encoded->chunk_ptrs =    d_chunk_ptrs;
        one_prec_kernel_args_encoded->chunk_lengths = d_chunk_lengths;
        one_prec_kernel_args_encoded->col_idxs =      d_col_idxs;
        one_prec_kernel_args_encoded->values =        d_values;
        one_prec_kernel_args_encoded->local_x =       d_x;
        one_prec_kernel_args_encoded->local_y =       d_y;
        one_prec_kernel_args_encoded->n_blocks =      &n_blocks;
        kernel_args_void_ptr = (void*) one_prec_kernel_args_encoded;
    }
#ifdef USE_CUSPARSE
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float     alpha           = 1.0f;
    float     beta            = 0.0f;

    cusparseCreate(&handle);

    if (config->kernel_format == "crs"){
        if(config->value_type == "dp"){
            cusparseCreateCsr(&matA, local_scs->n_rows, local_scs->n_cols, local_scs->nnz, 
                d_chunk_ptrs, d_col_idxs, d_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
            );
            cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_64F);
            cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_64F);

            cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            );
        }
        else if(config->value_type == "sp"){
            cusparseCreateCsr(&matA, local_scs->n_rows, local_scs->n_cols, local_scs->nnz, 
                d_chunk_ptrs, d_col_idxs, d_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
            cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_32F);
            cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_32F);

            cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            );
        }
        else if(config->value_type == "hp"){
            cusparseCreateCsr(&matA, local_scs->n_rows, local_scs->n_cols, local_scs->nnz, 
                d_chunk_ptrs, d_col_idxs, d_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
            cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_16F);
            cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_16F);

            cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_16F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            );
        }
        else{
            printf("CuSparse SpMV only enabled with CRS format in DP, SP, or HP\n");
            exit(1);
        }

    }
    else{
        if(config->value_type == "dp"){
            cusparseCreateSlicedEll(
                &matA, 
                local_scs->n_rows, 
                local_scs->n_cols, 
                local_scs->nnz,
                local_scs->n_elements,
                local_scs->C,
                d_chunk_ptrs,
                d_col_idxs,
                d_values,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_64F
            );
            // Create dense vector X
            cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_64F);
            // Create dense vector y
            cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_64F);
            // allocate an external buffer if needed
            cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            );
        }
        else if(config->value_type == "sp"){
            cusparseCreateSlicedEll(
                &matA, 
                local_scs->n_rows, 
                local_scs->n_cols, 
                local_scs->nnz,
                local_scs->n_elements,
                local_scs->C,
                d_chunk_ptrs,
                d_col_idxs,
                d_values,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_32F
            );
            // Create dense vector X
            cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_32F);
            // Create dense vector y
            cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_32F);
            // allocate an external buffer if needed
            cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            );
        }
        else if(config->value_type == "hp"){
            cusparseCreateSlicedEll(
                &matA, 
                local_scs->n_rows_padded, 
                local_scs->n_cols, 
                local_scs->nnz,
                local_scs->n_elements,
                local_scs->C,
                d_chunk_ptrs,
                d_col_idxs,
                d_values,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_16F
            );
            // Create dense vector X
            cusparseCreateDnVec(&vecX, local_scs->n_rows_padded, d_x, CUDA_R_16F);
            // Create dense vector y
            cusparseCreateDnVec(&vecY, local_scs->n_rows_padded, d_y, CUDA_R_16F);
            // allocate an external buffer if needed
            cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_16F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            );
        }
        else{
            printf("CuSparse SELL-P only enabled with SCS format in DP, SP, or HP\n");
            exit(1);
        }
    }

    cudaMalloc(&dBuffer, bufferSize);

    cusparse_args_encoded->handle = handle;
    cusparse_args_encoded->opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparse_args_encoded->alpha = &alpha;
    cusparse_args_encoded->matA = matA;
    cusparse_args_encoded->vecX = vecX;
    cusparse_args_encoded->beta = &beta;
    cusparse_args_encoded->vecY = vecY;

    if(config->value_type == "dp"){
        cusparse_args_encoded->computeType = CUDA_R_64F;
    }
    else if(config->value_type == "sp"){
        cusparse_args_encoded->computeType = CUDA_R_32F;
    }
    else if(config->value_type == "hp"){
        cusparse_args_encoded->computeType = CUDA_R_16F;
    }

    cusparse_args_encoded->alg = CUSPARSE_SPMV_ALG_DEFAULT;
    cusparse_args_encoded->externalBuffer = dBuffer;
    cusparse_args_void_ptr = (void*) cusparse_args_encoded;
#endif
#else
    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        multi_prec_kernel_args_encoded->dp_C = &dp_local_scs->C;
        multi_prec_kernel_args_encoded->dp_n_chunks = &dp_local_scs->n_chunks; //shared for now
        multi_prec_kernel_args_encoded->dp_chunk_ptrs = dp_local_scs->chunk_ptrs.data();
        multi_prec_kernel_args_encoded->dp_chunk_lengths = dp_local_scs->chunk_lengths.data();
        multi_prec_kernel_args_encoded->dp_col_idxs = dp_local_scs->col_idxs.data();
        multi_prec_kernel_args_encoded->dp_values = dp_local_scs->values.data();
        multi_prec_kernel_args_encoded->dp_local_x = &(dp_local_x_permuted)[0];
        multi_prec_kernel_args_encoded->dp_local_y = &(*dp_local_y)[0];
        multi_prec_kernel_args_encoded->sp_C = &sp_local_scs->C;
        multi_prec_kernel_args_encoded->sp_n_chunks = &dp_local_scs->n_chunks; //shared for now
        multi_prec_kernel_args_encoded->sp_chunk_ptrs = sp_local_scs->chunk_ptrs.data();
        multi_prec_kernel_args_encoded->sp_chunk_lengths = sp_local_scs->chunk_lengths.data();
        multi_prec_kernel_args_encoded->sp_col_idxs = sp_local_scs->col_idxs.data();
        multi_prec_kernel_args_encoded->sp_values = sp_local_scs->values.data();
        multi_prec_kernel_args_encoded->sp_local_x = &(sp_local_x_permuted)[0];
        multi_prec_kernel_args_encoded->sp_local_y = &(*sp_local_y)[0];
#ifdef HAVE_HALF_MATH
        multi_prec_kernel_args_encoded->hp_C = &hp_local_scs->C;
        multi_prec_kernel_args_encoded->hp_n_chunks = &dp_local_scs->n_chunks; //shared for now
        multi_prec_kernel_args_encoded->hp_chunk_ptrs = hp_local_scs->chunk_ptrs.data();
        multi_prec_kernel_args_encoded->hp_chunk_lengths = hp_local_scs->chunk_lengths.data();
        multi_prec_kernel_args_encoded->hp_col_idxs = hp_local_scs->col_idxs.data();
        multi_prec_kernel_args_encoded->hp_values = hp_local_scs->values.data();
        multi_prec_kernel_args_encoded->hp_local_x = &(hp_local_x_permuted)[0];
        multi_prec_kernel_args_encoded->hp_local_y = &(*hp_local_y)[0];
#endif
        kernel_args_void_ptr = (void*) multi_prec_kernel_args_encoded;
    }
    else{
        // Encode kernel args into struct
        one_prec_kernel_args_encoded->C = &local_scs->C;
        one_prec_kernel_args_encoded->n_chunks = &local_scs->n_chunks;
        one_prec_kernel_args_encoded->chunk_ptrs = local_scs->chunk_ptrs.data();
        one_prec_kernel_args_encoded->chunk_lengths = local_scs->chunk_lengths.data();
        one_prec_kernel_args_encoded->col_idxs = local_scs->col_idxs.data();
        one_prec_kernel_args_encoded->values = local_scs->values.data();
        one_prec_kernel_args_encoded->local_x = &(local_x_permuted)[0];
        one_prec_kernel_args_encoded->local_y = &(*local_y)[0];
        kernel_args_void_ptr = (void*) one_prec_kernel_args_encoded;
    }

#ifdef USE_MPI
    // Encode comm args into struct
    comm_args_encoded->local_context = local_context;
    comm_args_encoded->to_send_elems = to_send_elems;
    comm_args_encoded->work_sharing_arr = work_sharing_arr;
    comm_args_encoded->perm = local_scs->old_to_new_idx.data();
    comm_args_encoded->recv_requests = recv_requests; // pointer to first element of array
    comm_args_encoded->nzs_size = &nzs_size;
    comm_args_encoded->send_requests = send_requests;
    comm_args_encoded->nzr_size = &nzr_size;
    comm_args_encoded->num_local_elems = &(local_context->num_local_rows);
#endif
#endif

    comm_args_encoded->my_rank = &my_rank;
    comm_args_encoded->comm_size = &comm_size;
    comm_args_void_ptr = (void*) comm_args_encoded;
/////////////////////////////////////////////////////////////////////////////////////////////////////

    // Pass args to construct spmv_kernel object

    SpmvKernel<VT, IT> spmv_kernel(
        config, 
        kernel_args_void_ptr, 
        cusparse_args_void_ptr,
        comm_args_void_ptr
    );

    // Enter main COMM-SPMV-SWAP loop, bench mode
    if(config->mode == 'b'){
#ifdef __CUDACC__
    cudaEvent_t start, stop, warmup_start, warmup_stop;
    cudaEventCreate(&warmup_start);
    cudaEventCreate(&warmup_stop);
#endif

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Warm-up
#ifdef __CUDACC__
    cudaEventRecord(warmup_start, 0);
    cudaDeviceSynchronize();
#else
        double begin_warm_up_loop_time, end_warm_up_loop_time;
        
#ifdef USE_MPI
        begin_warm_up_loop_time = MPI_Wtime();     
#else
        begin_warm_up_loop_time = getTimeStamp();
#endif
#endif
        for(int k = 0; k < WARM_UP_REPS; ++k){
#ifdef USE_MPI
            spmv_kernel.init_halo_exchange();
            spmv_kernel.finalize_halo_exchange();
#endif
            spmv_kernel.execute_warmup_spmv();

#ifdef USE_MPI
            if(config->ba_synch)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

#ifdef __CUDACC__
        float warmup_runtime;
        cudaEventRecord(warmup_stop, 0);
        cudaEventSynchronize(warmup_stop);
        cudaEventElapsedTime(&warmup_runtime, warmup_start, warmup_stop);
        std::cout << "warm up time: " << warmup_runtime * MILLI_TO_SEC << std::endl;
#else

#ifdef USE_MPI
        end_warm_up_loop_time = MPI_Wtime();
#else
        end_warm_up_loop_time = getTimeStamp();
#endif
        std::cout << "warm up time: " << end_warm_up_loop_time - begin_warm_up_loop_time << std::endl;  
#endif

#ifdef __CUDACC__
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#else
        double begin_bench_loop_time, end_bench_loop_time = 0.0;
#endif

        float runtime;

        // Initialize number of repetitions for actual benchmark
        int n_iter = 2;

#ifndef __CUDACC__
#ifdef USE_LIKWID
        register_likwid_markers(config);
#endif
#endif

        if(config->comm_halos){
#ifdef USE_MPI
            do{
                MPI_Barrier(MPI_COMM_WORLD);
                begin_bench_loop_time = MPI_Wtime();
                for(int k=0; k<n_iter; ++k) {
                    spmv_kernel.init_halo_exchange();
                    spmv_kernel.finalize_halo_exchange();
                    spmv_kernel.execute_spmv();
                    spmv_kernel.swap_local_vectors();
                    if(config->ba_synch)
                        MPI_Barrier(MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                n_iter = n_iter*2;
                runtime = MPI_Wtime() - begin_bench_loop_time;
            } while (runtime < config->bench_time);
            n_iter = n_iter/2;
#else
    printf("ERROR: Cannot communicate halo elements.\n \
        Validate that either USE_MPI = 0 and comm_halos = 0,\n \
        or that USE_MPI = 1.\n");
    exit(1);
#endif
        }
        else if(!config->comm_halos){
            do{
#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
                begin_bench_loop_time = MPI_Wtime();
#else
#ifdef __CUDACC__
                cudaEventRecord(start);
#else
                begin_bench_loop_time = getTimeStamp();
#endif
#endif
                
                for(int k=0; k<n_iter; ++k) {
                    spmv_kernel.execute_spmv();
#ifdef USE_MPI
                    if(config->ba_synch)
                        MPI_Barrier(MPI_COMM_WORLD);
#endif
                }
#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                n_iter = n_iter*2;
#ifdef USE_MPI
                runtime = MPI_Wtime() - begin_bench_loop_time;
#else
#ifdef __CUDACC__
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&runtime, start, stop);
#else
                runtime = getTimeStamp() - begin_bench_loop_time;
                // std::cout << "runtime = " << runtime << std::endl;
#endif
#endif
#ifdef __CUDACC__
            } while (runtime * MILLI_TO_SEC < config->bench_time);
#else
            } while (runtime < config->bench_time);
#endif

            n_iter = n_iter/2;
        }
        r->n_calls = n_iter;
#ifdef __CUDACC__
        r->duration_total_s = runtime * MILLI_TO_SEC;
#else
        r->duration_total_s = runtime;
#endif
        r->duration_kernel_s = r->duration_total_s/ r->n_calls;
        r->perf_gflops = (double)local_context->total_nnz * 2.0
                            / r->duration_kernel_s
                            / 1e9;                   // Only count usefull flops
    }
    else if(config->mode == 's') { // Enter main COMM-SPMV-SWAP loop, solve mode
        // Selects the first (n_rows)-many elements of a sorted y vector, and chops off padding
        std::vector<VT> sorted_local_y(local_y->size(), 0);
        std::vector<double> sorted_dp_local_y(local_y->size(), 0);

        for (int i = 0; i < config->n_repetitions; ++i)
        {
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "before comm spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "before comm spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif

#ifdef USE_MPI
            spmv_kernel.init_halo_exchange();
            spmv_kernel.finalize_halo_exchange();
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "after comm spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "after comm spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif
#endif
            spmv_kernel.execute_spmv();

#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "after_kernel spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "after_kernel spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif

            spmv_kernel.swap_local_vectors();
            
#ifdef DEBUG_MODE_FINE
            if(my_rank == 0){
                std::cout << "after_kernel and swap spmv_kernel->local_x" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_x[i] << std::endl;
                }
                std::cout << "after_kernel and swap spmv_kernel->local_y" << std::endl;
                for(int i = 0; i < local_x->size(); ++i){
                    std::cout << spmv_kernel.local_y[i] << std::endl;
                }
            }
#endif

#ifdef USE_MPI
            if(config->ba_synch)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

// Copy permuted results back into local_x
#ifdef __CUDACC__
        if(config->value_type == "ap"){
            cudaMemcpy(&(*dp_local_x)[0], spmv_kernel.dp_local_x, local_scs->n_rows_padded*sizeof(double), cudaMemcpyDeviceToHost);
        }

        if(config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){
            if(config->value_type == "dp")
                cudaMemcpy(&(*dp_local_x)[0], spmv_kernel.dp_local_x, local_scs->n_rows_padded*sizeof(double), cudaMemcpyDeviceToHost);
            else if(config->value_type == "sp")
                cudaMemcpy(&(*sp_local_x)[0], spmv_kernel.sp_local_x, local_scs->n_rows_padded*sizeof(float), cudaMemcpyDeviceToHost);
            else if(config->value_type == "hp"){
#ifdef HAVE_HALF_MATH
                cudaMemcpy(&(*hp_local_x)[0], spmv_kernel.hp_local_x, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyDeviceToHost);
#endif
            }
        }
        else if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_hp]"){
            printf("ERROR: copy back to host not yet implemented for ap.\n");
            exit(1);

        }
        else if(config->value_type == "ap[dp_sp_hp]"){
            printf("ERROR: copy back to host not yet implemented for ap.\n");
            exit(1);

        }

#else
        if(config->value_type == "ap[dp_sp_hp]" || config->value_type == "ap[dp_sp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_hp]"){
            printf("ERROR: Results collection not yet implemented for ap.\n");
            exit(1);
            for(int i = 0; i < local_scs->n_rows_padded; ++i){
                (*dp_local_x)[i] = (spmv_kernel.dp_local_x)[i];
            }
        }
        else{
            for(int i = 0; i < local_scs->n_rows_padded; ++i){
                (*local_x)[i] = (spmv_kernel.local_x)[i];
            }
        }
#endif

        // TODO: Clean up!
        if(config->value_type == "ap[dp_sp_hp]" || config->value_type == "ap[dp_sp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_hp]"){
            printf("ERROR: Results collection not yet implemented for ap.\n");
            exit(1);
            apply_permutation<double, IT>(&(sorted_dp_local_y)[0], &(*dp_local_x)[0], &(dp_local_scs->old_to_new_idx)[0], dp_local_scs->n_rows);
        
            // Give result to local_y for results output
            for(int i = 0; i < local_y->size(); ++i){
                (*local_y)[i] = (sorted_dp_local_y)[i];
            }
        }
        else{

            apply_permutation<VT, IT>(&(sorted_local_y)[0], &(*local_x)[0], &(local_scs->old_to_new_idx)[0], local_scs->n_rows);
                
            // Give result to local_y for results output
            for(int i = 0; i < local_y->size(); ++i){
                (*local_y)[i] = (sorted_local_y)[i];
            }

        }

        // Manually resize for ease later on (and I don't see a better way)
        local_y->resize(local_context->num_local_rows);
    }

    // Delete the allocated space for each other process send buffers
#ifdef USE_MPI
    for(int i = 0; i < nz_comms; ++i){
        delete[] to_send_elems[i];
    }
#endif

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

    if(config->value_type == "dp")
        r->value_type_str = "double";
    
    else if(config->value_type == "sp")
        r->value_type_str = "float";
    else
        r->value_type_str = "half";

    r->index_type_str = typeid(IT).name();
    r->value_type_size = sizeof(VT);
    r->index_type_size = sizeof(IT);

    // TODO: ????
    // r->was_matrix_sorted = local_scs->is_sorted;
    r->was_matrix_sorted = 1;

    r->fill_in_percent = ((double)local_scs->n_elements / local_scs->nnz - 1.0) * 100.0;
    r->C               = local_scs->C;
    r->sigma           = local_scs->sigma;
    r->beta = (double)local_scs->nnz / local_scs->n_elements;

    // Only relevant for adaptive precision
    r->dp_nnz = dp_local_scs->nnz;
    r->sp_nnz = sp_local_scs->nnz;
#ifdef HAVE_HALF_MATH
    r->hp_nnz = hp_local_scs->nnz;
#endif

    r->dp_beta = (double)dp_local_scs->nnz / dp_local_scs->n_elements;
    if(dp_local_scs->n_elements == 0)
        r->dp_beta = 0;
    r->sp_beta = (double)sp_local_scs->nnz / sp_local_scs->n_elements;
    if(sp_local_scs->n_elements == 0)
        r->sp_beta = 0;
#ifdef HAVE_HALF_MATH
    r->hp_beta = (double)hp_local_scs->nnz / hp_local_scs->n_elements;
    if(hp_local_scs->n_elements == 0)
        r->hp_beta = 0;
#endif

// TODO: How to destroy out here?
// #ifdef USE_MPI
//     delete[] recv_requests;
//     delete[] send_requests;
// #endif

// #ifdef USE_CUSPARSE
//     // destroy matrix/vector descriptors
//     cusparseDestroySpMat(matA);
//     cusparseDestroyDnVec(vecX);
//     cusparseDestroyDnVec(vecY);
//     cusparseDestroy(handle);
// #endif

// TODO: Memcheck doesn't like this for some reason
// #ifdef __CUDACC__
//     if(config->value_type == "ap"){
//         cudaFree(d_x_hp);
//         cudaFree(d_y_hp);
//         cudaFree(d_C_hp);
//         cudaFree(d_n_chunks_hp);
//         cudaFree(d_chunk_ptrs_hp);
//         cudaFree(d_chunk_lengths_hp);
//         cudaFree(d_col_idxs_hp);
//         cudaFree(d_values_hp);
//         cudaFree(d_x_lp);
//         cudaFree(d_y_lp);
//         cudaFree(d_C_lp);
//         cudaFree(d_n_chunks_lp);
//         cudaFree(d_chunk_ptrs_lp);
//         cudaFree(d_chunk_lengths_lp);
//         cudaFree(d_col_idxs_lp);
//         cudaFree(d_values_lp);
//     }
//         cudaFree(d_x);
//         cudaFree(d_y);
//         cudaFree(d_C);
//         cudaFree(d_n_chunks);
//         cudaFree(d_chunk_ptrs);
//         cudaFree(d_chunk_lengths);
//         cudaFree(d_col_idxs);
//         cudaFree(d_values);
// #endif

    delete comm_args_encoded;
    delete one_prec_kernel_args_encoded;
    delete multi_prec_kernel_args_encoded;
}

/**
    @brief Gather results (either result of computation, or benchmark metrics) to the root MPI process
    @param *config : struct to initialze default values and user input
    @param *r : a Result struct, in which results of the benchmark are stored
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *local_x_copy : copy of local RHS vector used for validation against MKL
    @param *local_y : Process-local results vector, instance of SimpleDenseMatrix class
*/
template <typename VT, typename IT>
void gather_results(
    Config *config,
    Result<VT, IT> *r,
    IT *work_sharing_arr,
    std::vector<VT> *local_x_copy,
    std::vector<VT> *local_y,
    int my_rank,
    int comm_size
){
    IT num_local_rows = 0;

#ifdef USE_MPI
    num_local_rows = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
#else
    num_local_rows = local_y->size();
#endif

    if(config->mode == 'b'){

        double *perfs_from_procs_arr = new double[comm_size];
        unsigned long *nnz_per_procs_arr = new unsigned long[comm_size];
        unsigned long *dp_nnz_per_procs_arr = new unsigned long[comm_size];
        unsigned long *sp_nnz_per_procs_arr = new unsigned long[comm_size];
#ifdef USE_MPI

        MPI_Gather(&(r->perf_gflops),
                1,
                MPI_DOUBLE,
                perfs_from_procs_arr,
                1,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

        MPI_Gather(&(r->dp_nnz),
                1,
                MPI_UNSIGNED_LONG,
                dp_nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);

        MPI_Gather(&(r->sp_nnz),
                1,
                MPI_UNSIGNED_LONG,
                sp_nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);

        MPI_Gather(&(r->nnz),
                1,
                MPI_UNSIGNED_LONG,
                nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);

        // NOTE: Garbage values for all but root process
        r->perfs_from_procs = std::vector<double>(perfs_from_procs_arr, perfs_from_procs_arr + comm_size);
        r->nnz_per_proc = std::vector<unsigned long>(nnz_per_procs_arr, nnz_per_procs_arr + comm_size);

        delete[] perfs_from_procs_arr;
        delete[] nnz_per_procs_arr;
#else
        perfs_from_procs_arr[0] = r->perf_gflops;
        nnz_per_procs_arr[0] = r->nnz;

        r->cumulative_dp_nnz = r->dp_nnz;
        r->cumulative_sp_nnz = r->sp_nnz;

        r->total_dp_percent = (r->cumulative_dp_nnz / (double)r->total_nnz) * 100.0;
        r->total_sp_percent = (r->cumulative_sp_nnz / (double)r->total_nnz) * 100.0;

#ifdef HAVE_HALF_MATH
        r->cumulative_hp_nnz = r->hp_nnz;
        r->total_hp_percent = (r->cumulative_hp_nnz / (double)r->total_nnz) * 100.0;
#endif

#endif
        // NOTE: Garbage values for all but root process
        r->perfs_from_procs = std::vector<double>(perfs_from_procs_arr, perfs_from_procs_arr + comm_size);
        r->nnz_per_proc = std::vector<unsigned long>(nnz_per_procs_arr, nnz_per_procs_arr + comm_size);

        delete[] perfs_from_procs_arr;
        delete[] nnz_per_procs_arr;

    }
    else if(config->mode == 's'){
        std::vector<VT> sorted_local_y(num_local_rows);
        r->x_out = (*local_x_copy);
        r->y_out = (*local_y);

        if (config->validate_result)
        {
#ifdef USE_MPI
            // TODO: is the size correct here?
            std::vector<VT> total_uspmv_result(work_sharing_arr[comm_size], 0);
            std::vector<VT> total_x(work_sharing_arr[comm_size], 0);

            IT counts_arr[comm_size];
            IT displ_arr_bk[comm_size];

            for(IT i = 0; i < comm_size; ++i){
                counts_arr[i] = IT{};
                displ_arr_bk[i] = IT{};
            }
            
            for (IT i = 0; i < comm_size; ++i){
                counts_arr[i] = work_sharing_arr[i + 1] - work_sharing_arr[i];
                displ_arr_bk[i] = work_sharing_arr[i];
            }

            if (config->value_type == "dp"){
                // Gather all y_vector results to root
                MPI_Gatherv(&(r->y_out)[0],
                            num_local_rows,
                            MPI_DOUBLE,
                            &total_uspmv_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);
                // Gather all x_vector copies to root for mkl validation
                MPI_Gatherv(&(r->x_out)[0],
                            num_local_rows,
                            MPI_DOUBLE,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);
            }
            else if (config->value_type == "sp"){
                MPI_Gatherv(&(r->y_out)[0],
                            num_local_rows,
                            MPI_FLOAT,
                            &total_uspmv_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);

                MPI_Gatherv(&(r->x_out)[0],
                            num_local_rows,
                            MPI_FLOAT,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);
            }
            else if (config->value_type == "hp"){
                // TODO: Need to define your own MPI datatype for HALF
            }

            // If we're verifying results, assign total vectors to Result object
            // NOTE: Garbage values for all but root process
            r->total_x = total_x;
            r->total_uspmv_result = total_uspmv_result;

#ifdef DEBUG_MODE_FINE
            std::cout << "r->total_x = [" << std::endl;
            for(int i = 0; i < 256; ++i){
                std::cout << r->total_x[i] << ", ";
            }
            std::cout << "]" << std::endl;

            std::cout << "r->total_uspmv_result = [" << std::endl;
            for(int i = 0; i < 256; ++i){
                std::cout << r->total_uspmv_result[i] << ", ";
            }
            std::cout << "]" << std::endl;
#endif

#else
            r->total_x = (*local_x_copy);
            r->total_uspmv_result = r->y_out;
#endif
        }
    }
}

/** 
    @brief Initialize total_mtx, segment and send this to local_mtx, convert to local_scs format, init comm information
    @param *local_scs : pointer to process-local scs struct
    @param *local_context : struct containing local_scs + communication information
    @param *total_mtx : global mtx struct
    @param *config : struct to initialze default values and user input
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template<typename VT, typename IT>
void init_local_structs(
    ScsData<VT, IT> *local_scs,
    ScsData<double, IT> *dp_local_scs,
    ScsData<float, IT> *sp_local_scs,
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> *hp_local_scs,
#endif
    ContextData<IT> *local_context,
    MtxData<VT, IT> *total_mtx,
    Config *config, // shouldn't this be const?
    IT *work_sharing_arr,
    int my_rank,
    int comm_size,
    int* metis_part = NULL,
    int* metis_perm = NULL,
    int* metis_inv_perm = NULL)
{
    MtxData<VT, IT> *local_mtx = new MtxData<VT, IT>;

    local_context->total_nnz = total_mtx->nnz;

#ifdef USE_MPI

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Segmenting and sending work to other processes.\n");}
#endif

    seg_and_send_work_sharing_arr<VT, IT>(config, total_mtx, work_sharing_arr, my_rank, comm_size, metis_part, metis_perm, metis_inv_perm);

    seg_and_send_matrix_data<VT, IT>(config, total_mtx, local_mtx, work_sharing_arr, my_rank, comm_size);

    localize_row_idx<VT, IT>(local_mtx);
#else
    local_mtx = total_mtx;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Converting COO matrix to SELL-C-SIG and permuting locally (NOTE: rows only, i.e. nonsymetrically).\n");}
#endif

    // If desired, scale the one-precision matrix
    if(
        (config->value_type == "dp" || 
        config->value_type == "sp" || 
        config->value_type == "hp") && 
        config->equilibrate)
    {
        equilibrate_matrix<VT, IT>(local_mtx);
    }

    // extract matrix mean (and give to x-vector if option chosen at cli)
    extract_matrix_min_mean_max(local_mtx, config);

    // convert local_mtx to local_scs (and permute rows if sigma > 1)
    convert_to_scs<VT, VT, IT>(local_mtx, config->chunk_size, config->sigma, local_scs, NULL, work_sharing_arr, my_rank);

    // Only used for adaptive precision
    MtxData<double, int> *dp_local_mtx = new MtxData<double, int>;
    MtxData<float, int> *sp_local_mtx = new MtxData<float, int>;
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *hp_local_mtx = new MtxData<_Float16, int>;
#endif

    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        std::vector<VT> largest_row_elems;//(local_mtx->n_cols, 0.0);
        std::vector<VT> largest_col_elems;//(local_mtx->n_cols, 0.0);

        // Scale local_mtx *and save the largest row and col elements*
        if(config->equilibrate){
            extract_largest_row_elems<VT, IT>(local_mtx, &largest_row_elems);
            scale_matrix_rows<VT, IT>(local_mtx, &largest_row_elems);

            extract_largest_col_elems<VT, IT>(local_mtx, &largest_col_elems);
            scale_matrix_cols<VT, IT>(local_mtx, &largest_col_elems);
        }

        // Pass largest row and col elements to precision partitioner
        partition_precisions<VT,IT>(
            config, 
            local_mtx, 
            dp_local_mtx, 
            sp_local_mtx,
#ifdef HAVE_HALF_MATH
            hp_local_mtx,
#endif
            &largest_row_elems, 
            &largest_col_elems, 
            my_rank
        );

        // We permute the lower precision struct(s) in the exact same way as the higher precision one
        if(config->value_type == "ap[dp_sp]"){
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
        
            // Empty struct, just pass through convert_to_scs for technical reasons
#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, NULL, work_sharing_arr, my_rank);
#endif
        }
        else if (config->value_type == "ap[dp_hp]"){
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
#endif
            // Empty struct, just pass through convert_to_scs for technical reasons
            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, NULL, work_sharing_arr, my_rank);
        
        }
        else if (config->value_type == "ap[sp_hp]"){
            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, NULL, work_sharing_arr, my_rank);
        
#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, &(sp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
#endif
       
            // Empty struct, just pass through convert_to_scs for technical reasons
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

        }
        else if (config->value_type == "ap[dp_sp_hp]"){
            printf("Converting dp struct\n");
            convert_to_scs<double, double, IT>(dp_local_mtx, config->chunk_size, config->sigma, dp_local_scs, NULL, work_sharing_arr, my_rank);

            printf("Converting sp struct\n");
            convert_to_scs<float, float, IT>(sp_local_mtx, config->chunk_size, config->sigma, sp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
        
#ifdef HAVE_HALF_MATH
            printf("Converting hp struct\n");
            convert_to_scs<_Float16, _Float16, IT>(hp_local_mtx, config->chunk_size, config->sigma, hp_local_scs, &(dp_local_scs->old_to_new_idx)[0], work_sharing_arr, my_rank);
#endif
        }



#ifdef OUTPUT_SPARSITY
        printf("Writing sparsity pattern to output file.\n");
        std::string file_out_name;
        file_out_name = "dp_local_scs";
        dp_local_scs->write_to_mtx_file(my_rank, file_out_name);
        file_out_name = "sp_local_scs";
        sp_local_scs->write_to_mtx_file(my_rank, file_out_name);
#ifdef HAVE_HALF_MATH
        file_out_name = "hp_local_scs";
        hp_local_scs->write_to_mtx_file(my_rank, file_out_name);
#endif
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        exit(0);
#endif
    }

    if (config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){
#ifdef OUTPUT_SPARSITY
        printf("Writing sparsity pattern to output file.\n");
        std::string file_out_name;
        file_out_name = "local_scs";
        local_scs->write_to_mtx_file(my_rank, file_out_name);
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        exit(0);
#endif
    }   

#ifdef USE_MPI
    // TODO: is an array of vectors better?
    // Vector of vecs, Keep track of which remote columns come from which processes
    std::vector<std::vector<IT>> communication_recv_idxs;
    std::vector<std::vector<IT>> communication_send_idxs;
    std::vector<IT> non_zero_receivers;
    std::vector<IT> non_zero_senders;
    std::vector<std::vector<IT>> send_tags;
    std::vector<std::vector<IT>> recv_tags;
    std::vector<IT> recv_counts_cumsum(comm_size + 1, 0);
    std::vector<IT> send_counts_cumsum(comm_size + 1, 0);

    // Main routine for collecting all sending and receiving information!
    collect_comm_info<VT, IT>(
        config, 
        local_scs, 
        work_sharing_arr, 
        &communication_recv_idxs,
        &communication_send_idxs,
        &non_zero_receivers,
        &non_zero_senders,
        &send_tags,
        &recv_tags,
        &recv_counts_cumsum,
        &send_counts_cumsum,
        my_rank,
        comm_size
    );
    
    // Collect all our hard work to single structure for convenience
    // NOTE: not used at all in the no-mpi case
    local_context->comm_send_idxs = communication_send_idxs;
    local_context->comm_recv_idxs = communication_recv_idxs;
    local_context->non_zero_receivers = non_zero_receivers;
    local_context->non_zero_senders = non_zero_senders;
    local_context->send_tags = send_tags;
    local_context->recv_tags = recv_tags;
    local_context->recv_counts_cumsum = recv_counts_cumsum;
    local_context->send_counts_cumsum = send_counts_cumsum;
    local_context->num_local_rows = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
#else
    local_context->num_local_rows = local_scs->n_rows;
#endif
    local_context->scs_padding = (IT)(local_scs->n_rows_padded - local_scs->n_rows);

    // TODO: For symmetric permutation of matrix data
    // permute_scs_cols(local_scs, &(local_scs->old_to_new_idx)[0]);

    // TODO: How to permute columns with here?
    // if (config->value_type == "ap"){
    //     // Permute column indices the same as the original scs struct
    //     // But rows are permuted differently (i.e. within the convert_to_scs routine)
    //     // permute_scs_cols(hp_local_scs, &(hp_local_scs->old_to_new_idx)[0]);
    //     // permute_scs_cols(lp_local_scs, &(hp_local_scs->old_to_new_idx)[0]);
    //     for(int i = 0; i < hp_local_scs->n_elements; ++i){
    //         std::cout << "hp_local_scs->col_idxs[" << i << "] = " << hp_local_scs->col_idxs[i] << std::endl;
    //     }
    //     for(int i = 0; i < lp_local_scs->n_elements; ++i){
    //         std::cout << "lp_local_scs->col_idxs[" << i << "] = " << lp_local_scs->col_idxs[i] << std::endl;
    //     }

    //     permute_scs_cols(hp_local_scs, &(hp_local_scs->old_to_new_idx)[0]);
    //     permute_scs_cols(lp_local_scs, &(lp_local_scs->old_to_new_idx)[0]);

    //     for(int i = 0; i < hp_local_scs->n_elements; ++i){
    //         std::cout << "hp_local_scs->col_idxs[" << i << "] = " << hp_local_scs->col_idxs[i] << std::endl;
    //     }
    //     for(int i = 0; i < lp_local_scs->n_elements; ++i){
    //         std::cout << "lp_local_scs->col_idxs[" << i << "] = " << lp_local_scs->col_idxs[i] << std::endl;
    //     }
    // }

}

/**
    @brief The main harness for the SpMV kernel, in which we:
        1. Segment and distribute the needed structs to each MPI process (init_local_structs),
        2. Benchmark the selected SpMV kernel (bench_spmv),
        3. Gather benchmark results to the root MPI process (gather_results).
    @param *total_mtx : global mtx struct, read from a .mtx file (or generated with ScaMaC TODO)
    @param *config : struct to initialze default values and user input
    @param *r : a Result struct, in which results of the benchmark/computation are stored
*/
template <typename VT, typename IT>
void compute_result(
    MtxData<VT, IT> *total_mtx,
    Config *config,
    Result<VT, IT> *r,
    int my_rank,
    int comm_size)
{
    // TODO: bring back matrix stats!
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    ScsData<VT, IT> local_scs;

    // Declare local structs on each process
    ScsData<double, IT> dp_local_scs;
    ScsData<float, IT> sp_local_scs;
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> hp_local_scs;
#endif

    ContextData<IT> local_context;

    // Used for distributed work sharing
    // Allocate space for work sharing array
    IT work_sharing_arr[comm_size + 1];
    work_sharing_arr[0] = 0; // Initialize first element, since it's used always

    // Used with METIS library, always initialized for convenience
    // Allocate global permutation vectors
    int *metis_part = NULL;
    int *metis_perm = NULL;
    int *metis_inv_perm = NULL;

#ifdef USE_MPI
    if(config->seg_method == "seg-metis"){
        metis_part = new int[total_mtx->n_rows];
        metis_perm = new int[total_mtx->n_rows];
        for(int i = 0; i < total_mtx->n_rows; ++i){
            metis_perm[i] = i;
        }
        metis_inv_perm = new int[total_mtx->n_rows];
    }
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Init local structures.\n");}
#endif

    init_local_structs<VT, IT>(
        &local_scs,
        &dp_local_scs,
        &sp_local_scs,
#ifdef HAVE_HALF_MATH
        &hp_local_scs,
#endif
        &local_context, 
        total_mtx,
        config, 
        work_sharing_arr, 
        my_rank, 
        comm_size,
        metis_part,
        metis_perm,
        metis_inv_perm
    );

    // Declare local vectors to be used
    SimpleDenseMatrix<VT, IT> local_x(&local_context);

    // Must be declared, but only used for mixed precision case
    // TODO: not efficient for storage, but used later for mp interop
    SimpleDenseMatrix<double, IT> dp_local_x(&local_context);
    SimpleDenseMatrix<float, IT> sp_local_x(&local_context);
#ifdef HAVE_HALF_MATH
    SimpleDenseMatrix<_Float16, IT> hp_local_x(&local_context);
#endif

    SimpleDenseMatrix<VT, IT> local_y(&local_context);

    // NOTE: a low precision y vector is needed for swapping with low precision x
    SimpleDenseMatrix<double, IT> dp_local_y(&local_context);
    SimpleDenseMatrix<float, IT> sp_local_y(&local_context);
#ifdef HAVE_HALF_MATH
    SimpleDenseMatrix<_Float16, IT> hp_local_y(&local_context);
#endif

    // Initialize local_x and y, either randomly, with default values defined in classes_structs.hpp,
    // or with 1s (by default)
    local_x.init(config, 'x');
    local_y.init(config, 'y');

    // Copy initialized RHS and LHS into other precisions
    // TODO: wrap in method or something
    for(int i = 0; i < (local_x.vec).size(); ++i){
        dp_local_x.vec[i] = static_cast<double>(local_x.vec[i]);
        sp_local_x.vec[i] = static_cast<float>(local_x.vec[i]);
#ifdef HAVE_HALF_MATH
        hp_local_x.vec[i] = static_cast<_Float16>(local_x.vec[i]);
#endif
    }

    for(int i = 0; i < (local_y.vec).size(); ++i){
        dp_local_y.vec[i] = 0.0;
        sp_local_y.vec[i] = 0.0f;
#ifdef HAVE_HALF_MATH
        hp_local_y.vec[i] = 0.0f16;
#endif
    }

    // Copy contents of local_x for output, and validation against mkl
    std::vector<VT> local_x_copy = local_x.vec;


#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter bench_spmv.\n");}
#endif
    bench_spmv<VT, IT>(
        config,
        &local_scs,
        &dp_local_scs,
        &sp_local_scs,
#ifdef HAVE_HALF_MATH
        &hp_local_scs,
#endif
        &local_context,
        work_sharing_arr,
        &local_y.vec,
        &dp_local_y.vec,
        &sp_local_y.vec,
#ifdef HAVE_HALF_MATH
        &hp_local_x.vec,
#endif
        &local_x.vec,
        &dp_local_x.vec,
        &sp_local_x.vec,
#ifdef HAVE_HALF_MATH
        &hp_local_x.vec,
#endif
        r,
        my_rank,
        comm_size
    );


#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete bench_spmv.\n");}
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Gather results to root process.\n");}
#endif

    gather_results(config, r, work_sharing_arr, &local_x_copy, &(local_y.vec), my_rank, comm_size);

#ifdef USE_MPI
    // Delete allocated permutation vectors, if metis used
    if(config->seg_method == "seg-metis"){
        delete[] metis_part;
        delete[] metis_perm;
        delete[] metis_inv_perm;
    }
#endif
}

void standalone_bench(
    Config config,
    std::string matrix_file_name,
    int my_rank,
    int comm_size,
    double begin_main_time
){
    MtxData<double, int> total_mtx;
    // Replicate structs for each precision
    MtxData<double, int> total_dp_mtx;
    MtxData<float, int> total_sp_mtx;
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> total_hp_mtx;
#endif

    Result<double, int> r_dp;
    Result<float, int> r_sp;
#ifdef HAVE_HALF_MATH
    Result<_Float16, int> r_hp;
#endif

    // The .mtx file is read only by the root process
    if(my_rank == 0){
#ifdef DEBUG_MODE
        if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
        read_mtx(matrix_file_name, config, &total_mtx, my_rank);

        if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
            // copy to total_dp_mtx
            total_dp_mtx.copy(total_mtx);
            r_dp.total_nnz = total_dp_mtx.nnz;
            r_dp.total_rows = total_dp_mtx.n_rows;
        }
        else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
            // copy to total_sp_mtx
            total_sp_mtx.copy(total_mtx);
            r_sp.total_nnz = total_sp_mtx.nnz;
            r_sp.total_rows = total_sp_mtx.n_rows;
        }
        else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
            // copy to total_hp_mtx
            total_hp_mtx.copy(total_mtx);
            r_hp.total_nnz = total_hp_mtx.nnz;
            r_hp.total_rows = total_hp_mtx.n_rows;
#else
            if(my_rank == 0){
                printf("ERROR: Cannot read matrix into HP struct. HAVE_HALF_MATH not defined.\n");
                exit(1);
            }
#endif
        }
        else{
            if(my_rank == 0){
                printf("ERROR: value_type not known.\n");
                exit(1);
            }
        }
    }

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif

    // What taks place in this routine depends on "config.mode", i.e. the "result" in
    // "compute_result" is either a measure of performance, or an output vector y to validate
    if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
        compute_result<double, int>(&total_dp_mtx, &config, &r_dp, my_rank, comm_size);
    }
    else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
        compute_result<float, int>(&total_sp_mtx, &config, &r_sp, my_rank, comm_size);
    }
    else if (config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
        compute_result<_Float16, int>(&total_hp_mtx, &config, &r_hp, my_rank, comm_size);
#endif
    }

    double elapsed_main_time;

#ifdef USE_MPI
    elapsed_main_time = MPI_Wtime() - begin_main_time;
#else
    elapsed_main_time = getTimeStamp() - begin_main_time;
#endif

    if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]")
        r_dp.total_walltime = elapsed_main_time;
    else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]")
        r_sp.total_walltime = elapsed_main_time;
    else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
        r_hp.total_walltime = elapsed_main_time;
#endif
    }


#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif

    if(my_rank == 0){
        if(config.mode == 's'){
#ifdef USE_MKL
            if(config.validate_result){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Validating results.\n");}
#endif
                std::vector<double> mkl_result;
                if(config.equilibrate){
                    equilibrate_matrix<double, int>(&total_mtx);
                }
                validate_result(
                    &total_mtx, 
                    &config, 
                    &r_dp, 
                    &r_sp,
#ifdef HAVE_HALF_MATH
                    &r_hp,
#endif 
                    &mkl_result
                );
                
#ifdef DEBUG_MODE
                if(my_rank == 0){printf("Writing validation results to file.\n");}
#endif
                if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
                    write_result_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r_dp, &mkl_result, comm_size);
                }
                else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
                    write_result_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r_sp, &mkl_result, comm_size);
                }
                else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
                    write_result_to_file<_Float16, int>(&matrix_file_name, &(config.seg_method), &config, &r_hp, &mkl_result, comm_size);
#endif
                }
            }
#endif
        }
        else if(config.mode == 'b'){
#ifdef DEBUG_MODE
            if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
            if(config.value_type == "dp" || config.value_type == "ap[dp_sp]" || config.value_type == "ap[dp_hp]" || config.value_type == "ap[dp_sp_hp]"){
                write_bench_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r_dp, comm_size);
            }
            else if(config.value_type == "sp" || config.value_type == "ap[sp_hp]"){
                write_bench_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r_sp, comm_size);
            }
            else if(config.value_type == "hp"){
#ifdef HAVE_HALF_MATH
                write_bench_to_file<_Float16, int>(&matrix_file_name, &(config.seg_method), &config, &r_hp, comm_size);
#endif            
            }
        }
    }
}

int main(int argc, char *argv[]){

#ifdef DEBUG_MODE
    std::cout << "Using c++ version: " << __cplusplus << std::endl;
#endif

    // Bogus parallel region pin threads to cores
    dummy_pin();

    // Initialize just out of convenience
    int my_rank = 0, comm_size = 1;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Beginning of USpMV main execution.\n");}
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_INIT;
#endif

    bogus_init_pin();

    double begin_main_time;

#ifdef USE_MPI
    begin_main_time = MPI_Wtime();
#else
    begin_main_time = getTimeStamp();
#endif

    Config config;
    std::string matrix_file_name{};

    // Set defaults for cl inputs
    // TODO: Really should be done elsewhere
    std::string seg_method = "seg-rows";
    std::string kernel_format = "scs";
    std::string value_type = "dp";
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;

    parse_cli_inputs(argc, argv, &matrix_file_name, &seg_method, &kernel_format, &value_type, &config, my_rank);

    config.seg_method = seg_method;
    config.kernel_format = kernel_format;
    config.value_type = value_type;

    standalone_bench(config, matrix_file_name, my_rank, comm_size, begin_main_time);

#ifdef USE_MPI
    MPI_Finalize();
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("End of USpMV main execution.\n");}
#endif

    return 0;
}