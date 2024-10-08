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
// TODO
// #include <float.h>


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
    ScsData<double, IT> *hp_local_scs,
    ScsData<float, IT> *lp_local_scs,
    ContextData<IT> *local_context,
    const IT *work_sharing_arr,
    std::vector<VT> *local_y,
    std::vector<double> *hp_local_y,
    std::vector<float> *lp_local_y,
    std::vector<VT> *local_x,
    std::vector<double> *hp_local_x,
    std::vector<float> *lp_local_x,
    Result<VT, IT> *r,
    int my_rank,
    int comm_size)
{
    // Permute x, in order to match the permutation which was done to the columns
    std::vector<VT> local_x_permuted(local_x->size(), 0);
    std::vector<double> hp_local_x_permuted(hp_local_x->size(), 0);
    std::vector<float> lp_local_x_permuted(hp_local_x->size(), 0);

    apply_permutation<VT, IT>(&(local_x_permuted)[0], &(*local_x)[0], &(local_scs->new_to_old_idx)[0], local_scs->n_rows);

    if(config->value_type == "mp"){
        // Currently, we fix one sigma. That is, we permute lp and hp exactly the same
        apply_permutation<double, IT>(&(hp_local_x_permuted)[0], &(*hp_local_x)[0], &(hp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
        apply_permutation<float, IT>(&(lp_local_x_permuted)[0], &(*lp_local_x)[0], &(hp_local_scs->new_to_old_idx)[0], local_scs->n_rows);
    }

    OnePrecKernelArgs<VT, IT> *one_prec_kernel_args_encoded = new OnePrecKernelArgs<VT, IT>;
    TwoPrecKernelArgs<IT> *two_prec_kernel_args_encoded = new TwoPrecKernelArgs<IT>;
    void *comm_args_void_ptr;
    void *kernel_args_void_ptr;
    void *cusparse_args_void_ptr;
#ifdef USE_CUSPARSE
    CuSparseArgs *cusparse_args_encoded = new CuSparseArgs;
#endif
    CommArgs<VT, IT> *comm_args_encoded = new CommArgs<VT, IT>;

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
//     if(config->value_type == "mp"){
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

    double *d_x_hp = new double;
    double *d_y_hp = new double;
    ST *d_C_hp = new ST;
    ST *d_n_chunks_hp = new ST;
    IT *d_chunk_ptrs_hp = new IT;
    IT *d_chunk_lengths_hp = new IT;
    IT *d_col_idxs_hp = new IT;
    double *d_values_hp = new double;
    float *d_x_lp = new float;
    float *d_y_lp = new float;
    ST *d_C_lp = new ST;
    ST *d_n_chunks_lp = new ST;
    IT *d_chunk_ptrs_lp = new IT;
    IT *d_chunk_lengths_lp = new IT;
    IT *d_col_idxs_lp = new IT;
    float *d_values_lp = new float;

    if(config->value_type == "mp"){
        long n_scs_elements_hp = hp_local_scs->chunk_ptrs[hp_local_scs->n_chunks - 1]
                    + hp_local_scs->chunk_lengths[hp_local_scs->n_chunks - 1] * hp_local_scs->C;
        long n_scs_elements_lp = lp_local_scs->chunk_ptrs[lp_local_scs->n_chunks - 1]
                    + lp_local_scs->chunk_lengths[lp_local_scs->n_chunks - 1] * lp_local_scs->C;

        // TODO: temporary way to get around memory courruption problem
        for(int i = 0; i < n_scs_elements_hp; ++i){
            if(hp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad hp element %i found at idx %i\n", hp_local_scs->col_idxs[i], i);
#endif
                hp_local_scs->col_idxs[i] = 0;
            }
        }
        for(int i = 0; i < n_scs_elements_lp; ++i){
            if(lp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad lp %i element found at idx %i\n", lp_local_scs->col_idxs[i], i);
#endif
                lp_local_scs->col_idxs[i] = 0;
            }
        }


        // Allocate space for MP structs on device
        cudaMalloc(&d_values_hp, n_scs_elements_hp*sizeof(double));
        cudaMalloc(&d_values_lp, n_scs_elements_lp*sizeof(float));
        cudaMemcpy(d_values_hp, &(hp_local_scs->values)[0], n_scs_elements_hp*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values_lp, &(lp_local_scs->values)[0], n_scs_elements_lp*sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_C_hp, sizeof(long));
        cudaMalloc(&d_n_chunks_hp, sizeof(long));
        cudaMalloc(&d_chunk_ptrs_hp, (hp_local_scs->n_chunks + 1)*sizeof(int));
        cudaMalloc(&d_chunk_lengths_hp, hp_local_scs->n_chunks*sizeof(int));
        cudaMalloc(&d_col_idxs_hp, n_scs_elements_hp*sizeof(int));
        cudaMalloc(&d_C_lp, sizeof(long));
        cudaMalloc(&d_n_chunks_lp, sizeof(long));
        cudaMalloc(&d_chunk_ptrs_lp, (lp_local_scs->n_chunks + 1)*sizeof(int));
        cudaMalloc(&d_chunk_lengths_lp, lp_local_scs->n_chunks*sizeof(int));
        cudaMalloc(&d_col_idxs_lp, n_scs_elements_lp*sizeof(int));

        // Copy matrix data to device
        cudaMemcpy(d_chunk_ptrs_hp, &(hp_local_scs->chunk_ptrs)[0], (hp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_lengths_hp, &(hp_local_scs->chunk_lengths)[0], hp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idxs_hp, &(hp_local_scs->col_idxs)[0], n_scs_elements_hp*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_hp, &hp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_chunks_hp, &hp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice);

        cudaMemcpy(d_chunk_ptrs_lp, &(lp_local_scs->chunk_ptrs)[0], (lp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_lengths_lp, &(lp_local_scs->chunk_lengths)[0], lp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idxs_lp, &(lp_local_scs->col_idxs)[0], n_scs_elements_lp*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_lp, &lp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_chunks_lp, &lp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice);

        // Copy x and y data to device
        double *local_x_hp_hardcopy = new double[local_scs->n_rows_padded];
        double *local_y_hp_hardcopy = new double[local_scs->n_rows_padded];
        float *local_x_lp_hardcopy = new float[local_scs->n_rows_padded];
        float *local_y_lp_hardcopy = new float[local_scs->n_rows_padded];

        #pragma omp parallel for
        for(int i = 0; i < local_scs->n_rows_padded; ++i){
            local_x_hp_hardcopy[i] = hp_local_x_permuted[i];
            local_y_hp_hardcopy[i] = (*hp_local_y)[i];
            local_x_lp_hardcopy[i] = lp_local_x_permuted[i];
            local_y_lp_hardcopy[i] = (*lp_local_y)[i];
        }

        cudaMalloc(&d_x_hp, local_scs->n_rows_padded*sizeof(double));
        cudaMalloc(&d_y_hp, local_scs->n_rows_padded*sizeof(double));
        cudaMalloc(&d_x_lp, local_scs->n_rows_padded*sizeof(float));
        cudaMalloc(&d_y_lp, local_scs->n_rows_padded*sizeof(float));

        cudaMemcpy(d_x_hp, local_x_hp_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_hp, local_y_hp_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_lp, local_x_lp_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_lp, local_y_lp_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice);

        delete local_x_hp_hardcopy;
        delete local_y_hp_hardcopy;
        delete local_x_lp_hardcopy;
        delete local_y_lp_hardcopy;

        // Pack pointers into struct 
        // TODO: allow for each struct to have it's own C
        two_prec_kernel_args_encoded->hp_C =             d_C_hp;
        two_prec_kernel_args_encoded->hp_n_chunks =      d_n_chunks_hp; //shared for now
        two_prec_kernel_args_encoded->hp_chunk_ptrs =    d_chunk_ptrs_hp;
        two_prec_kernel_args_encoded->hp_chunk_lengths = d_chunk_lengths_hp;
        two_prec_kernel_args_encoded->hp_col_idxs =      d_col_idxs_hp;
        two_prec_kernel_args_encoded->hp_values =        d_values_hp;
        two_prec_kernel_args_encoded->hp_local_x =       d_x_hp;
        two_prec_kernel_args_encoded->hp_local_y =       d_y_hp;
        two_prec_kernel_args_encoded->lp_C =             d_C_hp; // shared maybe don't?
        two_prec_kernel_args_encoded->lp_n_chunks =      d_n_chunks_hp; //shared for now
        two_prec_kernel_args_encoded->lp_chunk_ptrs =    d_chunk_ptrs_lp;
        two_prec_kernel_args_encoded->lp_chunk_lengths = d_chunk_lengths_lp;
        two_prec_kernel_args_encoded->lp_col_idxs =      d_col_idxs_lp;
        two_prec_kernel_args_encoded->lp_values =        d_values_lp;
        two_prec_kernel_args_encoded->lp_local_x =       d_x_lp;
        two_prec_kernel_args_encoded->lp_local_y =       d_y_lp;
        two_prec_kernel_args_encoded->n_blocks =         &n_blocks;
        kernel_args_void_ptr = (void*) two_prec_kernel_args_encoded;

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
        else{
            printf("CuSparse CRS only enabled with DP or SP\n");
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
        else{
            printf("CuSparse SELL-P only enabled with DP or SP\n");
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
    cusparse_args_encoded->alg = CUSPARSE_SPMV_ALG_DEFAULT;
    cusparse_args_encoded->externalBuffer = dBuffer;
    cusparse_args_void_ptr = (void*) cusparse_args_encoded;
#endif
#else
    if(config->value_type == "mp"){
        // Encode kernel args into struct
        
        // TODO: allow for each struct to have it's own C
        two_prec_kernel_args_encoded->hp_C = &hp_local_scs->C;
        two_prec_kernel_args_encoded->hp_n_chunks = &hp_local_scs->n_chunks; //shared for now
        two_prec_kernel_args_encoded->hp_chunk_ptrs = hp_local_scs->chunk_ptrs.data();
        two_prec_kernel_args_encoded->hp_chunk_lengths = hp_local_scs->chunk_lengths.data();
        two_prec_kernel_args_encoded->hp_col_idxs = hp_local_scs->col_idxs.data();
        two_prec_kernel_args_encoded->hp_values = hp_local_scs->values.data();
        two_prec_kernel_args_encoded->hp_local_x = &(hp_local_x_permuted)[0];
        two_prec_kernel_args_encoded->hp_local_y = &(*hp_local_y)[0];
        two_prec_kernel_args_encoded->lp_C = &lp_local_scs->C;
        two_prec_kernel_args_encoded->lp_n_chunks = &hp_local_scs->n_chunks; //shared for now
        two_prec_kernel_args_encoded->lp_chunk_ptrs = lp_local_scs->chunk_ptrs.data();
        two_prec_kernel_args_encoded->lp_chunk_lengths = lp_local_scs->chunk_lengths.data();
        two_prec_kernel_args_encoded->lp_col_idxs = lp_local_scs->col_idxs.data();
        two_prec_kernel_args_encoded->lp_values = lp_local_scs->values.data();
        two_prec_kernel_args_encoded->lp_local_x = &(lp_local_x_permuted)[0];
        two_prec_kernel_args_encoded->lp_local_y = &(*lp_local_y)[0];
        kernel_args_void_ptr = (void*) two_prec_kernel_args_encoded;
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
            // TODO: Is this #if-else correct with mpi? I don't think it is.
            if(config->value_type == "mp"){
                spmv_kernel.execute_warmup_mp_spmv();
                // spmv_kernel.swap_local_mp_vectors();
            }
            else{
                spmv_kernel.execute_warmup_spmv();
                // spmv_kernel.swap_local_vectors();
            }

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
                    if(config->value_type == "mp"){
                        spmv_kernel.execute_mp_spmv();
                        // spmv_kernel.swap_local_mp_vectors();
                    }
                    else{
                        spmv_kernel.execute_spmv();
                        // spmv_kernel.swap_local_vectors();
                    }
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
    else if(config->mode == 's'){ // Enter main COMM-SPMV-SWAP loop, solve mode
        // Selects the first (n_rows)-many elements of a sorted y vector, and chops off padding
        std::vector<VT> sorted_local_y(local_y->size(), 0);
        std::vector<double> sorted_hp_local_y(local_y->size(), 0);

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
            if(config->value_type == "mp"){
                spmv_kernel.execute_mp_spmv();
            }
            else{
                spmv_kernel.execute_spmv();
            }
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
            if(config->value_type == "mp"){
                spmv_kernel.swap_local_mp_vectors();    
            }
            else{
                spmv_kernel.swap_local_vectors();
            }
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
        if(config->value_type == "mp"){
            cudaMemcpy(&(*hp_local_x)[0], spmv_kernel.hp_local_x, local_scs->n_rows_padded*sizeof(double), cudaMemcpyDeviceToHost);
        }
        else if(config->value_type == "dp"){
            cudaMemcpy(&(*local_x)[0], spmv_kernel.local_x, local_scs->n_rows_padded*sizeof(double), cudaMemcpyDeviceToHost);
        }
        else if(config->value_type == "sp"){
            cudaMemcpy(&(*local_x)[0], spmv_kernel.local_x, local_scs->n_rows_padded*sizeof(float), cudaMemcpyDeviceToHost);
        }
#else
        if(config->value_type == "mp"){
            for(int i = 0; i < local_scs->n_rows_padded; ++i){
                (*hp_local_x)[i] = (spmv_kernel.hp_local_x)[i];
            }
        }
        else{
            for(int i = 0; i < local_scs->n_rows_padded; ++i){
                (*local_x)[i] = (spmv_kernel.local_x)[i];
            }
        }
#endif

        // TODO: Clean up!
        if(config->value_type == "mp"){
            
            apply_permutation<double, IT>(&(sorted_hp_local_y)[0], &(*hp_local_x)[0], &(hp_local_scs->old_to_new_idx)[0], hp_local_scs->n_rows);
        
            // Give result to local_y for results output
            for(int i = 0; i < local_y->size(); ++i){
                (*local_y)[i] = (sorted_hp_local_y)[i];
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

    r->value_type_str = typeid(VT).name();
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

    // Only relevant for mp
    r->hp_nnz = hp_local_scs->nnz;
    r->lp_nnz = lp_local_scs->nnz;

    if(hp_local_scs->n_elements == 0){
        r->hp_beta = 0;
    }
    else{
        r->hp_beta = (double)hp_local_scs->nnz / hp_local_scs->n_elements;
    }
        
    
    if(lp_local_scs->n_elements == 0){
        r->lp_beta = 0;
    }
    else{
        r->lp_beta = (double)lp_local_scs->nnz / lp_local_scs->n_elements;
    }
    
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
//     if(config->value_type == "mp"){
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
    delete two_prec_kernel_args_encoded;
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
#ifdef USE_MPI

        MPI_Gather(&(r->perf_gflops),
                1,
                MPI_DOUBLE,
                perfs_from_procs_arr,
                1,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

        MPI_Gather(&(r->hp_nnz),
                1,
                MPI_UNSIGNED_LONG,
                hp_nnz_per_procs_arr,
                1,
                MPI_UNSIGNED_LONG,
                0,
                MPI_COMM_WORLD);

        MPI_Gather(&(r->lp_nnz),
                1,
                MPI_UNSIGNED_LONG,
                lp_nnz_per_procs_arr,
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

        r->cumulative_hp_nnz = r->hp_nnz;
        r->cumulative_lp_nnz = r->lp_nnz;

        r->total_hp_percent = (r->cumulative_hp_nnz / (double)r->total_nnz) * 100.0;
        r->total_lp_percent = (r->cumulative_lp_nnz / (double)r->total_nnz) * 100.0;
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

            if (typeid(VT) == typeid(double)){
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
                MPI_Gatherv(local_x_copy[0],
                            num_local_rows,
                            MPI_DOUBLE,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_DOUBLE,
                            0,
                            MPI_COMM_WORLD);
            }
            else if (typeid(VT) == typeid(float)){
                MPI_Gatherv(&(r->y_out)[0],
                            num_local_rows,
                            MPI_FLOAT,
                            &total_uspmv_result[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);

                MPI_Gatherv(local_x_copy[0],
                            num_local_rows,
                            MPI_FLOAT,
                            &total_x[0],
                            counts_arr,
                            displ_arr_bk,
                            MPI_FLOAT,
                            0,
                            MPI_COMM_WORLD);
            }

            // If we're verifying results, assign total vectors to Result object
            // NOTE: Garbage values for all but root process
            r->total_x = total_x;
            r->total_uspmv_result = total_uspmv_result;
#else
            r->total_x = (*local_x_copy);
            r->total_uspmv_result = r->y_out;
#endif
        }
    }
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
    ScsData<double, IT> hp_local_scs;
    ScsData<float, IT> lp_local_scs;

    ContextData<IT> local_context;

    // Allocate space for work sharing array
    IT work_sharing_arr[comm_size + 1];
    work_sharing_arr[0] = 0; // Initialize first element, since it's used always

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
        &hp_local_scs,
        &lp_local_scs,
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
    SimpleDenseMatrix<double, IT> hp_local_x(&local_context);
    SimpleDenseMatrix<float, IT> lp_local_x(&local_context);

    SimpleDenseMatrix<VT, IT> local_y(&local_context);

    // NOTE: a low precision y vector is needed for swapping with low precision x
    SimpleDenseMatrix<double, IT> hp_local_y(&local_context);
    SimpleDenseMatrix<float, IT> lp_local_y(&local_context);

    // Initialize local_x and y, either randomly, with default values defined in classes_structs.hpp,
    // or with 1s (by default)
    local_x.init(config, 'x');
    local_y.init(config, 'y');

    // TODO: wrap in method or something
    if(config->value_type == "mp"){
        for(int i = 0; i < (local_x.vec).size(); ++i){
            hp_local_x.vec[i] = static_cast<double>(local_x.vec[i]);
            lp_local_x.vec[i] = static_cast<float>(local_x.vec[i]);
        }
    }

    // Copy contents of local_x for output, and validation against mkl
    std::vector<VT> local_x_copy = local_x.vec;


#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter bench_spmv.\n");}
#endif
    bench_spmv<VT, IT>(
        config,
        &local_scs,
        &hp_local_scs,
        &lp_local_scs,
        &local_context,
        work_sharing_arr,
        &local_y.vec,
        &hp_local_y.vec,
        &lp_local_y.vec,
        &local_x.vec,
        &hp_local_x.vec,
        &lp_local_x.vec,
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

int main(int argc, char *argv[]){

    // _Float16 a = 1.0;

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

    double begin_main_time;

#ifdef USE_MPI
    begin_main_time = MPI_Wtime();
#else
    begin_main_time = getTimeStamp();
#endif

    Config config;
    std::string matrix_file_name{};

    // Set defaults for cl inputs
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

    if (config.value_type == "dp")
    {
        MtxData<double, int> total_mtx;
        Result<double, int> r;

        if(my_rank == 0){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
            read_mtx<double, int>(matrix_file_name, config, &total_mtx, my_rank);
            r.total_nnz = total_mtx.nnz;
            r.total_rows = total_mtx.n_rows;
        }
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif
        compute_result<double, int>(&total_mtx, &config, &r, my_rank, comm_size);

#ifdef USE_MPI
        r.total_walltime = MPI_Wtime() - begin_main_time;
#else
        r.total_walltime = getTimeStamp() - begin_main_time;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif
        if(my_rank == 0){
            if(config.mode == 's'){
                if(config.validate_result){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Validating results.\n");}
#endif
                    std::vector<double> mkl_dp_result;
                    if(config.equilibrate){
                        equilibrate_matrix<double, int>(&total_mtx);
                        // std::vector<double> largest_elems(total_mtx.n_cols);
                        // extract_largest_elems<double, int>(&total_mtx, &largest_elems);
                        // scale_w_jacobi<double, int>(&total_mtx, &largest_elems);
                    }
                    validate_dp_result(&total_mtx, &config, &r, &mkl_dp_result);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing validation results to file.\n");}
#endif
                    write_result_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r, &mkl_dp_result, comm_size);
                }
                else{
                }
            }
            else if(config.mode == 'b'){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
                write_bench_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r, comm_size);
            }
        }
    }
    else if (config.value_type == "sp")
    {
        MtxData<float, int> total_mtx;
        Result<float, int> r;

        if(my_rank == 0){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
            read_mtx<float, int>(matrix_file_name, config, &total_mtx, my_rank);
            r.total_nnz = total_mtx.nnz;
            r.total_rows = total_mtx.n_rows;
        }
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif
        compute_result<float, int>(&total_mtx, &config, &r, my_rank, comm_size);

#ifdef USE_MPI
        r.total_walltime = MPI_Wtime() - begin_main_time;
#else
        r.total_walltime = getTimeStamp() - begin_main_time;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif

        if(my_rank == 0){
            if(config.mode == 's'){
                if(config.validate_result){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Validating results.\n");}
#endif
                    std::vector<float> mkl_sp_result;
                    if(config.equilibrate){
                        equilibrate_matrix<float, int>(&total_mtx);
                        // std::vector<float> largest_elems(total_mtx.n_cols);
                        // extract_largest_elems<float, int>(&total_mtx, &largest_elems);
                        // scale_w_jacobi<float, int>(&total_mtx, &largest_elems);
                    }
                    validate_sp_result(&total_mtx, &config, &r, &mkl_sp_result);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing validation results to file.\n");}
#endif
                    write_result_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r, &mkl_sp_result, comm_size);
                }
                else{
                }
            }
            else if(config.mode == 'b'){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
                write_bench_to_file<float, int>(&matrix_file_name, &(config.seg_method), &config, &r, comm_size);
            }
        }
    }
    else if (config.value_type == "mp")
    // Currently, everything is still read and results are written as doubles.
    // i.e. VT = double, IT = int.
    {
        MtxData<double, int> total_mtx;
        Result<double, int> r;

        if(my_rank == 0){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Reading mtx file.\n");}
#endif
            read_mtx<double, int>(matrix_file_name, config, &total_mtx, my_rank);
            r.total_nnz = total_mtx.nnz;
            r.total_rows = total_mtx.n_rows;
        }
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Enter compute_result.\n");}
#endif
        compute_result<double, int>(&total_mtx, &config, &r, my_rank, comm_size);

#ifdef USE_MPI
        r.total_walltime = MPI_Wtime() - begin_main_time;
#else
        r.total_walltime = getTimeStamp() - begin_main_time;
#endif

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Complete compute_result.\n");}
#endif

        if(my_rank == 0){
            if(config.mode == 's'){
                if(config.validate_result){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Validating results.\n");}
#endif
                    std::vector<double> mkl_dp_result;
                    if(config.equilibrate){
                        equilibrate_matrix<double, int>(&total_mtx);
                        // std::vector<double> largest_elems(total_mtx.n_cols);
                        // extract_largest_elems<double, int>(&total_mtx, &largest_elems);
                        // scale_w_jacobi<double, int>(&total_mtx, &largest_elems);
                    }
                    validate_dp_result(&total_mtx, &config, &r, &mkl_dp_result);
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing validation results to file.\n");}
#endif
                    // Validate against doubles, but it would be nice to validate against both dp and sp.
                    write_result_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r, &mkl_dp_result, comm_size);
                }
                else{
                }
            }
            else if(config.mode == 'b'){
#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Writing benchmark results to file.\n");}
#endif
                write_bench_to_file<double, int>(&matrix_file_name, &(config.seg_method), &config, &r, comm_size);
            }
        }
    }

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