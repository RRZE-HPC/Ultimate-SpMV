#ifndef CLASSES_STRUCTS
#define CLASSES_STRUCTS

#include "vectors.h"
#include "mmio.h"
#include "kernels.hpp"
#include <functional> 
#include <ctime>

#ifdef USE_CUSPARSE
#include <cusparse.h> 
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_METIS
#include <metis.h>
#endif

using ST = long;

// Initialize all matrices and vectors the same.
// Use -rand to initialize randomly.
static bool g_same_seed_for_every_vector = true;

struct Config
{
    long n_els_per_row{-1}; // ell
    long chunk_size{1};    // sell-c-sigma
    long sigma{1};         // sell-c-sigma

    // Initialize rhs vector with random numbers.
    char random_init_x = '0';

    // Scale elements in row by diagonal element, useful for matrices with high condition number
    int jacobi_scale = 0;

    // Override values of the matrix, read via mtx file, with random numbers.
    bool random_init_A{false};

    // Convenient way to pass data to x-vector init func...
    // TODO: probably not the cleanest way 
    double matrix_max = 1.0;
    double matrix_mean = 1.0;
    double matrix_min = 1.0;

    // No. of repetitions to perform.
    unsigned long n_repetitions{1};

    // Verify result of SpVM.
    int validate_result = 1;

    // Verify result against solution of COO kernel.
    int verify_result_with_coo = 0;

    // Sort rows/columns of sparse matrix before
    // converting it to a specific format.
    int sort_matrix = 1;

    int verbose_validation = 0;

    // activate profile logs, only root process
    int log_prof = 0;

    // communicate the halo elements in benchmark loop
    int comm_halos = 1;

    // synchronize with barriers each benchmark loop
    int ba_synch = 1; // To eliminate unwanted overlapping benefits. Make 0 if benching mpk.

    // Pack contiguous elements for MPI_Isend in parallel
    int par_pack = 0;

    // Scale rows and column of matrix
    int equilibrate = 0;

    // Configures if the code will be executed in bench mode (b) or solve mode (s)
    char mode = 'b'; 

    // Runs benchmark for a specified number of seconds
    double bench_time = 10.0;
    // double bench_time = 1.0;

    // Mixed Precision bucket size, used for partitioning matrix
    double ap_threshold_1 = 0.0;

    // Mixed Precision bucket size, used for partitioning matrix
    double ap_threshold_2 = 0.0;

    int dropout = 0;

    double dropout_threshold = 0.0;

    // Just to make it easier to report kernel launch configuration at the end
#ifdef __CUDACC__
    long num_blocks = 0;
    long tpb = 0;
#endif

    // Default matrix segmentation method
    std::string seg_method = "seg-rows";

    // Default matrix data value type
    std::string value_type = "dp";

    // Selects the default matrix storage format
    std::string kernel_format = "scs"; 

    // TODO: consolidate?
    // filename for single precision results printing
    std::string output_filename_sp = "spmv_mkl_compare_sp.txt";

    // filename for double precision results printing
    std::string output_filename_dp = "spmv_mkl_compare_dp.txt";

    // filename for double precision results printing
    std::string output_filename_hp = "spmv_mkl_compare_hp.txt";

    // filename for mixed precision results printing
    std::string output_filename_ap = "spmv_mkl_compare_ap.txt";

    // filename for benchmark results printing
    std::string output_filename_bench = "spmv_bench.txt";

};

template <typename IT>
struct ContextData
{
    std::vector<IT> non_zero_senders;
    std::vector<IT> non_zero_receivers;

    std::vector<std::vector<IT>> send_tags;
    std::vector<std::vector<IT>> recv_tags;

    // TODO: probably more performant to do calculations earlier, and not store here
    std::vector<std::vector<IT>> comm_send_idxs;
    std::vector<std::vector<IT>> comm_recv_idxs;

    std::vector<IT> recv_counts_cumsum;
    std::vector<IT> send_counts_cumsum;

    IT num_local_rows;
    IT scs_padding;
    IT total_nnz;
};

template <typename VT, typename IT>
struct CommArgs
{
#ifdef USE_MPI
    Config *config;
    ContextData<IT> *local_context;
    const IT *perm;
    VT **to_send_elems;
    const IT *work_sharing_arr;
    MPI_Request *recv_requests;
    const IT *nzs_size;
    MPI_Request *send_requests;
    const IT *nzr_size;
    const IT *num_local_elems;
#endif
    const IT *my_rank;
    const IT* comm_size;

};

template <typename VT, typename IT>
struct OnePrecKernelArgs
{
    ST * C;
    ST * n_chunks;
    IT * RESTRICT chunk_ptrs;
    IT * RESTRICT chunk_lengths;
    IT * RESTRICT col_idxs;
    VT * RESTRICT values;
    VT * RESTRICT local_x;
    VT * RESTRICT local_y;
#ifdef __CUDACC__
    ST * n_blocks;
#endif
};

template <typename IT>
struct MultiPrecKernelArgs
{
    ST * dp_C;
    ST * dp_n_chunks; // (same for both)
    IT * RESTRICT dp_chunk_ptrs;
    IT * RESTRICT dp_chunk_lengths;
    IT * RESTRICT dp_col_idxs;
    double * RESTRICT dp_values;
    double * RESTRICT dp_local_x;
    double * RESTRICT dp_local_y;
    ST * sp_C;
    ST * sp_n_chunks; // (same for both)
    IT * RESTRICT sp_chunk_ptrs;
    IT * RESTRICT sp_chunk_lengths;
    IT * RESTRICT sp_col_idxs;
    float * RESTRICT sp_values;
    float * RESTRICT sp_local_x;
    float * RESTRICT sp_local_y;
#ifdef HAVE_HALF_MATH
    ST * hp_C;
    ST * hp_n_chunks; // (same for both)
    IT * RESTRICT hp_chunk_ptrs;
    IT * RESTRICT hp_chunk_lengths;
    IT * RESTRICT hp_col_idxs;
    _Float16 * RESTRICT hp_values;
    _Float16 * RESTRICT hp_local_x;
    _Float16 * RESTRICT hp_local_y;
#endif
#ifdef __CUDACC__
    ST * n_blocks;
#endif
};

#ifdef USE_CUSPARSE
// template <typename VT, typename IT> Dont need templating?
struct CuSparseArgs
{
    cusparseHandle_t          handle;
    cusparseOperation_t       opA;
    const void*               alpha;
    cusparseConstSpMatDescr_t matA;  // non-const descriptor supported
    cusparseConstDnVecDescr_t vecX;  // non-const descriptor supported
    const void*               beta;
    cusparseDnVecDescr_t      vecY;
    cudaDataType              computeType;
    cusparseSpMVAlg_t         alg;
    void*                     externalBuffer;
};
#endif

// VTU and VTL only used for ap kernels
template <typename VT, typename IT>
class SpmvKernel {
    private:
        typedef std::function<void(
            const ST *, // C
            const ST *, // n_chunks
            const IT *, // chunk_ptrs
            const IT *, // chunk_lengths
            const IT *, // col_idxs
            const VT *, // values
            VT *, // x
            VT *, //y
#ifdef __CUDACC__
            const ST *, // n_blocks
#endif
            const int * // my_rank
        )> OnePrecFuncPtr;

        typedef std::function<void(
            const ST *, // up_C
            const ST *, // up_n_chunks // TODO same, for now.
            const IT * RESTRICT, // up_chunk_ptrs
            const IT * RESTRICT, // up_chunk_lengths
            const IT * RESTRICT, // up_col_idxs
            const double * RESTRICT, // up_values
            double * RESTRICT, // up_x
            double * RESTRICT, // up_y
            const ST *, // lp_C
            const ST *, // lp_n_chunks // TODO same, for now.
            const IT * RESTRICT, // lp_chunk_ptrs
            const IT * RESTRICT, // lp_chunk_lengths
            const IT * RESTRICT, // lp_col_idxs
            const float * RESTRICT, // lp_values
            float * RESTRICT, // lp_x
            float * RESTRICT, // lp_y
#ifdef HAVE_HALF_MATH
            const ST *, // lp_C
            const ST *, // lp_n_chunks // TODO same, for now.
            const IT * RESTRICT, // lp_chunk_ptrs
            const IT * RESTRICT, // lp_chunk_lengths
            const IT * RESTRICT, // lp_col_idxs
            const _Float16 * RESTRICT, // lp_values
            _Float16 * RESTRICT, // lp_x
            _Float16 * RESTRICT, // lp_y
#endif
#ifdef __CUDACC__
            const ST *, // n_blocks
#endif
            const int *
        )> MultiPrecFuncPtr;

        OnePrecFuncPtr one_prec_kernel_func_ptr;
        OnePrecFuncPtr one_prec_warmup_kernel_func_ptr;
        MultiPrecFuncPtr multi_prec_kernel_func_ptr;
        MultiPrecFuncPtr multi_prec_warmup_kernel_func_ptr;

        void *kernel_args_encoded;
        void *comm_args_encoded;
        void *cusparse_args_encoded;
        Config *config;

        // Decode kernel args
        OnePrecKernelArgs<VT, IT> *one_prec_kernel_args_decoded = (OnePrecKernelArgs<VT, IT>*) kernel_args_encoded;
        MultiPrecKernelArgs<IT> *multi_prec_kernel_args_decoded = (MultiPrecKernelArgs<IT>*) kernel_args_encoded;
#ifdef USE_CUSPARSE
        CuSparseArgs *cusparse_args_decoded = (CuSparseArgs*) cusparse_args_encoded;
#endif

        const ST * C = one_prec_kernel_args_decoded->C;
        const ST * n_chunks = one_prec_kernel_args_decoded->n_chunks;
        const IT * RESTRICT chunk_ptrs = one_prec_kernel_args_decoded->chunk_ptrs;
        const IT * RESTRICT chunk_lengths = one_prec_kernel_args_decoded->chunk_lengths;
        const IT * RESTRICT col_idxs = one_prec_kernel_args_decoded->col_idxs;
        const VT * RESTRICT values = one_prec_kernel_args_decoded->values;
#ifdef __CUDACC__
        const ST * n_blocks_1 = one_prec_kernel_args_decoded->n_blocks;
#endif

        // Need different names on all of unpacked args
        const ST * dp_C                         = multi_prec_kernel_args_decoded->dp_C;
        const ST * dp_n_chunks                  = multi_prec_kernel_args_decoded->dp_n_chunks; // TODO same, for now.
        const IT * RESTRICT dp_chunk_ptrs       = multi_prec_kernel_args_decoded->dp_chunk_ptrs;
        const IT * RESTRICT dp_chunk_lengths    = multi_prec_kernel_args_decoded->dp_chunk_lengths;
        const IT * RESTRICT dp_col_idxs         = multi_prec_kernel_args_decoded->dp_col_idxs;
        const double * RESTRICT dp_values       = multi_prec_kernel_args_decoded->dp_values;
        const ST * sp_C                         = multi_prec_kernel_args_decoded->sp_C;
        const ST * sp_n_chunks                  = multi_prec_kernel_args_decoded->sp_n_chunks; // TODO same, for now.
        const IT * RESTRICT sp_chunk_ptrs       = multi_prec_kernel_args_decoded->sp_chunk_ptrs;
        const IT * RESTRICT sp_chunk_lengths    = multi_prec_kernel_args_decoded->sp_chunk_lengths;
        const IT * RESTRICT sp_col_idxs         = multi_prec_kernel_args_decoded->sp_col_idxs;
        const float * RESTRICT sp_values        = multi_prec_kernel_args_decoded->sp_values;
#ifdef HAVE_HALF_MATH
        const ST * hp_C                         = multi_prec_kernel_args_decoded->hp_C;
        const ST * hp_n_chunks                  = multi_prec_kernel_args_decoded->hp_n_chunks; // TODO same, for now.
        const IT * RESTRICT hp_chunk_ptrs       = multi_prec_kernel_args_decoded->hp_chunk_ptrs;
        const IT * RESTRICT hp_chunk_lengths    = multi_prec_kernel_args_decoded->hp_chunk_lengths;
        const IT * RESTRICT hp_col_idxs         = multi_prec_kernel_args_decoded->hp_col_idxs;
        const _Float16 * RESTRICT hp_values     = multi_prec_kernel_args_decoded->hp_values;
#endif

#ifdef __CUDACC__
        const ST * n_blocks_2 = multi_prec_kernel_args_decoded->n_blocks;
#endif

#ifdef USE_CUSPARSE
        cusparseHandle_t          handle = cusparse_args_decoded->handle;
        cusparseOperation_t       opA = cusparse_args_decoded->opA;
        const void*               alpha = cusparse_args_decoded->alpha;
        cusparseConstSpMatDescr_t matA = cusparse_args_decoded->matA;
        cusparseConstDnVecDescr_t vecX = cusparse_args_decoded->vecX;
        const void*               beta = cusparse_args_decoded->beta;
        cusparseDnVecDescr_t      vecY = cusparse_args_decoded->vecY;
        cudaDataType              computeType = cusparse_args_decoded->computeType;
        cusparseSpMVAlg_t         alg = cusparse_args_decoded->alg;
        void*                     externalBuffer = cusparse_args_decoded->externalBuffer;
#endif

        // Decode comm args
        CommArgs<VT, IT> *comm_args_decoded = (CommArgs<VT, IT>*) comm_args_encoded;
#ifdef USE_MPI
        ContextData<IT> *local_context = comm_args_decoded->local_context;
        const IT *perm = comm_args_decoded->perm;
        VT **to_send_elems = comm_args_decoded->to_send_elems;
        const IT *work_sharing_arr = comm_args_decoded->work_sharing_arr;
        MPI_Request *recv_requests = comm_args_decoded->recv_requests;
        const IT nzs_size = *(comm_args_decoded->nzs_size);
        MPI_Request *send_requests = comm_args_decoded->send_requests;
        const IT nzr_size = *(comm_args_decoded->nzr_size);
        const IT num_local_elems = *(comm_args_decoded->num_local_elems);
#endif
        const IT my_rank = *(comm_args_decoded->my_rank);
        const IT comm_size = *(comm_args_decoded->comm_size);


    public:
        VT * RESTRICT local_x = one_prec_kernel_args_decoded->local_x; // NOTE: cannot be constant, changed by comm routine
        VT * RESTRICT local_y = one_prec_kernel_args_decoded->local_y; // NOTE: cannot be constant, changed by comp routine
        double * RESTRICT dp_local_x = multi_prec_kernel_args_decoded->dp_local_x;
        double * RESTRICT dp_local_y = multi_prec_kernel_args_decoded->dp_local_y;
        float * RESTRICT sp_local_x = multi_prec_kernel_args_decoded->sp_local_x;
        float * RESTRICT sp_local_y = multi_prec_kernel_args_decoded->sp_local_y;
#ifdef HAVE_HALF_MATH
        _Float16 * RESTRICT hp_local_x = multi_prec_kernel_args_decoded->hp_local_x;
        _Float16 * RESTRICT hp_local_y = multi_prec_kernel_args_decoded->hp_local_y;
#endif
        

        SpmvKernel(
            Config *config_, 
            void *kernel_args_encoded_,
            void *cusparse_args_encoded_,
            void *comm_args_encoded_
        ): 
        config(config_), 
        kernel_args_encoded(kernel_args_encoded_), 
        cusparse_args_encoded(cusparse_args_encoded_),
        comm_args_encoded(comm_args_encoded_){
                                            // CRS kernel selection //
            if (config->kernel_format == "crs" || config->kernel_format == "csr"){
                if (config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){
#ifdef USE_CUSPARSE
                    if(my_rank == 0){printf("CUSPARSE CRS kernel selected\n");}
#else
                    if(my_rank == 0){printf("CRS kernel selected\n");}
#endif
#ifdef __CUDACC__
                    one_prec_kernel_func_ptr = spmv_gpu_csr_launcher<VT, IT>;
                    one_prec_warmup_kernel_func_ptr = spmv_gpu_csr_launcher<VT, IT>;
#else
                    if(my_rank == 0){printf("CRS kernel selected\n");}
                    one_prec_kernel_func_ptr = spmv_omp_csr<VT, IT>;
                    one_prec_warmup_kernel_func_ptr = spmv_warmup_omp_csr<VT, IT>;
                    // one_prec_kernel_func_ptr = spmv_avx512_float16<VT, IT>;
                    // one_prec_warmup_kernel_func_ptr = spmv_avx512_float16<VT, IT>;
                    // one_prec_kernel_func_ptr = spmv_avx256_float16<VT, IT>;
                    // one_prec_warmup_kernel_func_ptr = spmv_avx256_float16<VT, IT>;
#endif
                }
                else if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
#ifdef __CUDACC__
                    // TODO: Finish AP support for GPUs
                    multi_prec_kernel_func_ptr = spmv_gpu_ap_csr_launcher<IT>;
                    multi_prec_warmup_kernel_func_ptr = spmv_gpu_ap_csr_launcher<IT>;
#else
                    if(config->value_type == "ap[dp_sp]"){
                        if(my_rank == 0){printf("ap[dp_sp] CRS kernel selected\n");}
                        multi_prec_kernel_func_ptr = spmv_omp_csr_apdpsp<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdpsp<IT>;
                    }
                    else if (config->value_type == "ap[dp_hp]"){
                        if(my_rank == 0){printf("ap[dp_hp] CRS kernel selected\n");}
#ifdef HAVE_HALF_MATH
                        multi_prec_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
#else
                        if(my_rank == 0){printf("HAVE_HALF_MATH undefined\n"); exit(1);}
#endif
                    }
                    else if (config->value_type == "ap[sp_hp]"){
                        if(my_rank == 0){printf("ap[sp_hp] CRS kernel selected\n");}
#ifdef HAVE_HALF_MATH
                        multi_prec_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
#else
                        if(my_rank == 0){printf("HAVE_HALF_MATH undefined\n"); exit(1);}
#endif
                    }
                    else if (config->value_type == "ap[dp_sp_hp]"){
                        if(my_rank == 0){printf("ap[dp_sp_hp] CRS kernel selected\n");}
#ifdef HAVE_HALF_MATH
                        multi_prec_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
#else
                        if(my_rank == 0){printf("HAVE_HALF_MATH undefined\n"); exit(1);}
#endif
                    }
#endif
                }
            }
                                // SCS kernel selection //
            else if (config->kernel_format == "scs"){
                // If C is a typical choice, use ADV kernels
                if(
                    config->chunk_size == 1
                    && config->chunk_size == 2 
                    && config->chunk_size == 4
                    && config->chunk_size == 8
                    && config->chunk_size == 16
                    && config->chunk_size == 32
                    && config->chunk_size == 64
                    && config->chunk_size == 128
                    && config->chunk_size == 256)
                {
#ifdef USE_CUSPARSE
                    if(my_rank == 0){printf("CUSPARSE SCS kernel selected\n");}
#else
                    if(my_rank == 0){printf("C = %i => Advanced SCS kernel selected\n", config->chunk_size);}
#endif
                    if (config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){

#ifdef __CUDACC__
                        one_prec_kernel_func_ptr = spmv_gpu_scs_adv_launcher<VT, IT>;
                        one_prec_warmup_kernel_func_ptr = spmv_gpu_scs_adv_launcher<VT, IT>;
#else
                        one_prec_kernel_func_ptr = spmv_omp_scs_adv<VT, IT>;
                        one_prec_warmup_kernel_func_ptr = spmv_omp_scs_adv<VT, IT>;
#endif
                    }
                    else if(config->value_type == "ap[dp_sp]"){
                        if(my_rank == 0){printf("Advanced ap[dp_sp] SCS kernel selected\n");}
#ifdef __CUDACC__
                        multi_prec_kernel_func_ptr = spmv_gpu_ap_scs_adv_launcher<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_gpu_ap_scs_adv_launcher<IT>;
#else
                        multi_prec_kernel_func_ptr = spmv_omp_scs_ap_adv<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_warmup_omp_scs_ap<IT>;
#endif
                    }
                    else if (config->value_type == "ap[dp_hp]"){
                        if(my_rank == 0){printf("Advanced ap[dp_hp] SCS kernel selected\n");}
#ifdef __CUDACC__
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Advanced ap[dp_hp] SCS kernel not yet implemented on GPU.\n");
                            exit(1);
                        }
#else
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Advanced ap[dp_hp] SCS kernel not yet implemented on CPU.\n");
                            exit(1);
                        }
#endif
                    }
                    else if (config->value_type == "ap[sp_hp]"){
#ifdef __CUDACC__
                        if(my_rank == 0){printf("Advanced ap[sp_hp] SCS kernel selected\n");}
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Advanced ap[sp_hp] SCS kernel not yet implemented on GPU.\n");
                            exit(1);
                        }
#else
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Advanced ap[sp_hp] SCS kernel not yet implemented on CPU.\n");
                            exit(1);
                        }
#endif
                    }
                    else if (config->value_type == "ap[dp_sp_hp]"){
                        if(my_rank == 0){printf("Advanced ap[dp_sp_hp] SCS kernel selected\n");}
#ifdef __CUDACC__
                        // three_prec_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        // three_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Advanced ap[dp_sp_hp] SCS kernel not yet implemented on GPU.\n");
                            exit(1);
                        }
#else
                        // three_prec_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        // three_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Advanced ap[dp_sp_hp] SCS kernel not yet implemented on CPU.\n");
                            exit(1);
                        }
#endif
                    }
                }
                else{
                    // If C is an uncommon choice, opt for non-adv kernels
#ifdef USE_CUSPARSE
                    if(my_rank == 0){printf("CUSPARSE SCS kernel selected\n");}
#else
                    if(my_rank == 0){printf("C = %i => Basic SCS kernel selected\n", config->chunk_size);}
#endif
                    if (config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){

#ifdef __CUDACC__
                        one_prec_kernel_func_ptr = spmv_gpu_scs_launcher<VT, IT>;
                        one_prec_warmup_kernel_func_ptr = spmv_gpu_scs_launcher<VT, IT>;
#else
                        one_prec_kernel_func_ptr = spmv_omp_scs<VT, IT>;
                        one_prec_warmup_kernel_func_ptr = spmv_omp_scs<VT, IT>;
#endif
                    }
                    else if(config->value_type == "ap[dp_sp]"){
                        if(my_rank == 0){printf("Basic ap[dp_sp] SCS kernel selected\n");}
#ifdef __CUDACC__
                        multi_prec_kernel_func_ptr = spmv_gpu_ap_scs_launcher<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_gpu_ap_scs_launcher<IT>;
#else
                        multi_prec_kernel_func_ptr = spmv_omp_scs_ap<IT>;
                        multi_prec_warmup_kernel_func_ptr = spmv_warmup_omp_scs_ap<IT>;
#endif
                    }
                    else if (config->value_type == "ap[dp_hp]"){
                        if(my_rank == 0){printf("Basic ap[dp_hp] SCS kernel selected\n");}
#ifdef __CUDACC__
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Basic ap[dp_hp] SCS kernel not yet implemented on GPU.\n");
                            exit(1);
                        }
#else
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Basic ap[dp_hp] SCS kernel not yet implemented on CPU.\n");
                            exit(1);
                        }
#endif
                    }
                    else if (config->value_type == "ap[sp_hp]"){
#ifdef __CUDACC__
                        if(my_rank == 0){printf("Basic ap[sp_hp] SCS kernel selected\n");}
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Basic ap[sp_hp] SCS kernel not yet implemented on GPU.\n");
                            exit(1);
                        }
#else
                        // two_prec_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        // two_prec_warmup_kernel_func_ptr = spmv_omp_csr_apsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Basic ap[sp_hp] SCS kernel not yet implemented on CPU.\n");
                            exit(1);
                        }
#endif
                    }
                    else if (config->value_type == "ap[dp_sp_hp]"){
                        if(my_rank == 0){printf("Basic ap[dp_sp_hp] SCS kernel selected\n");}
#ifdef __CUDACC__
                        // three_prec_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        // three_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Basic ap[dp_sp_hp] SCS kernel not yet implemented on GPU.\n");
                            exit(1);
                        }
#else
                        // three_prec_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        // three_prec_warmup_kernel_func_ptr = spmv_omp_csr_apdpsphp<IT>;
                        if(my_rank == 0){
                            fprintf(stderr, "ERROR: Basic ap[dp_sp_hp] SCS kernel not yet implemented on CPU.\n");
                            exit(1);
                        }
#endif
                    }
                }
            }
        }

        inline void init_halo_exchange(void){
#ifdef USE_MPI
            int outgoing_buf_size, incoming_buf_size;
            int receiving_proc, sending_proc;

            // First, post receives
            for (int from_proc_idx = 0; from_proc_idx < nzs_size; ++from_proc_idx)
            {
                sending_proc = local_context->non_zero_senders[from_proc_idx];
                incoming_buf_size = local_context->recv_counts_cumsum[sending_proc + 1] - local_context->recv_counts_cumsum[sending_proc];
#ifdef DEBUG_MODE
                std::cout << "I'm proc: " << my_rank << ", receiving: " << incoming_buf_size << " elements from a message with recv request: " << &recv_requests[from_proc_idx] << std::endl;
#endif
                if(config->value_type == "dp"){
                    MPI_Irecv(
                        &(local_x)[num_local_elems + local_context->recv_counts_cumsum[sending_proc]],
                        incoming_buf_size,
                        MPI_DOUBLE,
                        sending_proc,
                        (local_context->recv_tags[sending_proc])[my_rank],
                        MPI_COMM_WORLD,
                        &recv_requests[from_proc_idx]
                    );
                }
                else if (config->value_type == "sp")
                {
                    MPI_Irecv(
                        &(local_x)[num_local_elems + local_context->recv_counts_cumsum[sending_proc]],
                        incoming_buf_size,
                        MPI_FLOAT,
                        sending_proc,
                        (local_context->recv_tags[sending_proc])[my_rank],
                        MPI_COMM_WORLD,
                        &recv_requests[from_proc_idx]
                    );
                }
                else if(config->value_type == "hp"){
                    // TODO: Need to define your own MPI datatype for HALF
                }
            }

            // Second, fulfill those with sends
            for (int to_proc_idx = 0; to_proc_idx < nzr_size; ++to_proc_idx)
            {
                receiving_proc = local_context->non_zero_receivers[to_proc_idx];
                outgoing_buf_size = local_context->send_counts_cumsum[receiving_proc + 1] - local_context->send_counts_cumsum[receiving_proc];

#ifdef DEBUG_MODE
                if(local_context->comm_send_idxs[receiving_proc].size() != outgoing_buf_size){
                    std::cout << "init_halo_exchange ERROR: Mismatched buffer lengths in communication" << std::endl;
                    exit(1);
                }
#endif

                // Move non-contiguous data to a contiguous buffer for communication
                #pragma omp parallel for if(config->par_pack)   
                for(int i = 0; i < outgoing_buf_size; ++i)
                    (to_send_elems[to_proc_idx])[i] = local_x[perm[local_context->comm_send_idxs[receiving_proc][i]]];
                
#ifdef DEBUG_MODE
                std::cout << "I'm proc: " << my_rank << ", sending: " << outgoing_buf_size << " elements with a message with send request: " << &send_requests[to_proc_idx] << std::endl;
#endif
                if(config->value_type == "dp"){
                    MPI_Isend(
                        &(to_send_elems[to_proc_idx])[0],
                        outgoing_buf_size,
                        MPI_DOUBLE,
                        receiving_proc,
                        (local_context->send_tags[my_rank])[receiving_proc],
                        MPI_COMM_WORLD,
                        &send_requests[to_proc_idx]
                    );
                }
                else if (config->value_type == "sp")
                {
                    MPI_Isend(
                        &(to_send_elems[to_proc_idx])[0],
                        outgoing_buf_size,
                        MPI_FLOAT,
                        receiving_proc,
                        (local_context->send_tags[my_rank])[receiving_proc],
                        MPI_COMM_WORLD,
                        &send_requests[to_proc_idx]
                    );
                }
                else if(config->value_type == "hp"){
                    // TODO: Need to define your own MPI datatype for HALF
                }
            }
#endif
        }

        inline void finalize_halo_exchange(void){
#ifdef USE_MPI
            MPI_Waitall(nzr_size, send_requests, MPI_STATUS_IGNORE);
            MPI_Waitall(nzs_size, recv_requests, MPI_STATUS_IGNORE);
#endif
        }

        // Warmup kernel picker //
        // TODO: You really should think of a way around these warmup kernels...
        inline void execute_warmup_one_prec_spmv(void){
#ifdef USE_CUSPARSE
            cusparseSpMV(
                handle,
                opA,
                alpha,
                matA,
                vecX,
                beta,
                vecY,
                computeType,
                alg,
                externalBuffer
            );
#else
            one_prec_warmup_kernel_func_ptr(
                C,
                n_chunks,
                chunk_ptrs,
                chunk_lengths,
                col_idxs,
                values,
                local_x,
                local_y,
#ifdef __CUDACC__
                n_blocks_1,
#endif
                &my_rank
            );
#endif

#ifdef __CUDACC__
            cudaDeviceSynchronize();
#endif
        }

        inline void execute_warmup_two_prec_spmv(void){
            multi_prec_warmup_kernel_func_ptr(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_local_x,
                dp_local_y, 
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                sp_local_x,
                sp_local_y,
#ifdef HAVE_HALF_MATH
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                hp_local_x,
                hp_local_y,
#endif
#ifdef __CUDACC__
                n_blocks_2,
#endif
                &my_rank
            );

#ifdef __CUDACC__
            cudaDeviceSynchronize();
#endif
        }

        inline void execute_warmup_three_prec_spmv(void){
            multi_prec_warmup_kernel_func_ptr(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_local_x,
                dp_local_y, 
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                sp_local_x,
                sp_local_y,
#ifdef HAVE_HALF_MATH
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                hp_local_x,
                hp_local_y,
#endif
#ifdef __CUDACC__
                n_blocks_2,
#endif
                &my_rank
            );

#ifdef __CUDACC__
            cudaDeviceSynchronize();
#endif
        }

        void execute_warmup_spmv(void){
            if(config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){
                execute_warmup_one_prec_spmv();
            }
            else if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_hp]"){
                execute_warmup_two_prec_spmv();
            }
            else if(config->value_type == "ap[dp_sp_hp]"){
                execute_warmup_three_prec_spmv();
            }
        }


        inline void execute_one_prec_spmv(void){
#ifdef USE_CUSPARSE
            cusparseSpMV(
                handle,
                opA,
                alpha,
                matA,
                vecX,
                beta,
                vecY,
                computeType,
                alg,
                externalBuffer
            );
#else
            one_prec_kernel_func_ptr(
                C,
                n_chunks,
                chunk_ptrs,
                chunk_lengths,
                col_idxs,
                values,
                local_x,
                local_y,
#ifdef __CUDACC__
                n_blocks_1,
#endif
                &my_rank
            );
#endif

#ifdef __CUDACC__
            cudaDeviceSynchronize();
#endif
        }

        inline void execute_two_prec_spmv(void){
            multi_prec_kernel_func_ptr(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_local_x,
                dp_local_y, 
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                sp_local_x,
                sp_local_y,
#ifdef HAVE_HALF_MATH
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                hp_local_x,
                hp_local_y,
#endif
#ifdef __CUDACC__
                n_blocks_2,
#endif
                &my_rank
            );

#ifdef __CUDACC__
            cudaDeviceSynchronize();
#endif
        }

        inline void execute_three_prec_spmv(void){
            multi_prec_kernel_func_ptr(
                dp_C,
                dp_n_chunks,
                dp_chunk_ptrs,
                dp_chunk_lengths,
                dp_col_idxs,
                dp_values,
                dp_local_x,
                dp_local_y, 
                sp_C,
                sp_n_chunks,
                sp_chunk_ptrs,
                sp_chunk_lengths,
                sp_col_idxs,
                sp_values,
                sp_local_x,
                sp_local_y,
#ifdef HAVE_HALF_MATH
                hp_C,
                hp_n_chunks,
                hp_chunk_ptrs,
                hp_chunk_lengths,
                hp_col_idxs,
                hp_values,
                hp_local_x,
                hp_local_y,
#endif
#ifdef __CUDACC__
                n_blocks_2,
#endif
                &my_rank
            );

#ifdef __CUDACC__
            cudaDeviceSynchronize();
#endif
        }

        void execute_spmv(void){
            if(config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){
                execute_one_prec_spmv();
            }
            else if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_hp]"){
                execute_two_prec_spmv();
            }
            else if(config->value_type == "ap[dp_sp_hp]"){
                execute_three_prec_spmv();
            }
        }

        // NOTE: Should also work with GPUs?
        inline void swap_local_vectors(){
#ifdef USE_CUSPARSE
            // cusparseDnVecDescr_t tmp;
            // tmp = vecX;
            // vecX = vecY;
            // vecY = vecX;
            // std::swap(&(vecX.data), &(vecY.data));
#else
            if(config->value_type == "dp" || config->value_type == "sp" || config->value_type == "hp"){
                std::swap(local_x, local_y);
            }
            else if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_hp]"){
                printf("ERROR: swap_local_vectors not yet implemented for ap.\n");
                exit(1);
                // std::swap(sp_local_x, sp_local_y);
                // std::swap(dp_local_x, dp_local_y);
            }
            else if(config->value_type == "ap[dp_sp_hp]"){
                printf("ERROR: swap_local_vectors not yet implemented for ap.\n");
                exit(1);
                // std::swap(sp_local_x, sp_local_y);
                // std::swap(dp_local_x, dp_local_y);
            }
#endif
        }
};


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

    void print(void);

    void copy(const MtxData<double, int> &rhs);

    // Useful operators for unit testing
    bool operator==(MtxData<VT, IT> &rhs)
    {
        return (
            (n_rows == rhs.n_rows) &&
            (n_cols == rhs.n_cols) &&
            (nnz == rhs.nnz) &&
            (is_sorted == rhs.is_sorted) &&
            (is_symmetric == rhs.is_symmetric) &&
            (I == rhs.I) &&
            (J == rhs.J) &&
            (values == rhs.values)
        );
    }

    void operator^(MtxData<VT, IT> &rhs)
    {
        if (n_rows != rhs.n_rows){
            std::cout << "n_rows != rhs.n_rows" << std::endl;
        }
        if (n_cols != rhs.n_cols){
            std::cout << "n_cols != rhs.n_cols" << std::endl;
        }
        if (nnz != rhs.nnz){
            std::cout << "nnz != rhs.nnz" << std::endl;
        }
        if (is_sorted != rhs.is_sorted){
            std::cout << "is_sorted == rhs.is_sorted" << std::endl;
        }
        if (is_symmetric != rhs.is_symmetric){
            std::cout << "is_symmetric != rhs.is_symmetric" << std::endl;
        }
        if(I != rhs.I){
            std::cout << "I != rhs.I" << std::endl;
        }
        if(I.size() != rhs.I.size()){
            std::cout << "I.size() " << I.size() << " != rhs.I.size() " << rhs.I.size() << std::endl;
        }
        if(J != rhs.J){
            std::cout << "J != rhs.J" << std::endl;
        }
        if(J.size() != rhs.J.size()){
            std::cout << "J.size() " << J.size() << " != rhs.J.size() " << rhs.I.size() << std::endl;
        }
        if(values != rhs.values){
            std::cout << "values != rhs.values" << std::endl;
        }
        if(values.size() != rhs.values.size()){
            std::cout << "values.size() " << values.size() << " != rhs.values.size() " << rhs.values.size() << std::endl;
        }
    }
};

template <typename VT, typename IT>
void MtxData<VT, IT>::print(void){
    std::cout << "n_rows = " << n_rows << std::endl;
    std::cout << "n_cols = " << n_cols << std::endl;
    std::cout << "nnz = " << nnz << std::endl;
    std::cout << "is_sorted = " << is_sorted << std::endl;
    std::cout << "is_symmetric = " << is_symmetric << std::endl;

    std::cout << "I = [";
    for(int i = 0; i < nnz; ++i){
        std::cout << I[i];
        if(i == nnz-1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }

    std::cout << "J = [";
    for(int i = 0; i < nnz; ++i){
        std::cout << J[i];
        if(i == nnz-1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }

    std::cout << "values = [";
    for(int i = 0; i < nnz; ++i){
        std::cout << values[i];
        if(i == nnz-1)
            std::cout << "]" << std::endl;
        else
            std::cout << ", ";
    }
    printf("\n");
}

template <typename VT, typename IT>
void MtxData<VT, IT>::copy( const MtxData<double, int> & rhs){
    this->n_rows = rhs.n_rows;
    this->n_cols = rhs.n_cols;
    this->nnz = rhs.nnz;
    this->is_sorted = rhs.is_sorted;
    this->is_symmetric = rhs.is_symmetric;

    this->I.resize(nnz);
    this->J.resize(nnz);
    this->values.resize(nnz);

    for(int i = 0; i < nnz; ++i){
        this->I[i] = rhs.I[i];
        this->J[i] = rhs.J[i];
        this->values[i] = static_cast<VT>(rhs.values[i]);
    }
}

// This is only a unit testing convenience
template <typename VT, typename IT>
struct ScsExplicitData
{
    std::vector<IT> chunk_ptrs;
    std::vector<IT> chunk_lengths;
    std::vector<IT> col_idxs;
    std::vector<VT> values;
    std::vector<IT> old_to_new_idx;
    std::vector<IT> new_to_old_idx;
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

    // V<IT, IT> chunk_ptrs;    // Chunk start offsets into col_idxs & values.
    // V<IT, IT> chunk_lengths; // Length of one row in a chunk.
    // V<IT, IT> col_idxs;
    // V<VT, IT> values;
    // V<IT, IT> old_to_new_idx;
    std::vector<IT> chunk_ptrs;    // Chunk start offsets into col_idxs & values.
    std::vector<IT> chunk_lengths; // Length of one row in a chunk.
    std::vector<IT> col_idxs;
    std::vector<VT> values;
    std::vector<IT> old_to_new_idx;
    // std::vector<int> new_to_old_idx; //inverse of above
    IT * new_to_old_idx;
    // TODO: ^ make V object as well?


    void permute(IT *_perm_, IT*  _invPerm_);
    void write_to_mtx_file(int my_rank, std::string file_out_name);
    void assign_explicit_test_data(ScsExplicitData<VT, IT> *explicit_test_data);
    // void do_rcm(void);
    void print(void);

    // Useful operators for unit testing
    bool operator==(ScsData<VT, IT> &rhs)
    {
        bool chunk_ptrs_check = true;
        bool chunk_lengths_check = true;
        bool col_idxs_check = true;
        bool values_check = true;
        bool old_to_new_idx_check = true;
        bool new_to_old_idx_check = true;

        for(int i = 0; i < n_chunks+1; ++i){
            if(chunk_ptrs[i] != rhs.chunk_ptrs[i]){
                chunk_ptrs_check = false;
                break;
            }
        }

        for(int i = 0; i < n_chunks; ++i){
            if(chunk_lengths[i] != rhs.chunk_lengths[i]){
                chunk_lengths_check = false;
                break;
            }
        }

        ST n_scs_elements = chunk_ptrs[n_chunks - 1]
                    + chunk_lengths[n_chunks - 1] * C;

        for(int i = 0; i < n_scs_elements; ++i){
            if(col_idxs[i] != rhs.col_idxs[i]){
                col_idxs_check = false;
                break;
            }
        }

        for(int i = 0; i < n_scs_elements; ++i){
            if(values[i] != rhs.values[i]){
                values_check = false;
                break;
            }
        }

        for(int i = 0; i < n_rows; ++i){
            if(old_to_new_idx[i] != rhs.old_to_new_idx[i]){
                old_to_new_idx_check = false;
                break;
            }
        }

        for(int i = 0; i < n_rows; ++i){
            if(new_to_old_idx[i] != rhs.new_to_old_idx[i]){
                new_to_old_idx_check = false;
                break;
            }
        }

        return (
            (C == rhs.C) &&
            (sigma == rhs.sigma) &&
            (n_rows == rhs.n_rows) &&
            (n_cols == rhs.n_cols) &&
            (n_rows_padded == rhs.n_rows_padded) &&
            (n_chunks == rhs.n_chunks) &&
            (n_elements == rhs.n_elements) &&
            (nnz == rhs.nnz) &&
            chunk_ptrs_check &&
            chunk_lengths_check &&
            col_idxs_check &&
            values_check &&
            old_to_new_idx_check &&
            new_to_old_idx_check
        );
    }

    void operator^(ScsData<VT, IT> &rhs)
    {
        if (C != rhs.C){std::cout << "C != rhs.C" << std::endl;}
        if (sigma != rhs.sigma){std::cout << "sigma != rhs.sigma" << std::endl;}
        if (n_rows != rhs.n_rows){std::cout << "n_rows != rhs.n_rows" << std::endl;}
        if (n_cols != rhs.n_cols){std::cout << "n_cols != rhs.n_cols" << std::endl;}
        if (n_rows_padded != rhs.n_rows_padded){std::cout << "n_rows_padded != rhs.n_rows_padded" << std::endl;}
        if (n_chunks != rhs.n_chunks){std::cout << "n_chunks != rhs.n_chunks" << std::endl;}
        if (n_elements != rhs.n_elements){std::cout << "n_elements != rhs.n_elements" << std::endl;}
        if (nnz != rhs.nnz){std::cout << "nnz != rhs.nnz" << std::endl;}

        for(int i = 0; i < n_chunks+1; ++i){
            if(chunk_ptrs[i] != rhs.chunk_ptrs[i]){
                std::cout << "chunk_ptrs != rhs.chunk_ptrs" << std::endl;
                break;
            }
        }

        for(int i = 0; i < n_chunks; ++i){
            if(chunk_lengths[i] != rhs.chunk_lengths[i]){
                std::cout << "chunk_lengths != rhs.chunk_lengths" << std::endl;
                break;
            }
        }

        ST n_scs_elements = chunk_ptrs[n_chunks - 1]
                    + chunk_lengths[n_chunks - 1] * C;


        for(int i = 0; i < n_scs_elements; ++i){
            if(col_idxs[i] != rhs.col_idxs[i]){
                std::cout << "col_idxs != rhs.col_idxs" << std::endl;
                break;
            }
        }

        for(int i = 0; i < n_scs_elements; ++i){
            if(values[i] != rhs.values[i]){
                std::cout << "values != rhs.values" << std::endl;
                break;
            }
        }

        for(int i = 0; i < n_rows; ++i){
            if(old_to_new_idx[i] != rhs.old_to_new_idx[i]){
                std::cout << "old_to_new_idx != rhs.old_to_new_idx" << std::endl;
                break;
            }
        }

        for(int i = 0; i < n_rows; ++i){
            if(new_to_old_idx[i] != rhs.new_to_old_idx[i]){
                std::cout << "new_to_old_idx != rhs.new_to_old_idx" << std::endl;
                break;
            }
        }
    }
};

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
        std::cout << values[i];
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

// template <typename VT, typename IT>
// bool ScsData<VT, IT>::rcm_viable(void){}

// template <typename VT, typename IT>
// void ScsData<VT, IT>::do_rcm(void){
//     int nrows = n_rows;
//     int rowPtr = chunk_ptrs.data();
//     SpMP::CSR *csr = NULL;
//     csr = new SpMP::CSR(nrows, nrows, rowPtr, col, val);
//     int *rcmPerm;
//     int *rcmInvPerm;
//     if(csr->isSymmetric(true, true)){
//         int orig_threads = 1;
//         printf("Doing RCM permutation\n");
//         #pragma omp parallel
//         {
//             orig_threads = omp_get_num_threads();
//         }
//         omp_set_num_threads(1);

//         if(csr->isSymmetric(false,false))
//         {
//             bfsPerm = new int[nrows];
//             bfsInvPerm = new int[nrows];
//             csr->getBFSPermutation(perm, inversePerm);
//             csr->getRCMPermutation(rcmInvPerm, rcmPerm);
//         }
//         else
//         {
//             printf("Matrix not symmetric RCM cannot be done\n");
//         }
//         omp_set_num_threads(orig_threads);
//         delete csr;
//     }
//     else{
//         printf("do_rcm ERROR: Matrix not symmetric, cannot perform RCM.\n");
//         exit(1);
//     }
// }

// template <typename VT, typename IT>
// void ScsData<VT, IT>::do_bfs(void){
//     int nrows = n_rows;
//     int rowPtr = chunk_ptrs.data();
//     SpMP::CSR *csr = NULL;
//     csr = new SpMP::CSR(nrows, nrows, rowPtr, col, val);
//     int *rcmPerm;
//     int *rcmInvPerm;
//     if(csr->isSymmetric(true, true)){
//         int orig_threads = 1;
//         printf("Doing RCM permutation\n");
//         #pragma omp parallel
//         {
//             orig_threads = omp_get_num_threads();
//         }
//         omp_set_num_threads(1);

//         if(csr->isSymmetric(false,false))
//         {
//             bfsPerm = new int[nrows];
//             bfsInvPerm = new int[nrows];
//             csr->getBFSPermutation(perm, inversePerm);
//             csr->getRCMPermutation(rcmInvPerm, rcmPerm);
//         }
//         else
//         {
//             printf("Matrix not symmetric RCM cannot be done\n");
//         }
//         omp_set_num_threads(orig_threads);
//         delete csr;
//     }
//     else{
//         printf("do_rcm ERROR: Matrix not symmetric, cannot perform RCM.\n");
//         exit(1);
//     }
// }

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
void ScsData<VT, IT>::assign_explicit_test_data(ScsExplicitData<VT, IT> *explicit_test_data){
    // Uses new V operator
    chunk_ptrs=(&explicit_test_data->chunk_ptrs);
    chunk_lengths=(&explicit_test_data->chunk_lengths);
    col_idxs=(&explicit_test_data->col_idxs);
    values<=(&explicit_test_data->values);
    old_to_new_idx=(&explicit_test_data->old_to_new_idx);
    new_to_old_idx=explicit_test_data->new_to_old_idx;
}

template <typename VT, typename IT>
void ScsData<VT, IT>::write_to_mtx_file(
    int my_rank,
    std::string file_out_name)
{
    //Convert csr back to coo for mtx format printing
    std::vector<int> temp_rows(n_elements);
    std::vector<int> temp_cols(n_elements);
    std::vector<double> temp_values(n_elements);

    int elem_num = 0;
    for(int row = 0; row < n_rows; ++row){
        for(int idx = chunk_ptrs[row]; idx < chunk_ptrs[row + 1]; ++idx){
            temp_rows[elem_num] = row + 1; // +1 to adjust for 1 based indexing in mm-format
            temp_cols[elem_num] = col_idxs[idx] + 1;
            temp_values[elem_num] = values[idx];
            ++elem_num;
        }
    }

    std::string file_name = file_out_name + "_out_matrix_rank_" + std::to_string(my_rank) + ".mtx"; 

    mm_write_mtx_crd(
        &file_name[0], 
        n_rows, 
        n_cols, 
        n_elements, 
        &(temp_rows)[0], 
        &(temp_cols)[0], 
        &(temp_values)[0], 
        "MCRG" // TODO: <- make more general, i.e. flexible based on the matrix. Read from original mtx?
    );
}

template <typename VT, typename IT>
struct DefaultValues
{
#ifdef HAVE_HALF_MATH
    VT A{2.0f16};
    VT x{1.00f16};
#else 
    VT A{2.0};
    VT x{1.00};
#endif

    VT y{};

    VT *x_values{};
    ST n_x_values{};

    VT *y_values{};
    ST n_y_values{};
};

template <typename VT, typename IT>
struct Result
{
    double perf_gflops{};
    double mem_mb{};
    std::vector<double> perfs_from_procs; // used in Gather

    std::vector<unsigned long> nnz_per_proc;

    // Used in mp
    std::vector<unsigned long> dp_nnz_per_proc;
    std::vector<unsigned long> sp_nnz_per_proc;
    std::vector<unsigned long> hp_nnz_per_proc;
    unsigned long sp_nnz;
    unsigned long dp_nnz;
    unsigned long hp_nnz;
    unsigned long cumulative_dp_nnz;
    unsigned long cumulative_sp_nnz;
    unsigned long cumulative_hp_nnz;
    double total_dp_percent;
    double total_sp_percent;
    double total_hp_percent;

    unsigned long total_nnz;
    unsigned long total_rows;

    double euclid_dist;
    double mkl_magnitude;

    unsigned int size_value_type{};
    unsigned int size_index_type{};

    unsigned long n_calls{};
    double duration_total_s{};
    double duration_kernel_s{};

    bool is_result_valid{false};
    std::string notes;

    std::string value_type_str;
    std::string index_type_str;

    int value_type_size{};
    int index_type_size{};

    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    double fill_in_percent{};
    long C{};
    long sigma{};
    long nzr{};

    bool was_matrix_sorted{false};

    double mem_m_mb{};
    double mem_x_mb{};
    double mem_y_mb{};

    double beta{};
    double dp_beta{};
    double sp_beta{};
#ifdef HAVE_HALF_MATH
    double hp_beta{};
#endif

    double cb_a_0{};
    double cb_a_nzc{};

    std::vector<VT> y_out;
    std::vector<VT> x_out;
    std::vector<VT> total_uspmv_result;
    std::vector<VT> total_x;

    double total_walltime;
};

// NOTE: purely for convieience, not entirely necessary
template <typename ST>
struct MtxDataBookkeeping
{
    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    bool is_sorted{};
    bool is_symmetric{};
};

#endif