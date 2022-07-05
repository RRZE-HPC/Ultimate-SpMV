#include "spmv.h"
#include "mtx-reader.h"
#include "vectors.h"
#include "utilities.hpp"
#include "kernels.hpp"

#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

// The default C to use for sell-c-sigma, when no C is specified.
enum
{
    SCS_DEFAULT_C = 8
};

// Initialize all matrices and vectors the same.
// Use -rand to initialize randomly.
static bool g_same_seed_for_every_vector = true;

// // Log information.
// static bool g_log = false;

template <typename VT, typename IT>
using V = Vector<VT, IT>;

// TODO: combine Config and DefaultValues
// No need for two seperate structs
struct Config
{
    long n_els_per_row{-1}; // ell
    long chunk_size{SCS_DEFAULT_C};    // sell-c-sigma
    long sigma{1};         // sell-c-sigma

    // Initialize rhs vector with random numbers.
    bool random_init_x{true};
    // Override values of the matrix, read via mtx file, with random numbers.
    bool random_init_A{false};

    // No. of repetitions to perform. 0 for automatic detection.
    unsigned long n_repetitions{10};

    // Verify result of SpVM.
    bool verify_result{true};

    // Verify result against solution of COO kernel.
    bool verify_result_with_coo{true};

    // Print incorrect elements from solution.
    bool verbose_verification{true};

    // Sort rows/columns of sparse matrix before
    // converting it to a specific format.
    bool sort_matrix{true};

    bool validate_result{true};

    // Configures if the code will be executed in bench mode or compute mode
    std::string mode = "bench"; 
};

template <typename VT, typename IT>
struct DefaultValues
{
    VT A{2.0};
    VT x{1.01};
    VT y{};

    VT *x_values{};
    ST n_x_values{};

    VT *y_values{};
    ST n_y_values{};
};

struct BenchmarkResult
{
    double perf_gflops{};
    double mem_mb{};

    unsigned int size_value_type{};
    unsigned int size_index_type{};

    unsigned long n_calls{};
    double duration_total_s{};
    double duration_kernel_s{};

    bool is_result_valid{false};
    std::string notes;

    std::string value_type_str;
    std::string index_type_str;

    uint64_t value_type_size{};
    uint64_t index_type_size{};

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

    double cb_a_0{};
    double cb_a_nzc{};
};

template <typename VT>
static void
random_init(VT *begin, VT *end)
{
    std::mt19937 engine;

    if (!g_same_seed_for_every_vector)
    {
        std::random_device rnd_device;
        engine.seed(rnd_device());
    }

    std::uniform_real_distribution<double> dist(0.1, 2.0);

    for (VT *it = begin; it != end; ++it)
    {
        *it = random_number<VT, decltype(dist), decltype(engine)>::get(dist, engine);
    }
}

template <typename VT, typename IT>
static void
random_init(V<VT, IT> &v)
{
    random_init(v.data(), v.data() + v.n_rows);
}

template <typename VT, typename IT>
static void
init_with_ptr_or_value(V<VT, IT> &x,
                       ST n_x,
                       const std::vector<VT> *x_in,
                       VT default_value,
                       bool init_with_random_numbers = false)
{
    if (!init_with_random_numbers)
    {
        if (x_in)
        {
            if (x_in->size() != size_t(n_x))
            {
                fprintf(stderr, "ERROR: x_in has incorrect size.\n");
                exit(1);
            }

            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = (*x_in)[i];
            }
        }
        else
        {
            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = default_value;
            }
        }
    }
    else
    {
        random_init(x);
    }
}

template <typename VT>
static void
init_std_vec_with_ptr_or_value(std::vector<VT> &x,
                               ST n_x,
                               const std::vector<VT> *x_in,
                               VT default_value,
                               bool init_with_random_numbers = false)
{
    if (!init_with_random_numbers)
    {
        if (x_in)
        {
            if (x_in->size() != size_t(n_x))
            {
                fprintf(stderr, "ERROR: x_in has incorrect size.\n");
                exit(1);
            }

            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = (*x_in)[i];
            }
        }
        else
        {
            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = default_value;
            }
        }
    }
    else
    {
        random_init(&(*x.begin()), &(*x.end()));
    }
}


/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void collect_local_needed_heri(
    std::vector<IT> *local_needed_heri, 
    const MtxData<VT, IT> &local_mtx, 
    const int *work_sharing_arr){
    /* Here, we collect the row indicies of the halo elements needed for this process to have a valid local_x_scs to perform the SPMVM.
    These are organized as tuples in a vector, of the form (proc to, proc from, global row idx). The row_idx
    refers to the "global" x vector, this will be adjusted (localized) later when said element is "retrieved".
    The "proc" of where this needed element resides is known from the work sharing array.
    Each process needs to know the required halo elements for every other process. */

    // TODO: better to pass as args to function?
    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    IT from_proc, total_x_row_idx, remote_elem_candidate_col, remote_elem_col;
    // std::vector<std::tuple<IT, IT, IT>> global_needed_heri;
    int needed_heri_count = 0;
    // std::tuple<IT, IT, IT> needed_tup;

    // First, assemble the global view of required elements
    // for(int rank = 0; rank < comm_size; ++rank){ // TODO: Would love to parallelize
    for(int i = 0; i < local_mtx.nnz; ++i){ // this is only MY nnz, not the nnz of the process_local_mtx were looking at
        // If true, this is a remote element, and needs to be added to vector
        if(local_mtx.J[i] < work_sharing_arr[my_rank] || local_mtx.J[i] > work_sharing_arr[my_rank + 1] - 1){
            // printf("Remote Element!\n");
            remote_elem_col = local_mtx.J[i];
            // Deduce from which process the required remote element lies
            for(int j = 0; j < comm_size; ++j){
                // printf("%i, %i, %i\n", remote_elem_col, work_sharing_arr[j], work_sharing_arr[j + 1]);
                // NOTE: not comepletly sure about about cases on the edge here
                if(remote_elem_col >= work_sharing_arr[j] && remote_elem_col < work_sharing_arr[j + 1]){
                    // printf("I should be here twice\n");
                    from_proc = j;
                    local_needed_heri->push_back(my_rank);
                    local_needed_heri->push_back(from_proc);
                    local_needed_heri->push_back(remote_elem_col);
                    ++needed_heri_count;
                }
            }
        }
    }
}

/**
    Description...
    @param
    @return
*/
template <typename IT>
void collect_to_send_heri(
    std::vector<IT> *to_send_heri, 
    std::vector<IT> *local_needed_heri, 
    int *global_needed_heri,
    int *global_needed_heri_size
    ){

    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // int *all_local_needed_heri_sizes = new int[comm_size];
    int all_local_needed_heri_sizes[comm_size];// = {0}; // Do these need to be on the heap?
    // int *global_needed_heri_displ_arr = new int[comm_size]; //not needed anymore?
    int global_needed_heri_displ_arr[comm_size];// = {0};

    int local_needed_heri_size = local_needed_heri->size();
    
    // First, gather the sizes of messages for the Allgatherv later
    MPI_Allgather(&local_needed_heri_size,
                1,
                MPI_INT,
                all_local_needed_heri_sizes, //should be an array of length "comm size"
                1,
                MPI_INT,
                MPI_COMM_WORLD);

    int intermediate_size = 0;

    // TODO: sort of extra work, since the global_needed_heri_size already calculated? Maybe rename.
    // Second, sum this array, which will be the size of the global_needed_heri_size
    for(int i = 0; i < comm_size; ++i){
        global_needed_heri_displ_arr[i] = intermediate_size;

        // TODO: what are the causes and implications of global_needed_heri_size?
        intermediate_size += all_local_needed_heri_sizes[i];
    }

    MPI_Allgatherv(&(*local_needed_heri)[0],
                local_needed_heri_size,
                MPI_INT,
                global_needed_heri,
                all_local_needed_heri_sizes,
                global_needed_heri_displ_arr,
                MPI_INT,
                MPI_COMM_WORLD);

    // Finally, sort the global_needed_heri into "to_send_heri". Encoded 3-tuples as array
    for(int from_proc_idx = 1; from_proc_idx < intermediate_size; from_proc_idx += 3){
        if(global_needed_heri[from_proc_idx] == my_rank){
            to_send_heri->push_back(global_needed_heri[from_proc_idx - 1]);
            to_send_heri->push_back(global_needed_heri[from_proc_idx]);
            to_send_heri->push_back(global_needed_heri[from_proc_idx + 1]);
        }
    }
}

/**
    Description...
    @param
    @return
*/
template <typename IT>
void calc_heri_shifts(int *global_needed_heri, int *global_needed_heri_size, int *shift_arr, int *incidence_arr){
    /* We pre-calculate the shift for each given proc_to and proc_from, to look it up quickly in the benchmark loop later */

    int comm_size, my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int current_row, current_col, from_proc, to_proc;

    for(int i = 0; i < *global_needed_heri_size; i += 3){
        to_proc = global_needed_heri[i];
        from_proc = global_needed_heri[i + 1];

        incidence_arr[comm_size * from_proc + to_proc] += 1;
    }

    for(int row = 1; row < comm_size; ++row){
        int rows_remaining = comm_size - row;
        for(int i = 0; i < rows_remaining; ++i){
            for(int col = 0; col < comm_size; ++col){
                shift_arr[comm_size * (row + i) + col] += incidence_arr[comm_size * (row - 1) + col];
            }
        }
    }
}

// TODO: Best to pass pointers explicitly? Make consistent
/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void communicate_halo_elements(
    std::vector<IT> *local_needed_heri, 
    std::vector<IT> *to_send_heri, 
    std::vector<VT> *local_x_scs_vec, 
    int const *shift_arr, 
    const int *work_sharing_arr){
    /* The purpose of this function, is to allow each process to exchange it's proc-local "remote elements"
    with the other respective processes. For now, elements are sent one at a time. 
    Recall, tuple elements in needed_heri are formatted (proc to, proc from, global x idx). 
    Since this function is in the benchmark loop, need to do as little work possible, ideally.*/

    // TODO: better to pass as args to function?
    int my_rank, comm_size;
    int rows_in_to_proc, rows_in_from_proc, to_proc, from_proc, global_row_idx, num_local_elems;
    int recv_shift = 0, send_shift = 0; // TODO: calculate more efficiently
    int recieved_elems = 0, sent_elems = 0;
    int test_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Request request; // TODO: What does this do?

    // Declare and populate arrays to keep track of number of elements already
    // recieved or sent to respective procs
    int recv_counts[comm_size];
    int send_counts[comm_size];
    for(int i = 0; i < comm_size; ++i){
        recv_counts[i] = 0;
        send_counts[i] = 0;
    }

    // // TODO: DRY
    if(typeid(VT) == typeid(float)){
        // In order to place incoming elements in padded region of local_x, i.e. AFTER local elements
        rows_in_to_proc = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

        // TODO: definetly OpenMP this loop? Improve somehow
        for(int from_proc_idx = 1; from_proc_idx < local_needed_heri->size(); from_proc_idx += 3){
            // How is this calculated, and is it totally necessary?
            rows_in_from_proc = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

            to_proc = (*local_needed_heri)[from_proc_idx - 1];
            from_proc = (*local_needed_heri)[from_proc_idx];

            // NOTE: Source of uniqueness for tag
            // NOTE: essentially "transpose" shift array here
            recv_shift = shift_arr[comm_size * from_proc + to_proc];

            MPI_Irecv(
                &(*local_x_scs_vec)[rows_in_to_proc + recv_shift + recv_counts[from_proc]],
                1,
                MPI_FLOAT,
                from_proc,
                recv_shift + recv_counts[from_proc],
                MPI_COMM_WORLD,
                &request);

            // if(my_rank == test_rank){
            //     std::cout << "Proc " << my_rank << " asks for tag " << recv_shift + recv_counts[from_proc] << " from " << from_proc << std::endl;
            // }
          
            ++recv_counts[from_proc];
        }
        for(int to_proc_idx = 0; to_proc_idx < to_send_heri->size(); to_proc_idx += 3){
            global_row_idx = to_proc_idx + 2;
            to_proc = (*to_send_heri)[to_proc_idx];
            from_proc = my_rank;

            send_shift = shift_arr[comm_size * from_proc + to_proc];

            MPI_Send(
                &(*local_x_scs_vec)[(*to_send_heri)[global_row_idx] - work_sharing_arr[my_rank]],
                1,
                MPI_FLOAT,
                to_proc,
                send_shift + send_counts[to_proc],
                MPI_COMM_WORLD);

            // if(my_rank == test_rank){
            //     float data =  (*local_x_scs_vec)[(*to_send_heri)[global_row_idx] - work_sharing_arr[my_rank]]; // assume this is the correct data to send for now
            //     std::cout << "Proc: " << my_rank << " sends to " << to_proc << " the data " << data << " with a tag of " << send_shift + send_counts[to_proc] << std::endl;
            // }
            ++send_counts[to_proc];
        }
    }
    else if(typeid(VT) == typeid(double)){
        rows_in_to_proc = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

        for(int from_proc_idx = 1; from_proc_idx < local_needed_heri->size(); from_proc_idx += 3){
            rows_in_from_proc = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

            to_proc = (*local_needed_heri)[from_proc_idx - 1];
            from_proc = (*local_needed_heri)[from_proc_idx];

            recv_shift = shift_arr[comm_size * from_proc + to_proc];

            MPI_Irecv(
                &(*local_x_scs_vec)[rows_in_to_proc + recv_shift + recv_counts[from_proc]],
                1,
                MPI_DOUBLE,
                from_proc,
                recv_shift + recv_counts[from_proc],
                MPI_COMM_WORLD,
                &request);
          
            ++recv_counts[from_proc];
        }
        for(int to_proc_idx = 0; to_proc_idx < to_send_heri->size(); to_proc_idx += 3){
            global_row_idx = to_proc_idx + 2;
            to_proc = (*to_send_heri)[to_proc_idx];
            from_proc = my_rank;

            send_shift = shift_arr[comm_size * from_proc + to_proc];

            MPI_Send(
                &(*local_x_scs_vec)[(*to_send_heri)[global_row_idx] - work_sharing_arr[my_rank]],
                1,
                MPI_DOUBLE,
                to_proc,
                send_shift + send_counts[to_proc],
                MPI_COMM_WORLD);

            ++send_counts[to_proc];
        }
    }
}

// TODO: Only need col_idxs. So dont pass reference to entire struct?
/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void adjust_halo_col_idxs(
    ScsData<VT, IT> & scs,
    // const MtxData<VT, IT> &local_mtx,
    int *amnt_local_elements,
    const int *work_sharing_arr){

    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int remote_elem_count = 0;
    int amnt_lhs_remote_elems = 0;
    // int test_rank = 1;

    // if(my_rank == test_rank){
    //     std::cout << "Proc: " << my_rank << " col idx" << std::endl;
    //      for(int i = 0; i < scs.n_elements; ++i){
    //         std::cout << scs.col_idxs[i] << std::endl;
    //      }
    // }

    // For every non zero scs element, count how many are left of local region
    for(int i = 0; i < scs.n_elements; ++i){
        if(scs.values[i] != 0){
            if(scs.col_idxs[i] < work_sharing_arr[my_rank]){ // assume same ordering...?
                ++amnt_lhs_remote_elems;
            }
        }
    }

    for(int i = 0; i < scs.n_elements; ++i){
        if(scs.values[i] != 0){
            if(scs.col_idxs[i] < work_sharing_arr[my_rank] || scs.col_idxs[i] > work_sharing_arr[my_rank + 1] - 1){
                scs.col_idxs[i] = *amnt_local_elements + remote_elem_count;
                ++remote_elem_count;
            }
            else{
                scs.col_idxs[i] -= work_sharing_arr[my_rank];
                // scs.col_idxs[i] -= remote_elem_count;
            }
        }
    }

    // if(my_rank == test_rank){
    //     std::cout << "Proc: " << my_rank << " new col idx" << std::endl;
    //      for(int i = 0; i < scs.n_elements; ++i){
    //         std::cout << scs.col_idxs[i] << std::endl;
    //      }
    // }
}

// TODO: what do I return for this?
// NOTE: every process will return something...
/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void bench_spmv_scs(
    const Config *config,
    const MtxData<VT, IT> &local_mtx,
    const int *work_sharing_arr,
    std::vector<VT> *total_y, // IDK if this is const
    DefaultValues<VT, IT> &defaults,
    const std::vector<VT> *x_in = nullptr)
{
    // TODO: More efficient just to deduce this from worksharingArr size?
    int comm_size;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    // const int c_comm_size = comm_size;

    log("allocate and place CPU matrices start\n");

    // TODO: what to do with the benchmark object?
    // BenchmarkResult r;

    ScsData<VT, IT> scs;

    log("allocate and place CPU matrices end\n");
    log("converting to scs format start\n");

    // TODO: fuse with x idx adjustments potentially
    convert_to_scs<VT, IT>(local_mtx, config->chunk_size, config->sigma, scs);

    log("converting to scs format end\n");

    // This y is only process local. Need an Allgather for each proc to have
    // all of the solution segments
    V<VT, IT> local_y_scs = V<VT, IT>(scs.n_rows_padded);
    std::uninitialized_fill_n(local_y_scs.data(), local_y_scs.n_rows, defaults.y);

    // NOTE: Every processes needs counts array
    int counts_arr[comm_size] = {0};
    int displ_arr_bk[comm_size] = {0};

    IT updated_col_idx, initial_col_idx = scs.col_idxs[0];

    int amnt_local_elements = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

    adjust_halo_col_idxs<VT, IT>(scs, &amnt_local_elements, work_sharing_arr);

    std::vector<VT> local_x(amnt_local_elements, 0);

    init_std_vec_with_ptr_or_value(local_x, local_x.size(), x_in,
                            defaults.x, false);

    if(config->mode == "solver"){
        // TODO: make modifications for solver mode
        // the object "returned" will be the gathered results vector
    }
    else if(config->mode == "bench"){
        // TODO: make modifications for bench mode
        // the object returned will be the benchmark results
    }

    // TODO: How bad is this for scalability? Is there a way around this?
    std::vector<VT> local_y_scs_vec(local_y_scs.data(), local_y_scs.data() + local_y_scs.n_rows);

    // heri := halo element row indices
    // "local_needed_heri" is all the halo elements that this process needs
    std::vector<IT> local_needed_heri;

    collect_local_needed_heri<VT, IT>(&local_needed_heri, local_mtx, work_sharing_arr);

    // "to_send_heri" are all halo elements that this process is to send
    std::vector<IT> to_send_heri;

    int local_needed_heri_size = local_needed_heri.size();
    int global_needed_heri_size;

    // TODO: Is this actually necessary?
    MPI_Allreduce(&local_needed_heri_size,
                &global_needed_heri_size,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);

    int *global_needed_heri = new int[global_needed_heri_size];

    for(int i = 0; i < global_needed_heri_size; ++i){
        global_needed_heri[i] = 0;
    }

    collect_to_send_heri<IT>(
        &to_send_heri, 
        &local_needed_heri, 
        global_needed_heri,
        &global_needed_heri_size
        );

    // The shift array is used in the tag-generation scheme in halo communication.
    // the row idx is the "from_proc", the column is the "to_proc", and the element is the shift
    // after the local element index to make for the incoming halo elements
    int *shift_arr = new int[comm_size * comm_size];
    int *incidence_arr = new int[comm_size * comm_size]; 

    for(int i = 0; i < comm_size * comm_size; ++i){
        shift_arr[i] = 0;
        incidence_arr[i] = 0;
    }

    calc_heri_shifts<IT>(global_needed_heri, &global_needed_heri_size, shift_arr, incidence_arr); //NOTE: always symmetric?

    // Test with 1 proc
    // if(my_rank == 0){
    //     local_x[0] = 1;
    //     local_x[1] = 2;
    //     local_x[2] = 3;
    //     local_x[3] = 4;
    //     local_x[4] = 5;
    // }

    // Test with 2 procs
    // if(my_rank == 0){
    //     local_x[0] = 1;
    //     local_x[1] = 2;
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
    if(my_rank == 0){
        local_x[0] = 1;
    }
    if(my_rank == 1){
        local_x[0] = 2;
    }
    if(my_rank == 2){
        local_x[0] = 3;
    }
    if(my_rank == 3){
        local_x[0] = 4;
        local_x[1] = 5;
    }

    // NOTE: should always be a multiple of 3
    int local_x_padding = local_needed_heri.size() / 3;

    // Prepare buffers for communication
    local_x.resize(local_x.size() + local_x_padding);
    
    // Enter main loop
    for (int i = 0; i < config->n_repetitions; ++i)
    {    
        // int test_rank = 1;

        // MPI_Barrier(MPI_COMM_WORLD); 
        // if(my_rank == test_rank){
        //     std::cout << "local_x before comm, before SPMV, before swap: " << std::endl;
        //     for(int i = 0; i < local_x.size(); ++i){
        //         std::cout << local_x[i] << std::endl;
        //     }
        // }

        communicate_halo_elements<VT, IT>(&local_needed_heri, &to_send_heri, &local_x, shift_arr, work_sharing_arr);
        // printf("\n");
        MPI_Barrier(MPI_COMM_WORLD); 

        // if(my_rank == test_rank){
        //     std::cout << "local_x AFTER comm, before SPMV, before swap: " << std::endl;
        //     for(int i = 0; i < local_x.size(); ++i){
        //         std::cout << local_x[i] << std::endl;
        //     }
        // }
        // MPI_Barrier(MPI_COMM_WORLD); // temporary
        // exit(0);

        spmv_omp_scs_c<VT, IT>( scs.C, scs.n_chunks, scs.chunk_ptrs.data(), 
                                scs.chunk_lengths.data(), scs.col_idxs.data(), 
                                scs.values.data(), &(local_x)[0], &(local_y_scs_vec)[0]);

        // TODO: verify, dont swap on last iteration?
        if(i != config->n_repetitions - 1){
            std::swap(local_x, local_y_scs_vec);
        }
    }

    // TODO: use a pragma parallel for?
    // Reformat proc-local result vectors. Only take the useful (non-padded) elements 
    // from the scs formatted local_y_scs, and assign to local_y
    std::vector<VT> local_y(scs.n_rows, 0);

    for (int i = 0; i < scs.old_to_new_idx.n_rows; ++i)
    {
        local_y[i] = local_y_scs_vec[scs.old_to_new_idx[i]];
    }

    // NOTE: only for collecting local results.
    // i.e. wont be apart of actual loop after halo comm implemented
    for (int i = 0; i < comm_size; ++i)
    {
        counts_arr[i] = work_sharing_arr[i + 1] - work_sharing_arr[i];
        displ_arr_bk[i] = work_sharing_arr[i];
    }

    if(config->validate_result){
        // Collect results from each proc to y_total
        if (typeid(VT) == typeid(double))
        {
            MPI_Allgatherv(&local_y[0],
                        local_y.size(),
                        MPI_DOUBLE,
                        &(*total_y)[0],
                        counts_arr,
                        displ_arr_bk,
                        MPI_DOUBLE,
                        MPI_COMM_WORLD);
        }
        else if (typeid(VT) == typeid(float))
        {
            MPI_Allgatherv(&local_y[0],
                        local_y.size(),
                        MPI_FLOAT,
                        &(*total_y)[0],
                        counts_arr,
                        displ_arr_bk,
                        MPI_FLOAT,
                        MPI_COMM_WORLD);
        }
    }
    else{
        // Dont collect local results to global, and just "return" local results
        total_y->resize(local_y.size());
        for (int i = 0; i < scs.old_to_new_idx.n_rows; ++i)
        {
            (*total_y)[i] = local_y[i];
        }
    }

    delete[] shift_arr;  
    delete[] incidence_arr;
    delete[] global_needed_heri;

    MPI_Barrier(MPI_COMM_WORLD); // temporary
    if(my_rank == 0){
        printf("\n");
        std::cout << "Resulting vector with: " << config->n_repetitions << " revisions" << std::endl; 
        for(int i = 0; i < total_y->size(); ++i){
            std::cout << (*total_y)[i] << std::endl;
        }
    }
}

// Honestly, probably not necessary
template <typename ST>
struct MtxDataBookkeeping
{
    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    bool is_sorted{};
    bool is_symmetric{};
};

/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void seg_work_sharing_arr(const MtxData<VT, IT> &mtx, int *work_sharing_arr, std::string seg_method, int comm_size)
{
    work_sharing_arr[0] = 0;

    int segment;

    if ("seg-rows" == seg_method)
    {
        int rowsPerProc;

        // Evenly split the number of rows
        rowsPerProc = mtx.n_rows / comm_size;

        // Segment rows to work on via. array
        for (segment = 1; segment < comm_size + 1; ++segment)
        {
            // Can only do this because of "constant sized" segments
            work_sharing_arr[segment] = segment * rowsPerProc;
            if (segment == comm_size)
            {
                // Set the last element to point to directly after the final row
                // (takes care of remainder rows)
                work_sharing_arr[comm_size] = mtx.I[mtx.nnz - 1] + 1;
            }
        }
    }
    else if ("seg-nnz" == seg_method)
    {
        int nnzPerProc; //, remainderNnz;

        // Split the number of rows based on non zeros
        nnzPerProc = mtx.nnz / comm_size;
        // remainderNnz = mtx.nnz % nnzPerProc;

        int global_ctr, local_ctr;
        segment = 1;
        local_ctr = 0;

        // Segment rows to work on via. array
        for (global_ctr = 0; global_ctr < mtx.nnz; ++global_ctr)
        {
            if (local_ctr == nnzPerProc)
            {
                // Assign rest of the current row to this segment
                work_sharing_arr[segment] = mtx.I[global_ctr] + 1;
                ++segment;
                local_ctr = 0;
                continue;
            }
            ++local_ctr;
        }
        // Set the last element to point to directly after the final row
        // (takes care of remainder rows)
        work_sharing_arr[comm_size] = mtx.I[mtx.nnz - 1] + 1;
    }
}

/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void segMtxStruct(const MtxData<VT, IT> &mtx, std::vector<IT> *local_I, std::vector<IT> *local_J, std::vector<VT> *local_vals, int *work_sharing_arr, int loop_rank)
{
    int start_idx, run_idx, finish_idx;
    int next_row;

    // Assign rows, columns, and values to process local vectors
    for (int row = work_sharing_arr[loop_rank]; row < work_sharing_arr[loop_rank + 1]; ++row)
    {
        next_row = row + 1;

        // Return the first instance of that row present in mtx.
        start_idx = get_index<IT>(mtx.I, row);

        // once we have the index of the first instance of the row,
        // we calculate the index of the first instance of the next row
        if (next_row != mtx.n_rows)
        {
            finish_idx = get_index<IT>(mtx.I, next_row);
        }
        else
        {
            // for the "last row" case, just set finish_idx to the number of non zeros in mtx
            finish_idx = mtx.nnz;
        }
        run_idx = start_idx;

        // This do-while loop will go "across the rows", basically filling the process local vectors
        // TODO: is this better than a while loop here?
        do
        {
            local_I->push_back(mtx.I[run_idx]);
            local_J->push_back(mtx.J[run_idx]);
            local_vals->push_back(mtx.values[run_idx]);
            ++run_idx;
        } while (run_idx != finish_idx);
    }
}

/**
    Description...
    @param
    @return
*/
void define_bookkeeping_type(MtxDataBookkeeping<long int> *send_bk, MPI_Datatype *bk_type)
{

    // Declare and define MPI Datatype
    int block_length_arr[2];
    MPI_Aint displ_arr_bk[2], first_address, second_address;

    MPI_Datatype type_arr[2];
    type_arr[0] = MPI_LONG;
    type_arr[1] = MPI_CXX_BOOL;
    block_length_arr[0] = 3; // using 3 int elements
    block_length_arr[1] = 2; // and 2 bool elements
    MPI_Get_address(&send_bk->n_rows, &first_address);
    MPI_Get_address(&send_bk->is_sorted, &second_address);

    displ_arr_bk[0] = (MPI_Aint)0; // calculate displacements from addresses
    displ_arr_bk[1] = MPI_Aint_diff(second_address, first_address);
    MPI_Type_create_struct(2, block_length_arr, displ_arr_bk, type_arr, bk_type);
    MPI_Type_commit(bk_type);
}

/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT, typename ST>
void seg_and_send_data(MtxData<VT, IT> &local_mtx, Config config, std::string seg_method, std::string file_name_str, int *work_sharing_arr, int my_rank, int comm_size)
{
    // Declare functions to be used locally
    // void segwork_sharing_arr(const MtxData<VT, IT>, int *, const char *, int);
    // void segMtxStruct(const MtxData<VT, IT>, std::vector<IT> *, std::vector<IT> *, std::vector<VT> *, int *, int);

    MPI_Status status_bk, status_cols, status_rows, status_vals;

    MtxDataBookkeeping<ST> send_bk, recv_bk;
    MPI_Datatype bk_type;

    define_bookkeeping_type(&send_bk, &bk_type);

    int msg_length;

    if (my_rank == 0)
    {
        // NOTE: Matrix will be read in as SORTED by default
        // Only root proc will read entire matrix
        MtxData<VT, IT> mtx = read_mtx_data<VT, IT>(file_name_str.c_str(), config.sort_matrix);

        // Segment global row pointers, and place into an array
        // int *work_sharing_arr = new int[comm_size + 1];
        seg_work_sharing_arr<VT, IT>(mtx, work_sharing_arr, seg_method, comm_size);

        // Eventhough we're iterting through the ranks, this loop is
        // (in the present implementation) executing sequentially on the root proc
        for (int loop_rank = 0; loop_rank < comm_size; ++loop_rank)
        { // NOTE: This loop assumes we're using all ranks 0 -> comm_size-1
            std::vector<IT> local_I;
            std::vector<IT> local_J;
            std::vector<VT> local_vals;

            // Assign rows, columns, and values to process local vectors
            segMtxStruct<VT, IT>(mtx, &local_I, &local_J, &local_vals, work_sharing_arr, loop_rank);

            // Count the number of rows in each processes
            int local_row_cnt = std::set<IT>(local_I.begin(), local_I.end()).size();

            // Here, we segment data for the root process
            if (loop_rank == 0)
            {
                local_mtx = {
                    local_row_cnt,
                    mtx.n_cols,
                    local_vals.size(),
                    config.sort_matrix,
                    0,          // NOTE: These "sub matricies" will (almost) never be symmetric
                    local_I, // should work as both local and global row ptr
                    local_J,
                    local_vals};
            }
            // Here, we segment and send data to another proc
            else
            {
                send_bk = {
                    local_row_cnt,
                    mtx.n_cols, // TODO: Actually constant, do dont need to send to each proc
                    local_vals.size(),
                    config.sort_matrix,
                    0};

                // First, send BK struct
                MPI_Send(&send_bk, 1, bk_type, loop_rank, 99, MPI_COMM_WORLD);

                // Next, send three arrays
                MPI_Send(&local_I[0], local_I.size(), MPI_INT, loop_rank, 42, MPI_COMM_WORLD);
                MPI_Send(&local_J[0], local_J.size(), MPI_INT, loop_rank, 43, MPI_COMM_WORLD);
                if (typeid(VT) == typeid(double))
                {
                    MPI_Send(&local_vals[0], local_vals.size(), MPI_DOUBLE, loop_rank, 44, MPI_COMM_WORLD);
                }
                else if (typeid(VT) == typeid(float))
                {
                    MPI_Send(&local_vals[0], local_vals.size(), MPI_FLOAT, loop_rank, 44, MPI_COMM_WORLD);
                }
            }
        }
        // delete[] work_sharing_arr;
    }
    else if (my_rank != 0)
    {
        // First, recieve BK struct
        MPI_Recv(&recv_bk, 1, bk_type, 0, 99, MPI_COMM_WORLD, &status_bk);

        // Next, allocate space for incoming arrays
        msg_length = recv_bk.nnz;
        IT *recv_buf_global_row_coords = new IT[msg_length];
        IT *recv_buf_col_coords = new IT[msg_length];
        VT *recv_buf_vals = new VT[msg_length];

        // Next, recieve 3 arrays that we've allocated space for on local proc
        MPI_Recv(recv_buf_global_row_coords, msg_length, MPI_INT, 0, 42, MPI_COMM_WORLD, &status_rows);
        MPI_Recv(recv_buf_col_coords, msg_length, MPI_INT, 0, 43, MPI_COMM_WORLD, &status_cols);
        if (typeid(VT) == typeid(double))
        {
            MPI_Recv(recv_buf_vals, msg_length, MPI_DOUBLE, 0, 44, MPI_COMM_WORLD, &status_vals);
        }
        else if (typeid(VT) == typeid(float))
        {
            MPI_Recv(recv_buf_vals, msg_length, MPI_FLOAT, 0, 44, MPI_COMM_WORLD, &status_vals);
        }
        // TODO: Just how bad is this?... Are we copying array -> vector?
        // TODO: is this doing what you think it is?
        std::vector<IT> global_rows_vec(recv_buf_global_row_coords, recv_buf_global_row_coords + msg_length);
        std::vector<IT> cols_vec(recv_buf_col_coords, recv_buf_col_coords + msg_length);
        std::vector<VT> vals_vec(recv_buf_vals, recv_buf_vals + msg_length);

        local_mtx = {
            recv_bk.n_rows,
            recv_bk.n_cols,
            recv_bk.nnz,
            recv_bk.is_sorted,
            recv_bk.is_symmetric,
            global_rows_vec,
            cols_vec,
            vals_vec};
    }

    // Each process exchanges it's global row ptrs for local row ptrs
    IT first_global_row = local_mtx.I[0];
    IT *global_row_coords = new IT[local_mtx.nnz];
    IT *local_row_coords = new IT[local_mtx.nnz];

    for (int nz = 0; nz < local_mtx.nnz; ++nz)
    {
        // save proc's global row ptr
        global_row_coords[nz] = local_row_coords[nz];

        // subtract first pointer from the rest, to make them process local
        local_row_coords[nz] = local_mtx.I[nz] - first_global_row;
    }

    std::vector<IT> loc_rows_vec(local_row_coords, local_row_coords + local_mtx.nnz);

    // assign local row ptrs to struct
    local_mtx.I = loc_rows_vec;

    // Broadcast work sharing array to other processes
    MPI_Bcast(work_sharing_arr,
              comm_size + 1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    // NOTE: what possibilities are there to use global row coords later?
    delete[] global_row_coords;
    delete[] local_row_coords;

    MPI_Type_free(&bk_type);
}

/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void check_if_result_valid(const char *file_name, std::vector<VT> *y_total, const std::string name, bool sort_matrix)
{
    DefaultValues<VT, IT> defaults;

    // Root proc reads all of mtx
    MtxData<VT, IT> mtx = read_mtx_data<VT, IT>(file_name, sort_matrix);

    std::vector<VT> x_total(mtx.n_cols, 0);
    // std::uninitialized_fill_n(x_total.data(), x_total.n_rows, idk.y);

    // recreate_x_total()

    // TODO: Only works since seed is same. Not flexible to swapping.
    init_std_vec_with_ptr_or_value<VT>(x_total, mtx.n_cols, nullptr, defaults.x);

    // bool is_result_valid = spmv_verify<VT, IT>(name, mtx, x_total, *y_total);

    // std::cout << result_valid << std::endl;

    // TODO: come back to validity checking later
    // if (is_result_valid)
    // {
    //     printf("Results valid.\n");
    // }
    // else
    // {
    //     printf("Results NOT valid.\n");
    // }
}

/**
    Description...
    @param
    @return
*/
template <typename VT, typename IT>
void compute_result(std::string file_name, std::string seg_method, Config config)
{
    // BenchmarkResult result;

    // TODO: bring back matrix stats
    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    // Initialize MPI variables
    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Declare struct on each process
    MtxData<VT, IT> local_mtx;

    // Allocate space for work sharing array. Is populated in seg_and_send_data function
    int work_sharing_arr[comm_size + 1] = {0};// = new IT[comm_size + 1];

    seg_and_send_data<VT, IT, ST>(local_mtx, config, seg_method, file_name, work_sharing_arr, my_rank, comm_size);

    // Each process must allocate space for total y vector
    // Will these just be the same, i.e. dont need to return anything?
    std::vector<VT> total_y(work_sharing_arr[comm_size], 0);
    std::vector<VT> result(work_sharing_arr[comm_size], 0);

    // TODO: make bench function more general/simple
    // result = bench_spmv<VT, IT>(name, config, k_entry, local_mtx, work_sharing_arr, &y_total);
    //  bench_spmv(const std::string &kernel_name,
    //            const Config &config,
    //            const Kernel::entry_t &k_entry,
    //            const MtxData<VT, IT> &mtx,
    //            const int *work_sharing_arr,
    //            std::vector<VT> *y_total, // IDK if this is const
    //            DefaultValues<VT, IT> *defaults = nullptr,
    //            const std::vector<VT> *x_in = nullptr,
    //            std::vector<VT> *y_out_opt = nullptr)


    // TODO: whats the best way to handle defaults? Just throw in config struct?
    DefaultValues<VT, IT> default_values;
    // TODO: initialize defaults and configs better
    // if (!defaults) {
    //     defaults = &default_values;
    // }

    bench_spmv_scs<VT, IT>(&config,
                                   local_mtx,
                                   work_sharing_arr, 
                                   &total_y,
                                   default_values);


    // TODO: verify results
    // if (print_proc_local_stats)
    // {
    //     print_results(print_list, name, matrix_stats, result, n_cpu_threads, print_details);
    // }
    // if (config.verify_result)
    // {
    //     // But have root proc check results, because all processes have the same y_total
    //     if (my_rank == 0)
    //     {
    //         check_if_result_valid<VT, IT>(file_name, &y_total, name, config.sort_matrix);
    //     }
    // }
}

/**
    Description...
    @param
    @return
*/
void verifyAndAssignInputs(int argc, char *argv[], std::string &file_name_str, std::string &seg_method, std::string &value_type, bool *random_init_x, Config *config){
    if (argc < 2){
        fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
            "options [defaults]: -c[%li], -s[%li], -rev[%li], -rand-x[%i], -sp/dp[%s], -seg-nnz/seg-rows[%s], -bench/solver[%s]\n",
            argv[0], config->chunk_size, config->sigma, config->n_repetitions, *random_init_x, value_type.c_str(), seg_method.c_str(), config->mode);
        exit(1);
    }

    file_name_str = argv[1];

    int args_start_index = 2;
    for (int i = args_start_index; i < argc; ++i){
        std::string arg = argv[i];
        if (arg == "-c")
        {
            config->chunk_size = atoi(argv[++i]);

            if (config->chunk_size < 1)
            {
                fprintf(stderr, "ERROR: chunk size must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-s")
        {

            config->sigma = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->sigma < 1)
            {
                fprintf(stderr, "ERROR: sigma must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-rev")
        {
            config->n_repetitions = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->n_repetitions < 1)
            {
                fprintf(stderr, "ERROR: revisions must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-rand-x")
        {
            *random_init_x = true;
        }
        else if (arg == "-dp")
        {
            value_type = "dp";
        }
        else if (arg == "-sp")
        {
            value_type = "sp";
        }
        else if (arg == "-seg-rows")
        {
            seg_method = "seg-rows";
        }
        else if (arg == "-seg-nnz")
        {
            seg_method = "seg-nnz";
        }
        else if (arg == "-bench")
        {
            config->mode = "bench";
        }
        else if (arg == "-solver")
        {
            config->mode = "solver";
        }
        else
        {
            fprintf(stderr, "ERROR: unknown argument.\n");
            exit(1);
        }
    }
    
    if (config->sigma > config->chunk_size){
            fprintf(stderr, "ERROR: sigma must be smaller than chunk size.\n");
            exit(1);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    Config config;
    std::string file_name_str{};

    // Set defaults for cl inputs
    std::string seg_method{"seg-rows"};
    // std::string kernel_to_benchmark{"csr"}; still needed?
    std::string value_type = {"dp"};
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;
    bool random_init_x = false;

    MARKER_INIT();

    verifyAndAssignInputs(argc, argv, file_name_str, seg_method, value_type, &random_init_x, &config);

    if (value_type == "sp" )
    {
        compute_result<float, int>(file_name_str, seg_method, config);
    }
    else if (value_type == "dp")
    {
        compute_result<double, int>(file_name_str, seg_method, config);
    }
    
    log("benchmarking kernel: scs end\n");

    MPI_Finalize();

    log("main end\n");

    MARKER_DEINIT();

    return 0;
}