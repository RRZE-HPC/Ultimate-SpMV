#ifndef MPI_FUNCS
#define MPI_FUNCS

#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <map>
#include <unistd.h>

#include "utilities.hpp"
// #include "classes_structs.hpp"

#include <set>

/**
    @brief Generate unique tags for communication, based on the cantor pairing function. 
    @param *send_tags : tags used in the main communication loop for Isend
    @param *recv_tags : tags used in the main communication loop for Irecv
*/
template <typename VT, typename IT>
void gen_unique_comm_tags(
    std::vector<std::vector<IT>> *send_tags,
    std::vector<std::vector<IT>> *recv_tags,
    int my_rank,
    int comm_size)
{
    for(int i = 0; i < comm_size; ++i){
        for(int j = 0; j < comm_size; ++j){
            (*send_tags)[j].push_back(cantor_pairing(j, i)); // TODO: seems too large
            (*recv_tags)[i].push_back(cantor_pairing(i, j)); // this size seems right
        }
    }
}


/**
    @brief An all to all communication routine, in which the local_x indices each process is to send is communicated
    @param *communication_send_idxs : The buffer that recieves the incoming messeage, stating which of it's local_x indicies to send
    @param *communication_recv_idxs : The buffer that is to be distributed to the other processes
    @param *send_counts_cumsum : A cumulative sum, stating the size of the incoming message
*/
template <typename VT, typename IT>
void collect_comm_idxs(
    std::vector<std::vector<IT>> *communication_send_idxs,
    std::vector<std::vector<IT>> *communication_recv_idxs,
    std::vector<IT> *send_counts_cumsum,
    int my_rank,
    int comm_size)
{
    int incoming_buf_size, outgoing_buf_size;
    int tag_send, tag_recv;

    MPI_Request recv_requests[comm_size];
    MPI_Request send_requests[comm_size];

    // TODO: avoid unnecessary comm
    for (int from_proc = 0; from_proc < comm_size; ++from_proc)
    {
        incoming_buf_size = (*send_counts_cumsum)[from_proc + 1] - (*send_counts_cumsum)[from_proc];

        tag_recv = cantor_pairing(from_proc, my_rank);

        // resize vector to recieve incoming message
        (*communication_send_idxs)[from_proc].resize(incoming_buf_size);

        MPI_Irecv(
            &((*communication_send_idxs)[from_proc])[0],
            incoming_buf_size,
            MPI_INT,
            from_proc,
            tag_recv,
            MPI_COMM_WORLD,
            &recv_requests[from_proc]
        );
    }

    for (int to_proc = 0; to_proc < comm_size; ++to_proc)
    {
        outgoing_buf_size = (*communication_recv_idxs)[to_proc].size();
        
        tag_send = cantor_pairing(my_rank, to_proc);

        MPI_Isend(
            &((*communication_recv_idxs)[to_proc])[0],
            outgoing_buf_size,
            MPI_INT,
            to_proc,
            tag_send,
            MPI_COMM_WORLD,
            &send_requests[to_proc]
        );
    }

    MPI_Waitall(comm_size, recv_requests, MPI_STATUS_IGNORE);
}

/**
    @brief Organizes the send and recv cumulative sums, which are neseccary for communication buffer offsets and sizes
    @param *send_counts_cumsum : States how many elements this process sends to all other processes
    @param *recv_counts_cumsum : States how many elements this process recieves from all other processes
*/
template <typename VT, typename IT>
void organize_cumsums(
    std::vector<IT> *send_counts_cumsum,
    std::vector<IT> *recv_counts_cumsum,
    int my_rank,
    int comm_size)
{
    std::vector<IT> all_recv_counts_cumsum(comm_size, 0);
    int *all_recv_cumsum_buf = new int[comm_size * (comm_size + 1)];
    // To recieve "comm_size" many buffers of length "comm_size + 1"

    MPI_Allgather(&(*recv_counts_cumsum)[0],
                comm_size + 1,
                MPI_INT,
                all_recv_cumsum_buf,
                comm_size + 1,
                MPI_INT,
                MPI_COMM_WORLD);

    // construct send_counts_cumsum
    int loop_rank = -1;
    int send_counts[comm_size];
    for(int i = 0; i < comm_size; ++i){
        send_counts[i] = 0;
    }
    int sending_proc = 0;
    int idx = 1;
    int sent_elems;

    for(int i = 0; i < comm_size * (comm_size + 1); ++i){

        if(i % (comm_size + 1) == 0){
            ++loop_rank;
            --idx; // don't incremement idx when in-between loop ranks
            sending_proc = 0; // reset sending_proc
            continue;
        };


        // Avoids the "last element" in the cumsum
        if(sending_proc == my_rank){
            send_counts[loop_rank] += all_recv_cumsum_buf[i] - all_recv_cumsum_buf[i - 1];
        }
        ++idx;
        ++sending_proc;
    }

    // Construct cumsum for "to send" heri
    for(int i = 1; i < comm_size + 1; ++i){
        (*send_counts_cumsum)[i] = send_counts[i - 1] + (*send_counts_cumsum)[i - 1];
    }

    delete[] all_recv_cumsum_buf;
}


/**
    @brief Organizes the send and recv cumulative sums, which are neseccary for communication buffer offsets and sizes
    @param *communication_recv_idxs : States which indices this process is to recieve from all others
    @param *recv_counts_cumsum : States how many elements this process recieves from all other processes
    @param *local_scs : pointer to local scs struct
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template <typename VT, typename IT>
void collect_local_needed_heri(
    std::vector<std::vector<IT>> *communication_recv_idxs,
    std::vector<IT> *recv_counts_cumsum,
    ScsData<VT, IT> *local_scs,
    const IT *work_sharing_arr,
    int my_rank,
    int comm_size)
{
    // std::cout << "got here 2" << std::endl;
    IT from_proc, to_proc, elem_col;
    IT needed_heri_count = 0;
    IT amnt_lhs_halo_elems = 0;

    // To remember which columns have already been accounted for
    std::unordered_set<IT> remote_elem_col_bk;
    std::vector<IT> remote_elem_idxs;
    std::vector<IT> original_col_idxs(local_scs->col_idxs.data(), local_scs->col_idxs.data() + local_scs->n_elements);

    // TODO: these sizes are too large. Should be my_rank and comm_size - my_rank
    int lhs_needed_heri_counts[comm_size];
    for(int i = 0; i < comm_size; ++i){
        lhs_needed_heri_counts[i] = 0;
    }
    int rhs_needed_heri_counts[comm_size];
    for(int i = 0; i < comm_size; ++i){
        rhs_needed_heri_counts[i] = 0;
    }
    // int lhs_needed_heri_counts[my_rank] = {0};
    // int rhs_needed_heri_counts[comm_size - my_rank] = {0};

    // COUNTING LOOP / PROC
    for (IT i = 0; i < local_scs->n_elements; ++i)
    {
        // If true, this is a remote element, and needs to be added to vector
        elem_col = local_scs->col_idxs[i];

        // TODO: what is happening now, with these "0 element" communicators?
        // if this column corresponds to a padded element, continue to next nnz
        // if(elem_col == 0 && local_scs->values[i] == 0) {continue;}

        if (elem_col < work_sharing_arr[my_rank])
        {
            remote_elem_idxs.push_back(i);
            if(remote_elem_col_bk.find(elem_col) == remote_elem_col_bk.end()){
                // if this column has not yet been seen
                for (IT j = 0; j < comm_size; ++j) //TODO: change to only go until my_rank
                {
                    if (elem_col >= work_sharing_arr[j] && elem_col < work_sharing_arr[j + 1])
                    {
                        // Remember column corresponding to remote element
                        // remote_elem_col_bk.push_back(remote_elem_col);
                        remote_elem_col_bk.insert(elem_col);

                        ((*communication_recv_idxs)[j]).push_back(elem_col - work_sharing_arr[j]);

                        ++lhs_needed_heri_counts[j]; // This array describes how many remote elements

                        break;
                    }
                }
                //if nothing found, error
            }
        }
        else if (elem_col > work_sharing_arr[my_rank + 1] - 1)
        { // i.e. if RHS remote element
            remote_elem_idxs.push_back(i);
            if(remote_elem_col_bk.find(elem_col) == remote_elem_col_bk.end()){
                for (IT j = 0; j < comm_size; ++j) // TODO: change to start at my_rank
                {
                    if (elem_col >= work_sharing_arr[j] && elem_col < work_sharing_arr[j + 1])
                    {
                        // Remember column corresponding to remote element
                        remote_elem_col_bk.insert(elem_col);

                        ((*communication_recv_idxs)[j]).push_back(elem_col - work_sharing_arr[j]);

                        ++rhs_needed_heri_counts[j];
                        break;
                    }
                }

            }
        }
        else
        { // i.e. local element
            local_scs->col_idxs[i] -= work_sharing_arr[my_rank];
        }
    }

    IT local_elem_offset = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];

    // TODO: not sure if these are the correct size. comm_size would be too big
    int lhs_cumsum_heri_counts[comm_size + 1];
    int rhs_cumsum_heri_counts[comm_size + 1];
    for(int i = 0; i < comm_size+1; ++i){
        lhs_cumsum_heri_counts[i] = 0;
        rhs_cumsum_heri_counts[i] = 0;
    }

    for(int i = 1; i < comm_size + 1; ++i){
        lhs_cumsum_heri_counts[i] = lhs_needed_heri_counts[i - 1] + lhs_cumsum_heri_counts[i - 1];
        rhs_cumsum_heri_counts[i] = rhs_needed_heri_counts[i - 1] + rhs_cumsum_heri_counts[i - 1];
    }

    int lhs_heri_ctr[comm_size]; //TODO: change to only have size my_rank
    int rhs_heri_ctr[comm_size];
    for(int i = 0; i < comm_size; ++i){
        lhs_heri_ctr[i] = 0;
        rhs_heri_ctr[i] = 0;
    }

    std::map<int, int> remote_cols;

    // ASSIGNMENT LOOP
    for (auto remote_elem_idx : remote_elem_idxs)
    {
        elem_col = original_col_idxs[remote_elem_idx];

        // if this column corresponds to a padded element, continue to next nnz
        // if(elem_col == 0 && local_scs->values[i] == 0) {continue;}

        if (elem_col < work_sharing_arr[my_rank])
        { 
            for (IT j = 0; j < comm_size; ++j)//TODO: change to only go until my_rank
            {
                if (elem_col >= work_sharing_arr[j] && elem_col < work_sharing_arr[j + 1])
                {
                    // So, on the current process, I will know from which process this 
                    // new col_idx for lhs halo element
                    
                    if(remote_cols.find(elem_col) == remote_cols.end()){
                        remote_cols[elem_col] = local_elem_offset + lhs_cumsum_heri_counts[j] + lhs_heri_ctr[j];
                        ++lhs_heri_ctr[j];
                    }

                    local_scs->col_idxs[remote_elem_idx] = remote_cols[elem_col];
                }
            }

        }
        else if (elem_col > work_sharing_arr[my_rank + 1] - 1)
        { // i.e. if RHS remote element
            // The rank of where this needed element resides is deduced from the work sharing array.
            for (IT j = 0; j < comm_size; ++j)// TODO: change to start at my_rank
            {
                if (elem_col >= work_sharing_arr[j] && elem_col < work_sharing_arr[j + 1])
                {
                    if(remote_cols.find(elem_col) == remote_cols.end()){
                        remote_cols[elem_col] = local_elem_offset + lhs_cumsum_heri_counts[my_rank] + rhs_cumsum_heri_counts[j] + rhs_heri_ctr[j];
                        ++rhs_heri_ctr[j];
                    }
                    local_scs->col_idxs[remote_elem_idx] = remote_cols[elem_col];
                }
            }
        }
    }

    // Construct recv_counts_cumsum from lhs/rhs_cumsum_heri_counts
    for(int i = 0; i < my_rank; ++i){
        (*recv_counts_cumsum)[i] = lhs_cumsum_heri_counts[i];
    }
    for(int i = 0; i < comm_size - my_rank + 1; ++i){
        (*recv_counts_cumsum)[my_rank + i] = lhs_cumsum_heri_counts[my_rank] + rhs_cumsum_heri_counts[my_rank + i];
        // i.e. dont take the leading zero from the rhs_cumsum
    }
}

/**
    Halo element communication scheme, in which values needed by local_x in order to have a valid SPMVM are sent to this process by other processes.
        In other words, we allow each process to exchange it's proc-local "remote elements" with the other respective processes.
        The current implementation first posts all non-blocking recieve calls, then the Isends are posted afterwards.
        Since this function is in the benchmark loop, need to do as little work possible, ideally.

    @brief Communicate the halo elements to and from the local x-vectors
    @param *local_context : struct containing local_scs + communication information
    @param *local_x : the x vector corresponding to this process before the halo elements are communicated to it.
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template <typename VT, typename IT>
void communicate_halo_elements(
    ScsData<VT, IT> *local_scs,
    ContextData<VT, IT> *local_context,
    std::vector<VT> *local_x,
    VT *to_send_elems[],
    const IT *work_sharing_arr,
    MPI_Request *recv_requests,
    const int nzs_size,
    MPI_Request *send_requests,
    const int nzr_size,
    const int num_local_elems,
    const int my_rank,
    const int comm_size)
{
    int outgoing_buf_size, incoming_buf_size;
    int receiving_proc, sending_proc;

    // TODO: DRY
    if (typeid(VT) == typeid(float))
    {
        for (int from_proc_idx = 0; from_proc_idx < nzs_size; ++from_proc_idx)
        {
            sending_proc = local_context->non_zero_senders[from_proc_idx];
            incoming_buf_size = local_context->recv_counts_cumsum[sending_proc + 1] - local_context->recv_counts_cumsum[sending_proc];

            MPI_Irecv(
                &(*local_x)[num_local_elems + local_context->recv_counts_cumsum[sending_proc]],
                incoming_buf_size,
                MPI_FLOAT,
                sending_proc,
                (local_context->recv_tags[sending_proc])[my_rank],
                MPI_COMM_WORLD,
                &recv_requests[from_proc_idx]
            );
        }
        for (int to_proc_idx = 0; to_proc_idx < nzr_size; ++to_proc_idx)
        {
            receiving_proc = local_context->non_zero_receivers[to_proc_idx];
            outgoing_buf_size = local_context->send_counts_cumsum[receiving_proc + 1] - local_context->send_counts_cumsum[receiving_proc];

            // for sanity
            // if(local_context->comm_send_idxs[receiving_proc].size() != outgoing_buf_size){
            //     std::cout << "Mismatched buffer lengths in communication" << std::endl;
            //     exit(1);
            // }

            // Move non-contiguous data to a contiguous buffer for communication
            for(int i = 0; i < outgoing_buf_size; ++i){
                (to_send_elems[to_proc_idx])[i] = (*local_x)[  local_scs->old_to_new_idx[(local_context->comm_send_idxs[receiving_proc])[i]]  ];
            }

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
    }
    else if (typeid(VT) == typeid(double)){
        for (int from_proc_idx = 0; from_proc_idx < nzs_size; ++from_proc_idx)
        {
            sending_proc = local_context->non_zero_senders[from_proc_idx];
            incoming_buf_size = local_context->recv_counts_cumsum[sending_proc + 1] - local_context->recv_counts_cumsum[sending_proc];

            MPI_Irecv(
                &(*local_x)[num_local_elems + local_context->recv_counts_cumsum[sending_proc]],
                incoming_buf_size,
                MPI_DOUBLE,
                sending_proc,
                (local_context->recv_tags[sending_proc])[my_rank],
                MPI_COMM_WORLD,
                &recv_requests[from_proc_idx]
            );
        }

        // for (int to_proc = 0; to_proc < comm_size; ++to_proc)
        // for(auto to_proc : local_context->non_zero_receivers)
        for (int to_proc_idx = 0; to_proc_idx < nzr_size; ++to_proc_idx)
        {
            receiving_proc = local_context->non_zero_receivers[to_proc_idx];
            outgoing_buf_size = local_context->send_counts_cumsum[receiving_proc + 1] - local_context->send_counts_cumsum[receiving_proc];

            // for sanity
            // if(local_context->comm_send_idxs[receiving_proc].size() != outgoing_buf_size){
            //     std::cout << "Mismatched buffer lengths in communication" << std::endl;
            //     exit(1);
            // }

            // Move non-contiguous data to a contiguous buffer for communication
            for(int i = 0; i < outgoing_buf_size; ++i){
                (to_send_elems[to_proc_idx])[i] = (*local_x)[  local_scs->old_to_new_idx[(local_context->comm_send_idxs[receiving_proc])[i]]  ];
            }

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
    }

    MPI_Waitall(nzr_size, send_requests, MPI_STATUS_IGNORE);
    MPI_Waitall(nzs_size, recv_requests, MPI_STATUS_IGNORE);
}

/**
    @brief Partition the rows of the mtx structure, so that work is disributed (somewhat) evenly. The two options "seg-rows"
        and "seg-nnz" are there in an effort to have multiple load balancing techniques the segment work between processes.
    @param *total_mtx : data structure that was populated by the matrix market format reader
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
*/
template <typename VT, typename IT>
void seg_work_sharing_arr(
    const MtxData<VT, IT> *total_mtx,
    IT *work_sharing_arr,
    const std::string *seg_method,
    const IT comm_size,
    int my_rank)
{
    work_sharing_arr[0] = 0;

    IT segment;

    if ("seg-rows" == *seg_method)
    {
        IT rowsPerProc;

        // Evenly split the number of rows
        rowsPerProc = total_mtx->n_rows / comm_size;

        // Segment rows to work on via. array
        for (segment = 1; segment < comm_size + 1; ++segment)
        {
            // Can only do this because of "constant sized" segments
            work_sharing_arr[segment] = segment * rowsPerProc;
            if (segment == comm_size)
            {
                // Set the last element to point to directly after the final row
                // (takes care of remainder rows)
                work_sharing_arr[comm_size] = total_mtx->I[total_mtx->nnz - 1] + 1;
            }
        }
    }
    else if ("seg-nnz" == *seg_method)
    {
        IT nnzPerProc; //, remainderNnz;

        // Split the number of rows based on non zeros
        nnzPerProc = total_mtx->nnz / comm_size;

        IT global_ctr, local_ctr;
        segment = 1;
        local_ctr = 0;

        // Segment rows to work on via. array
        for (global_ctr = 0; global_ctr < total_mtx->nnz; ++global_ctr)
        {
            if (local_ctr == nnzPerProc)
            {
                // Assign rest of the current row to this segment
                work_sharing_arr[segment] = total_mtx->I[global_ctr] + 1;
                ++segment;
                local_ctr = 0;
                continue;
            }
            ++local_ctr;
        }
        // Set the last element to point to directly after the final row
        // (takes care of remainder rows)
        work_sharing_arr[comm_size] = total_mtx->I[total_mtx->nnz - 1] + 1;
    }

    // TODO: What other cases are there to protect against?
    // Protect against edge case where last process gets no work
    if(work_sharing_arr[comm_size - 1] == work_sharing_arr[comm_size]){
        for(IT loop_rank = 1; loop_rank < comm_size; ++loop_rank){
            work_sharing_arr[loop_rank] -= 1;
        }
    }

    // Print work sharing array. Nice for debugging
    // printf("Work sharing array: ");
    // for(int i = 0; i < comm_size + 1; ++i){
    //     printf("%i, ", work_sharing_arr[i]);
    // }
    // printf("\n");
}

/**
    Collect row idxs, col idxs, and corresponding values from the original mtx structure in order to distribute to the other processes.
        The idea is to reconstruct into local mtx structures on each process after communication.

    @brief Collect indices into local vectors
    @param *total_mtx : data structure that was populated by the matrix market format reader mtx_reader.h
    @param *local_I : pointer to the vector to contain the row idx data, taken from original mtx struct
    @param *local_J : pointer to the vector to contain the col idx data, taken from original mtx struct
    @param *local_vals : pointer to the vector to contain the value data, taken from original mtx struct
    @param *work_sharing_arr : the array describing the partitioning of the rows of the global mtx struct
    @param *loop_rank : the current rank that data is being segmented for
*/
template <typename VT, typename IT>
void seg_mtx_struct(
    const MtxData<VT, IT> *total_mtx,
    std::vector<IT> *local_I,
    std::vector<IT> *local_J,
    std::vector<VT> *local_vals,
    const IT *work_sharing_arr,
    const IT loop_rank)
{
    IT start_idx, run_idx, finish_idx;
    IT next_row;

    // Return the first instance of that row present in mtx.
    start_idx = get_index<IT>(total_mtx->I, work_sharing_arr[loop_rank]);

    // once we have the index of the first instance of the row,
    // we calculate the index of the first instance of the next row
    if (work_sharing_arr[loop_rank + 1] != total_mtx->n_rows)
    {
        finish_idx = get_index<IT>(total_mtx->I, work_sharing_arr[loop_rank + 1]);
    }
    else
    {
        // for the "last row" case, just set finish_idx to the number of non zeros in mtx
        finish_idx = total_mtx->nnz;
    }
    run_idx = start_idx;

    // This do-while loop will go "across the rows", basically filling the process local vectors
    do
    {
        local_I->push_back(total_mtx->I[run_idx]);
        local_J->push_back(total_mtx->J[run_idx]);
        local_vals->push_back(total_mtx->values[run_idx]);
        ++run_idx;
    } while (run_idx != finish_idx);
}

/**
    @brief Create custom bookkeeping datatype, in order to more easily keep track of meta data.
    @param *send_bk : an instance of bk_type, for the data sent from root to other processes
    @param *bk_type : a constructed datatype for mpi to keep track of bookkeeping data for the local mtx structure
*/
template <typename IT>
void define_bookkeeping_type(
    MtxDataBookkeeping<ST> *send_bk,
    MPI_Datatype *bk_type)
{
    // Declare and define MPI Datatype
    IT block_length_arr[2];
    MPI_Aint displ_arr_bk[2], first_address, second_address;

    MPI_Datatype type_arr[2];
    type_arr[0] = MPI_LONG;
    type_arr[1] = MPI_CXX_BOOL;
    block_length_arr[0] = 3; // using 3 IT elements
    block_length_arr[1] = 2; // and 2 bool elements
    MPI_Get_address(&send_bk->n_rows, &first_address);
    MPI_Get_address(&send_bk->is_sorted, &second_address);

    displ_arr_bk[0] = (MPI_Aint)0; // calculate displacements from addresses
    displ_arr_bk[1] = MPI_Aint_diff(second_address, first_address);
    MPI_Type_create_struct(2, block_length_arr, displ_arr_bk, type_arr, bk_type);
    MPI_Type_commit(bk_type);
}

/** 
    @brief Initialize total_mtx, segment and send this to local_mtx, convert to local_scs format, init comm information
    @param *local_scs : pointer to local scs struct
    @param *local_context : struct containing local_scs + communication information
    @param *total_mtx : complete mtx struct
    @param *config : struct to initialze default values and user input
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template<typename VT, typename IT>
void mpi_init_local_structs(
    ScsData<VT, IT> *local_scs,
    ContextData<VT, IT> *local_context,
    MtxData<VT, IT> *total_mtx,
    Config *config, // shouldn't this be const?
    const std::string *seg_method,
    IT *work_sharing_arr,
    int my_rank,
    int comm_size)
{
    MtxData<VT, IT> local_mtx;

    MPI_Status status_bk, status_cols, status_rows, status_vals;

    MtxDataBookkeeping<ST> send_bk, recv_bk;
    MPI_Datatype bk_type;

    define_bookkeeping_type<IT>(&send_bk, &bk_type);

    IT msg_length;

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Segmenting and sending work to other processes.\n");}
#endif

    // Only split up work if there are more than 1 processes
    // if(comm_size > 1){ ??
    if (my_rank == 0){
        // Segment global row pointers, and place into an array
        seg_work_sharing_arr<VT, IT>(total_mtx, work_sharing_arr, seg_method, comm_size, my_rank);

        // Eventhough we're iterting through the ranks, this loop is
        // (in the present implementation) executing sequentially on the root proc
        // STRONG CONTENDER FOR PRAGMA PARALELL
        for (IT loop_rank = 0; loop_rank < comm_size; ++loop_rank)
        { // NOTE: This loop assumes we're using all ranks 0 -> comm_size-1

            std::vector<IT> local_I;
            std::vector<IT> local_J;
            std::vector<VT> local_vals; // an attempt to not have so much resizing in seg_mtx_struct

            // Assign rows, columns, and values to process local vectors
            seg_mtx_struct<VT, IT>(total_mtx, &local_I, &local_J, &local_vals, work_sharing_arr, loop_rank);

            // Give the total nnz count to root process
            local_context->total_nnz = total_mtx->nnz;

            // Count the number of rows in each processes
            IT local_row_cnt = std::set<IT>(local_I.begin(), local_I.end()).size();

            // Here, we segment data for the root process
            if (loop_rank == 0)
            {
                local_mtx.n_rows = local_row_cnt;
                local_mtx.n_cols = total_mtx->n_cols;
                local_mtx.nnz = local_vals.size();
                local_mtx.is_sorted = config->sort_matrix;
                local_mtx.is_symmetric = 0; // NOTE: These "sub matricies" will (almost) never be symmetric
                local_mtx.I = local_I;      // should work as both local and global row ptr
                local_mtx.J = local_J;
                local_mtx.values = local_vals;
            }
            // Here, we segment and send data to another proc
            else
            {
                send_bk = {
                    local_row_cnt,
                    total_mtx->n_cols, // TODO: Actually constant, do dont need to send to each proc
                    static_cast<long>( local_vals.size() ),
                    static_cast<bool>( config->sort_matrix ),
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
    }
    else if (my_rank != 0)
    {
        // TODO: should these be blocking?
        // First, recieve BK struct
        MPI_Recv(&recv_bk, 1, bk_type, 0, 99, MPI_COMM_WORLD, &status_bk);

        // Next, allocate space for incoming arrays
        // TODO: should these be on the heap?
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
        std::vector<IT> global_rows_vec(recv_buf_global_row_coords, recv_buf_global_row_coords + msg_length);
        std::vector<IT> cols_vec(recv_buf_col_coords, recv_buf_col_coords + msg_length);
        std::vector<VT> vals_vec(recv_buf_vals, recv_buf_vals + msg_length);

        local_mtx.n_rows = recv_bk.n_rows;
        local_mtx.n_cols = recv_bk.n_cols;
        local_mtx.nnz = recv_bk.nnz;
        local_mtx.is_sorted = recv_bk.is_sorted;
        local_mtx.is_symmetric = recv_bk.is_symmetric;
        local_mtx.I = global_rows_vec;
        local_mtx.J = cols_vec;
        local_mtx.values = vals_vec;
        
        delete[] recv_buf_global_row_coords;
        delete[] recv_buf_col_coords;
        delete[] recv_buf_vals;
    }

    std::vector<IT> local_row_coords(local_mtx.nnz, 0);

    for (IT i = 0; i < local_mtx.nnz; ++i)
    {
        // subtract first pointer from the rest, to make them "process local"
        local_row_coords[i] = local_mtx.I[i] - local_mtx.I[0];
    }

    // assign local row ptrs to struct
    local_mtx.I = local_row_coords;

    // just let every process know the total number of nnz.
    // TODO: necessary?
    MPI_Bcast(&(local_context->total_nnz),
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD);

    // Broadcast work sharing array to other processes
    MPI_Bcast(work_sharing_arr,
              comm_size + 1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

#ifdef DEBUG_MODE
    if(my_rank == 0){printf("Converting COO matrix to SELL-C-SIG and permuting.\n");}
#endif
    // convert local_mtx to local_scs and permute rows (if applicable)
    convert_to_scs<VT, IT>(&local_mtx, config->chunk_size, config->sigma, local_scs, work_sharing_arr, my_rank);

    // permute columns with additional routine
    // std::vector<int> non_perm(local_scs->n_rows);
    // std::iota(std::begin(non_perm), std::end(non_perm), 0); // Fill with 0, 1, ..., scs->n_rows.
    // local_scs->permute(&(local_scs->old_to_new_idx)[0], &(local_scs->new_to_old_idx)[0]);

    MPI_Type_free(&bk_type);

    // TODO: is an array of vectors better?
    // Vector of vecs, Keep track of which remote columns come from which processes
    std::vector<std::vector<IT>> communication_recv_idxs;
    std::vector<std::vector<IT>> communication_send_idxs;

    // Fill vectors with empty vectors, representing places to store "to_send_idxs"
    for(int i = 0; i < comm_size; ++i){
        communication_recv_idxs.push_back(std::vector<IT>());
        communication_send_idxs.push_back(std::vector<IT>());
    }

    std::vector<IT> recv_counts_cumsum(comm_size + 1, 0);
    std::vector<IT> send_counts_cumsum(comm_size + 1, 0);

    collect_local_needed_heri<VT, IT>(&communication_recv_idxs, &recv_counts_cumsum, local_scs, work_sharing_arr, my_rank, comm_size);

    organize_cumsums<VT, IT>(&send_counts_cumsum, &recv_counts_cumsum, my_rank, comm_size);

    collect_comm_idxs<VT, IT>(&communication_send_idxs, &communication_recv_idxs, &send_counts_cumsum, my_rank, comm_size);

    // Necessary for "high comm" instances. Just leave it.
    MPI_Barrier(MPI_COMM_WORLD);

    // Determine which of the other processes THIS processes send anything to
    std::vector<IT> non_zero_receivers;
    std::vector<IT> non_zero_senders;

    for(int i = 0; i < communication_send_idxs.size(); ++i){
        if(communication_send_idxs[i].size() > 0){
           non_zero_receivers.push_back(i); 
        }
    }
    for(int i = 0; i < communication_recv_idxs.size(); ++i){
        if(communication_recv_idxs[i].size() > 0){
           non_zero_senders.push_back(i); 
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);


    std::vector<std::vector<IT>> send_tags;
    std::vector<std::vector<IT>> recv_tags;

    for(int i = 0; i < comm_size; ++i){
        send_tags.push_back(std::vector<IT>());
        recv_tags.push_back(std::vector<IT>());
    }

    gen_unique_comm_tags<VT, IT>(&send_tags, &recv_tags, my_rank, comm_size);

    local_context->comm_send_idxs = communication_send_idxs;
    local_context->comm_recv_idxs = communication_recv_idxs;
    local_context->non_zero_receivers = non_zero_receivers;
    local_context->non_zero_senders = non_zero_senders;
    local_context->send_tags = send_tags;
    local_context->recv_tags = recv_tags;
    local_context->recv_counts_cumsum = recv_counts_cumsum;
    local_context->send_counts_cumsum = send_counts_cumsum;
    local_context->num_local_rows = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
    local_context->scs_padding = (IT)(local_scs->n_rows_padded - local_scs->n_rows);
}
#endif