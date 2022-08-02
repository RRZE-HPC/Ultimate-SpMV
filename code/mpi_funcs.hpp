#ifndef MPI_FUNCS
#define MPI_FUNCS

// #include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "classes_structs.hpp"

#include <set>

template <typename VT, typename IT>
void new_gen_halo_hms(
    ScsData<VT, IT> *local_scs,
    std::unordered_map<int, int> *lhs_halo_hm,
    std::unordered_map<int, int> *rhs_halo_hm,
    std::vector<IT> *original_col_idxs,
    const IT *work_sharing_arr,
    const int *my_rank,
    const int *comm_size
){
    
    IT remote_elem_count = 0;
    IT amnt_lhs_remote_elems = 0;
    IT amnt_rhs_remote_elems = 0;
    IT exists_nz_elem = 0;
    std::vector<IT> col_all_inst_idx;

    // std::vector<IT> original_col_idxs(local_scs->col_idxs.data(), local_scs->col_idxs.data() + local_scs->n_elements);
    std::unordered_set<IT> original_col_idxs_uset(original_col_idxs->begin(), original_col_idxs->end());
    IT amnt_local_elems = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];


    int lhs_halo_col_ctr = 0;
    // Proc 0 will never have LHS remote elements
    if(*my_rank != 0)
    {
        // start at the "left wall" of the matrix, and iterate columns until we reach local elements
        for (int col = 0; col < work_sharing_arr[*my_rank]; ++col)
        {
            if(original_col_idxs_uset.find(col) != original_col_idxs_uset.end()){
             // if there exists ANY element in this column...

                col_all_inst_idx = find_items<IT>(*original_col_idxs, col);

                // TODO: make faster
                // First need to know if there are any non zero elements with nonzero value 
                // exists_nz_elem = 0;
                for(auto idx : col_all_inst_idx)
                {
                    if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0)
                    {
                            // (*lhs_halo_hm)[original_col_idxs[idx]] = amnt_local_elems + lhs_halo_col_ctr;
                            // ++lhs_halo_col_ctr;
                            // break;
                        exists_nz_elem = 1; // is this harness loop necessary?
                        break;
                    }
                }

                // Then, is a nonzero element does exist in this column, assign to hm (this column idx : new column idx)
                if(exists_nz_elem)
                { // and if this element is not padding
                    // #pragma omp parallel for
                    // for(auto idx : col_all_inst_idx)
                    // {
                    //     if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0) //second condition needed?
                    //     {
                            (*lhs_halo_hm)[col] = amnt_local_elems + lhs_halo_col_ctr;
                            ++lhs_halo_col_ctr;
                            // break;
                            // (*lhs_halo_hm)[idx] = amnt_local_elems + lhs_halo_col_ctr;
                            // local_scs->col_idxs[idx] = amnt_local_elems + lhs_halo_col_ctr;
                    //     }
                    // }
                // If at least one nonzero element exists in this column, increment counter
                // ++lhs_halo_col_ctr;
                }
            }
        }   
    }


    int rhs_halo_col_ctr = 0;
    // Last proc will never have RHS remote elements
    if(*my_rank != (*comm_size - 1)){
        // start at the "left wall" of the matrix, and iterate columns until we reach local elements
        for (int col = work_sharing_arr[*my_rank + 1]; col < work_sharing_arr[*comm_size]; ++col)
        {
            if(original_col_idxs_uset.find(col) != original_col_idxs_uset.end()){
             // if there exists ANY element in this column...

                col_all_inst_idx = find_items<IT>(*original_col_idxs, col);
                // exists_nz_elem = 0;
                for(auto idx : col_all_inst_idx)
                {
                    if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0)
                    {
                        exists_nz_elem = 1;
                        break;
                    }
                }

                if(exists_nz_elem)
                { // and if this element is not padding
                    // #pragma omp parallel for
                    // for(auto idx : col_all_inst_idx)
                    // {
                    //     if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0)
                    //     {
                            (*rhs_halo_hm)[col] = amnt_local_elems + lhs_halo_col_ctr + rhs_halo_col_ctr;
                            ++rhs_halo_col_ctr;
                            // break;
                            // (*rhs_halo_hm)[idx] = amnt_local_elems + lhs_halo_col_ctr + rhs_halo_col_ctr;
                            // local_scs->col_idxs[idx] = amnt_local_elems + lhs_halo_col_ctr + rhs_halo_col_ctr;
                    //     }
                    // }
                    // If at least one element exists in this column, increment counter
                    // ++rhs_halo_col_ctr;
                }
            }
        }   
    }
}

template <typename VT, typename IT>
void original_gen_halo_hms(
    ScsData<VT, IT> *local_scs,
    std::unordered_map<int, int> *lhs_halo_hm,
    std::unordered_map<int, int> *rhs_halo_hm,
    std::vector<IT> *original_col_idxs,
    const IT *work_sharing_arr,
    const int *my_rank,
    const int *comm_size
){
    
    IT remote_elem_count = 0;
    IT amnt_lhs_remote_elems = 0;
    IT amnt_rhs_remote_elems = 0;
    IT exists_nz_elem = 0;
    std::vector<IT> col_all_inst_idx;
    IT amnt_local_elems = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];

    int lhs_halo_col_ctr = 0;
    // Proc 0 will never have LHS remote elements
    if(*my_rank != 0)
    {
        // start at the "left wall" of the matrix, and iterate columns until we reach local elements
        for (int col = 0; col < work_sharing_arr[*my_rank]; ++col)
        {
            if ((std::find(original_col_idxs->begin(), original_col_idxs->end(), col) != original_col_idxs->end()))
            { // if there exists ANY element in this column...

                col_all_inst_idx = find_items<IT>(*original_col_idxs, col);

                for(auto idx : col_all_inst_idx)
                {
                    if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0)
                    {
                        exists_nz_elem = 1; // is this harness loop necessary?
                        break;
                    }
                }

                if(exists_nz_elem)
                { // and if this element is not padding
                    // #pragma omp parallel for
                    for(auto idx : col_all_inst_idx)
                    {
                        if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0) //second condition needed?
                        {
                            (*lhs_halo_hm)[idx] = amnt_local_elems + lhs_halo_col_ctr;
                            // local_scs->col_idxs[idx] = amnt_local_elems + lhs_halo_col_ctr;
                        }
                    }
                // If at least one nonzero element exists in this column, increment counter
                ++lhs_halo_col_ctr;
                }
            }
        }   
    }


    int rhs_halo_col_ctr = 0;
    // Last proc will never have RHS remote elements
    if(*my_rank != (*comm_size - 1)){
        // start at the "left wall" of the matrix, and iterate columns until we reach local elements
        for (int col = work_sharing_arr[*my_rank + 1]; col < work_sharing_arr[*comm_size]; ++col)
        {
            if ((std::find(original_col_idxs->begin(), original_col_idxs->end(), col) != original_col_idxs->end()))
            { // if there exists ANY element in this column...
                col_all_inst_idx = find_items<IT>(*original_col_idxs, col);

                for(auto idx : col_all_inst_idx)
                {
                    if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0)
                    {
                        exists_nz_elem = 1;
                        break;
                    }
                }

                if(exists_nz_elem)
                { // and if this element is not padding
                    // #pragma omp parallel for
                    for(auto idx : col_all_inst_idx)
                    {
                        if((*original_col_idxs)[idx] == col && local_scs->values[idx] != 0)
                        {
                            (*rhs_halo_hm)[idx] = amnt_local_elems + lhs_halo_col_ctr + rhs_halo_col_ctr;
                            // local_scs->col_idxs[idx] = amnt_local_elems + lhs_halo_col_ctr + rhs_halo_col_ctr;
                        }
                    }
                    // If at least one element exists in this column, increment counter
                    ++rhs_halo_col_ctr;
                }
            }
        }   
    }
}

/**
    Collect the row indicies of the halo elements needed for THIS process, which is used to generate a valid local_x to perform the SPMVM.
        These are organized/encoded as 3-tuples in an array, of the form (proc_to, proc_from, global_row_idx). The global_row_idx
        refers to the "global" x vector, this will be adjusted (localized) later when said element is "retrieved".

    @brief Collect the row indicies of the halo elements needed for this process
    @param *local_needed_heri : pointer to vector that contains encoded 3-tuples of the form (proc_to, proc_from, global_row_idx)
    @param *local_scs : pointer to local scs struct
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template <typename VT, typename IT>
void collect_local_needed_heri(
    std::vector<IT> *local_needed_heri,
    ScsData<VT, IT> *local_scs,
    const IT *work_sharing_arr,
    const int *my_rank,
    const int *comm_size)
{
    IT from_proc, to_proc;
    IT total_x_row_idx, remote_elem_candidate_col, elem_col, remote_elem_col;
    IT needed_heri_count = 0;
    IT amnt_lhs_halo_elems = 0;

    // To remember which columns have already been accounted for
    std::vector<int> remote_elem_col_bk;
    std::tuple<IT, IT, IT> heri_tuple;
    std::tuple<IT, IT, IT> inner_tuple;
    std::tuple<IT, std::tuple<IT, IT, IT>> unordered_heri_tuple;
    std::vector<std::tuple<IT, std::tuple<IT, IT, IT>>> unordered_heri_tuples_vec;
    std::vector<IT> original_col_idxs(local_scs->col_idxs.data(), local_scs->col_idxs.data() + local_scs->n_elements);

    std::unordered_map<int, int> lhs_halo_hm;
    std::unordered_map<int, int> rhs_halo_hm;
    // original_gen_halo_hms<VT, IT>(local_scs, &lhs_halo_hm, &rhs_halo_hm, &original_col_idxs, work_sharing_arr, my_rank, comm_size);
    new_gen_halo_hms<VT, IT>(local_scs, &lhs_halo_hm, &rhs_halo_hm, &original_col_idxs, work_sharing_arr, my_rank, comm_size);


    for (IT i = 0; i < local_scs->n_elements; ++i)
    {
        // If true, this is a remote element, and needs to be added to vector
        elem_col = local_scs->col_idxs[i];

        // if this column corresponds to a padded element, continue to next nnz
        if(elem_col == 0 && local_scs->values[i] == 0) {continue;}

        if (elem_col < work_sharing_arr[*my_rank])
        { // if LHS remote element
            if (!(std::find(remote_elem_col_bk.begin(), remote_elem_col_bk.end(), elem_col) != remote_elem_col_bk.end()))
            { // if this column has not yet been seen
                remote_elem_col = elem_col;
                // First, determine which rank this needed element will come from
                // The rank of where this needed element resides is deduced from the work sharing array.
                for (IT j = 0; j < *comm_size; ++j)
                {
                    if (remote_elem_col >= work_sharing_arr[j] && remote_elem_col < work_sharing_arr[j + 1])
                    {
                        // Remember column corresponding to remote element
                        remote_elem_col_bk.push_back(remote_elem_col);

                        // Just to make process more clear
                        from_proc = j;
                        to_proc = *my_rank;

                        // The heri tuple, which is then used inside another tuple for ordering later
                        heri_tuple = std::make_tuple(to_proc, from_proc, remote_elem_col);

                        unordered_heri_tuple = std::make_tuple(remote_elem_col, heri_tuple);

                        unordered_heri_tuples_vec.push_back(unordered_heri_tuple);

                        ++needed_heri_count;
                        break;
                    }
                }
            }
            // local_scs->col_idxs[i] = lhs_halo_hm[i];
            local_scs->col_idxs[i] = lhs_halo_hm[original_col_idxs[i]];

        }
        else if (elem_col > work_sharing_arr[*my_rank + 1] - 1)
        { // i.e. if RHS remote element
            if (!(std::find(remote_elem_col_bk.begin(), remote_elem_col_bk.end(), elem_col) != remote_elem_col_bk.end()))
            { // if this column has not yet been seen
                remote_elem_col = elem_col;
                // The rank of where this needed element resides is deduced from the work sharing array.
                for (IT j = 0; j < *comm_size; ++j)
                {
                    if (remote_elem_col >= work_sharing_arr[j] && remote_elem_col < work_sharing_arr[j + 1])
                    {
                        // Remember column corresponding to remote element
                        remote_elem_col_bk.push_back(remote_elem_col);

                        // Just to make process more clear
                        from_proc = j;
                        to_proc = *my_rank;

                        // The heri tuple, which is then used inside another tuple for ordering later
                        heri_tuple = std::make_tuple(to_proc, from_proc, remote_elem_col);

                        unordered_heri_tuple = std::make_tuple(remote_elem_col, heri_tuple);

                        unordered_heri_tuples_vec.push_back(unordered_heri_tuple);

                        ++needed_heri_count;
                        break;
                    }
                }

            }
            // local_scs->col_idxs[i] = rhs_halo_hm[i];
            local_scs->col_idxs[i] = rhs_halo_hm[original_col_idxs[i]];
        }
        else
        { // i.e. local element
            local_scs->col_idxs[i] -= work_sharing_arr[*my_rank];
        }
    }

    std::sort(unordered_heri_tuples_vec.begin(), unordered_heri_tuples_vec.end());
    // MPI_Barrier(MPI_COMM_WORLD);

    for(auto& tuple : unordered_heri_tuples_vec){
        inner_tuple = std::get<1>(tuple);
        local_needed_heri->push_back(std::get<0>(inner_tuple));
        local_needed_heri->push_back(std::get<1>(inner_tuple));
        local_needed_heri->push_back(std::get<2>(inner_tuple));
    }

    // IT remote_elem_count = 0;
    // IT amnt_lhs_remote_elems = 0;
    // IT amnt_rhs_remote_elems = 0;
    // IT test_rank = 1, show_steps = 0;
    // IT exists_nz_elem = 0;
    // IT idx_ctr;
    // IT elem_idx;




    // if(show_steps){
    //     if(*my_rank == test_rank){
    //         std::cout << "column indices BEFORE adjustment: " << std::endl;
    //         for(int idx = 0; idx < local_scs->n_elements; ++idx){
    //             std::cout << local_scs->col_idxs[idx] << std::endl;
    //         }
    //         printf("\n");
    //     }
    // }

    // #pragma omp parallel for
    // for (IT i = 0; i < local_scs->n_elements; ++i)
    // {
    //     // NOTE: Is this enough? Or do we also need local_scs->col_idxs[i] != 0?
    //     if (local_scs->values[i] != 0) // ignore padding
    //     {
    //         if ((local_scs->col_idxs[i] >= work_sharing_arr[*my_rank]) && local_scs->col_idxs[i] < work_sharing_arr[*my_rank + 1])
    //         { // i.e. if local
    //             local_scs->col_idxs[i] -= work_sharing_arr[*my_rank];
    //         }
    //     }
    // }


    // if(show_steps){
    //     if(*my_rank == test_rank){
    //         std::cout << "column indices AFTER adjustment: " << std::endl;
    //         for(int idx = 0; idx < local_scs->n_elements; ++idx){
    //             std::cout << local_scs->col_idxs[idx] << std::endl;
    //         }
    //         printf("\n");
    //     }
    // }


}

/**
    @brief By using the global_needed_heri array, we can push back all the heri this process will need to send onto to_send_heri.
    @param *to_send_heri : heri to send from this process
    @param *local_needed_heri : the heri needed by this process
    @param *global_needed_heri : array containing all the needed heri tuple encodings
*/
template <typename IT>
void collect_to_send_heri(
    std::vector<IT> *to_send_heri,
    const std::vector<IT> *local_needed_heri,
    IT *global_needed_heri,
    const int *my_rank,
    const int *comm_size)
{
    std::tuple<IT, IT, IT> heri_tuple;
    std::tuple<IT, IT, IT> inner_tuple;
    std::tuple<IT, std::tuple<IT, IT, IT>> unordered_heri_tuple;
    std::vector<std::tuple<IT, std::tuple<IT, IT, IT>>> unordered_heri_tuples_vec;
    IT from_proc, to_proc, remote_elem_col;


    // NOTE: Do these need to be on the heap?
    IT all_local_needed_heri_sizes[*comm_size];
    IT global_needed_heri_displ_arr[*comm_size];

    for (IT i = 0; i < *comm_size; ++i)
    {
        all_local_needed_heri_sizes[i] = IT{};
        global_needed_heri_displ_arr[i] = IT{};
    }

    IT local_needed_heri_size = local_needed_heri->size();

    // First, gather the sizes of messages for the Allgatherv later
    MPI_Allgather(&local_needed_heri_size,
                  1,
                  MPI_INT,
                  all_local_needed_heri_sizes, // should be an array of length "comm size"
                  1,
                  MPI_INT,
                  MPI_COMM_WORLD);

    IT intermediate_size = IT{};

    for (IT i = 0; i < *comm_size; ++i)
    {
        global_needed_heri_displ_arr[i] = intermediate_size;
        intermediate_size += all_local_needed_heri_sizes[i];
    }

    // Collect to each process, the entire view of needed elements
    MPI_Allgatherv(&(*local_needed_heri)[0],
                   local_needed_heri_size,
                   MPI_INT,
                   global_needed_heri,
                   all_local_needed_heri_sizes,
                   global_needed_heri_displ_arr,
                   MPI_INT,
                   MPI_COMM_WORLD);

    // Finally, sort the global_needed_heri into "to_send_heri". Encoded 3-tuples as array
    for (IT from_proc_idx = 1; from_proc_idx < intermediate_size; from_proc_idx += 3)
    {
        if (global_needed_heri[from_proc_idx] == *my_rank)
        {// i.e. if were sending FROM this rank

            // Just to make process more clear
            remote_elem_col = global_needed_heri[from_proc_idx + 1];
            to_proc = global_needed_heri[from_proc_idx - 1];
            from_proc = *my_rank;

            // The heri tuple, which is then used inside another tuple for ordering later
            heri_tuple = std::make_tuple(to_proc, from_proc, remote_elem_col);

            unordered_heri_tuple = std::make_tuple(remote_elem_col, heri_tuple);

            unordered_heri_tuples_vec.push_back(unordered_heri_tuple);
        }
    }

    std::sort(unordered_heri_tuples_vec.begin(), unordered_heri_tuples_vec.end());

    for(auto& tuple : unordered_heri_tuples_vec){
        inner_tuple = std::get<1>(tuple);
        to_send_heri->push_back(std::get<0>(inner_tuple));
        to_send_heri->push_back(std::get<1>(inner_tuple));
        to_send_heri->push_back(std::get<2>(inner_tuple));
    }
}

/**
    @brief Pre-calculate the shift for each given proc_to and proc_from, to look it up quickly in the benchmark loop later. Used in the tag-generation scheme in halo communication.
    @param *global_needed_heri : array containing all the needed heri tuple encodings
    @param *global_needed_heri_size : the size of heri needed across all processes
    @param *shift_arr : Essentially a cumulative from top -> bottom of incedence_arr. The row idx is the "from_proc",
        the column is the "to_proc", and the element is the shift after the local element index to make for the incoming halo elements
    @param *incidence_arr : uses the global_needed_heri to keep track of which processes communicate with which.
*/
template <typename IT>
void calc_heri_shifts(
    const IT *global_needed_heri,
    const IT *global_needed_heri_size,
    std::vector<IT> *shift_arr,
    std::vector<IT> *incidence_arr,
    const int *comm_size)
{
    IT current_row, current_col, from_proc, to_proc;

    for (IT i = 0; i < *global_needed_heri_size; i += 3)
    {
        to_proc = global_needed_heri[i];
        from_proc = global_needed_heri[i + 1];

        (*incidence_arr)[*comm_size * from_proc + to_proc] += 1;
    }

    for (IT row = 1; row < *comm_size; ++row)
    {
        IT rows_remaining = *comm_size - row;
        for (IT i = 0; i < rows_remaining; ++i)
        {
            for (IT col = 0; col < *comm_size; ++col)
            {
                (*shift_arr)[*comm_size * (row + i) + col] += (*incidence_arr)[*comm_size * (row - 1) + col];
            }
        }
    }
}

/**
    Halo element communication scheme, in which values needed by local_x in order to have a valid SPMVM are sent to this process by other processes.
        In other words, we allow each process to exchange it's proc-local "remote elements" with the other respective processes.
        The current implementation first posts all non-blocking recieve calls, then
        The shift_arr created earlier is essential is creating a unique tag for the datamoving around between local_x buffers on each process.
        Recall, tuple elements in needed_heri are formatted (proc_to, proc_from, global_row_idx).
        Since this function is in the benchmark loop, need to do as little work possible, ideally.

    @brief Communicate the halo elements to and from the local x-vectors
    @param *local_context : 
    @param *local_x : the x vector corresponding to this process before the halo elements are communicated to it.
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template <typename VT, typename IT>
void communicate_halo_elements(
    ContextData<IT> *local_context,
    std::vector<VT> *local_x,
    const IT *work_sharing_arr,
    const int *my_rank,
    const int *comm_size)
{
    IT rows_in_to_proc, rows_in_from_proc, to_proc, from_proc, global_row_idx, num_local_elems;
    IT recv_shift = 0, send_shift = 0; // TODO: calculate more efficiently
    IT recieved_elems = 0, sent_elems = 0;
    IT test_rank = 1;

    // MPI_Request request; // TODO: What does this do?
    MPI_Status status;

    // Declare and populate arrays to keep track of number of elements already
    // recieved or sent to respective procs
    IT recv_counts[*comm_size];
    IT send_counts[*comm_size];

    for (IT i = 0; i < *comm_size; ++i)
    {
        recv_counts[i] = 0;
        send_counts[i] = 0;
    }

    // // TODO: DRY
    if (typeid(VT) == typeid(float))
    {
        // In order to place incoming elements in padded region of local_x, i.e. AFTER local elements
        rows_in_to_proc = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];
        for (IT to_proc_idx = 0; to_proc_idx < (local_context->to_send_heri).size(); to_proc_idx += 3)
        {
            global_row_idx = to_proc_idx + 2;
            to_proc = (local_context->to_send_heri)[to_proc_idx];
            // std::cout << "to_prcc: " << to_proc << ", from proc "<< *my_rank << std::endl;
            from_proc = *my_rank;

            send_shift = (local_context->shift_vec)[*comm_size * from_proc + to_proc];

            MPI_Send(
                &(*local_x)[(local_context->to_send_heri)[global_row_idx] - work_sharing_arr[*my_rank]],
                1,
                MPI_FLOAT,
                to_proc,
                send_shift + send_counts[to_proc],
                MPI_COMM_WORLD);

            ++send_counts[to_proc];
        }
        // TODO: definetly OpenMP this loop? Improve somehow
        for (IT from_proc_idx = 1; from_proc_idx < (local_context->local_needed_heri).size(); from_proc_idx += 3)
        {
            // How is this calculated, and is it totally necessary?
            rows_in_from_proc = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];

            to_proc = (local_context->local_needed_heri)[from_proc_idx - 1];
            from_proc = (local_context->local_needed_heri)[from_proc_idx];


            // NOTE: Source of uniqueness for tag
            // NOTE: essentially "transpose" shift array here
            recv_shift = (local_context->shift_vec)[*comm_size * from_proc + to_proc];

            MPI_Recv(
                &(*local_x)[rows_in_to_proc + recv_shift + recv_counts[from_proc]],
                1,
                MPI_FLOAT,
                from_proc,
                recv_shift + recv_counts[from_proc],
                MPI_COMM_WORLD,
                &status);

            ++recv_counts[from_proc];
        }
    }
    else if (typeid(VT) == typeid(double))
    {
        rows_in_to_proc = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];

        for (IT to_proc_idx = 0; to_proc_idx < (local_context->to_send_heri).size(); to_proc_idx += 3)
        {
            global_row_idx = to_proc_idx + 2;
            to_proc = (local_context->to_send_heri)[to_proc_idx];
            from_proc = *my_rank;

            send_shift = (local_context->shift_vec)[*comm_size * from_proc + to_proc];

            MPI_Send(
                &(*local_x)[(local_context->to_send_heri)[global_row_idx] - work_sharing_arr[*my_rank]],
                1,
                MPI_DOUBLE,
                to_proc,
                send_shift + send_counts[to_proc],
                MPI_COMM_WORLD);

            ++send_counts[to_proc];
        }

        for (IT from_proc_idx = 1; from_proc_idx < (local_context->local_needed_heri).size(); from_proc_idx += 3)
        {
            rows_in_from_proc = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];

            to_proc = (local_context->local_needed_heri)[from_proc_idx - 1];
            from_proc = (local_context->local_needed_heri)[from_proc_idx];

            recv_shift = (local_context->shift_vec)[*comm_size * from_proc + to_proc];

            MPI_Recv(
                &(*local_x)[rows_in_to_proc + recv_shift + recv_counts[from_proc]],
                1,
                MPI_DOUBLE,
                from_proc,
                recv_shift + recv_counts[from_proc],
                MPI_COMM_WORLD,
                &status);

            ++recv_counts[from_proc];
        }
    }
}

/**
    Adjust the col_idx from the scs data structure, so that elements of scs.values will multiply with the proper local_x elements.
        "local" elements of scs.values have their col_idx adjusted differently than "remote" elements of scs.values.

    @brief Adjust the col_idx from the scs data structure
    @param *local_scs : pointer to local scs struct
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template <typename VT, typename IT>
void adjust_halo_col_idxs(
    ScsData<VT, IT> *local_scs,
    const IT *work_sharing_arr,
    const int *my_rank,
    const int *comm_size)
{
    IT remote_elem_count = 0;
    IT amnt_lhs_remote_elems = 0;
    IT amnt_rhs_remote_elems = 0;
    IT test_rank = 1, show_steps = 0;
    IT exists_nz_elem = 0;
    IT idx_ctr;
    IT elem_idx;

    // TODO: Better to recalculate here? Or get from arguement?
    IT amnt_local_elems = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];

    std::vector<IT> original_col_idxs(local_scs->col_idxs.data(), local_scs->col_idxs.data() + local_scs->n_elements);
    std::vector<IT> col_all_inst_idx;


    if(show_steps){
        if(*my_rank == test_rank){
            std::cout << "column indices BEFORE adjustment: " << std::endl;
            for(int idx = 0; idx < local_scs->n_elements; ++idx){
                std::cout << local_scs->col_idxs[idx] << std::endl;
            }
            printf("\n");
        }
    }

    // #pragma omp parallel for
    for (IT i = 0; i < local_scs->n_elements; ++i)
    {
        // NOTE: Is this enough? Or do we also need local_scs->col_idxs[i] != 0?
        if (local_scs->values[i] != 0) // ignore padding
        {
            if ((local_scs->col_idxs[i] >= work_sharing_arr[*my_rank]) && local_scs->col_idxs[i] < work_sharing_arr[*my_rank + 1])
            { // i.e. if local
                local_scs->col_idxs[i] -= work_sharing_arr[*my_rank];
            }
        }
    }

    int lhs_halo_col_ctr = 0;
    // Proc 0 will never have LHS remote elements
    if(*my_rank != 0)
    {
        // start at the "left wall" of the matrix, and iterate columns until we reach local elements
        for (int col = 0; col < work_sharing_arr[*my_rank]; ++col)
        {
            if ((std::find(original_col_idxs.begin(), original_col_idxs.end(), col) != original_col_idxs.end()))
            { // if there exists ANY element in this column...

                col_all_inst_idx = find_items<IT>(original_col_idxs, col);

                for(auto idx : col_all_inst_idx)
                {
                    if(original_col_idxs[idx] == col && local_scs->values[idx] != 0)
                    {
                        exists_nz_elem = 1; // is this harness loop necessary?
                        break;
                    }
                }

                if(exists_nz_elem)
                { // and if this element is not padding
                    // #pragma omp parallel for
                    for(auto idx : col_all_inst_idx)
                    {
                        if(original_col_idxs[idx] == col && local_scs->values[idx] != 0) //second condition needed?
                        {
                            local_scs->col_idxs[idx] = amnt_local_elems + lhs_halo_col_ctr;
                        }
                    }
                // If at least one nonzero element exists in this column, increment counter
                ++lhs_halo_col_ctr;
                }
            }
        }   
    }


    int rhs_halo_col_ctr = 0;
    // Last proc will never have RHS remote elements
    if(*my_rank != (*comm_size - 1)){
        // start at the "left wall" of the matrix, and iterate columns until we reach local elements
        for (int col = work_sharing_arr[*my_rank + 1]; col < work_sharing_arr[*comm_size]; ++col)
        {
            if ((std::find(original_col_idxs.begin(), original_col_idxs.end(), col) != original_col_idxs.end()))
            { // if there exists ANY element in this column...
                col_all_inst_idx = find_items<IT>(original_col_idxs, col);

                for(auto idx : col_all_inst_idx)
                {
                    if(original_col_idxs[idx] == col && local_scs->values[idx] != 0)
                    {
                        exists_nz_elem = 1;
                        break;
                    }
                }

                if(exists_nz_elem)
                { // and if this element is not padding
                    // #pragma omp parallel for
                    for(auto idx : col_all_inst_idx)
                    {
                        if(original_col_idxs[idx] == col && local_scs->values[idx] != 0)
                        {
                            local_scs->col_idxs[idx] = amnt_local_elems + lhs_halo_col_ctr + rhs_halo_col_ctr;
                        }
                    }
                    // If at least one element exists in this column, increment counter
                    ++rhs_halo_col_ctr;
                }
            }
        }   
    }

    if(show_steps){
        if(*my_rank == test_rank){
            std::cout << "column indices AFTER adjustment: " << std::endl;
            for(int idx = 0; idx < local_scs->n_elements; ++idx){
                std::cout << local_scs->col_idxs[idx] << std::endl;
            }
            printf("\n");
        }
    }
}

/**
    @brief Partition the rows of the mtx structure, so that work is disributed (somewhat) evenly. The two options "seg-rows"
        and "seg-nnz" are there in an effort to have multiple load balancing techniques the segment work between processes.
    @param *total_mtx : data structure that was populated by the matrix market format reader mtx_reader.h
    @param *work_sharing_arr : the array describing the partitioning of the rows
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
*/
template <typename VT, typename IT>
void seg_work_sharing_arr(
    const MtxData<VT, IT> *total_mtx,
    IT *work_sharing_arr,
    const std::string *seg_method,
    const IT *comm_size)
{
    work_sharing_arr[0] = 0;

    IT segment;

    if ("seg-rows" == *seg_method)
    {
        IT rowsPerProc;

        // Evenly split the number of rows
        rowsPerProc = total_mtx->n_rows / *comm_size;

        // Segment rows to work on via. array
        for (segment = 1; segment < *comm_size + 1; ++segment)
        {
            // Can only do this because of "constant sized" segments
            work_sharing_arr[segment] = segment * rowsPerProc;
            if (segment == *comm_size)
            {
                // Set the last element to point to directly after the final row
                // (takes care of remainder rows)
                work_sharing_arr[*comm_size] = total_mtx->I[total_mtx->nnz - 1] + 1;
            }
        }
    }
    else if ("seg-nnz" == *seg_method)
    {
        IT nnzPerProc; //, remainderNnz;

        // Split the number of rows based on non zeros
        nnzPerProc = total_mtx->nnz / *comm_size;

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
        work_sharing_arr[*comm_size] = total_mtx->I[total_mtx->nnz - 1] + 1;
    }

    // Protect against edge case, where last process gets no work
    if(work_sharing_arr[*comm_size] == work_sharing_arr[*comm_size + 1]){
        for(IT loop_rank = 1; loop_rank < *comm_size; ++loop_rank){
            work_sharing_arr[loop_rank] -= 1;
        }
    }
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

/** TODO: make longer description
    @brief Initialize total_mtx, segment and send this to local_mtx, then to local_scs format
    @param *local_scs : pointer to local scs struct
    @param *local_context : struct containing information about local_scs communication needs
    @param *total_mtx : complete mtx struct, read .mtx file with mtx_reader.h
    @param *config : struct to initialze default values and user input
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *work_sharing_arr : the array describing the partitioning of the rows
*/
template<typename VT, typename IT>
void mpi_init_local_structs(
    ScsData<VT, IT> *local_scs,
    ContextData<IT> *local_context,
    MtxData<VT, IT> *total_mtx,
    Config *config, // shouldn't this be const?
    const std::string *seg_method,
    IT *work_sharing_arr,
    const IT *my_rank,
    const IT *comm_size)
{
    MtxData<VT, IT> local_mtx;

    MPI_Status status_bk, status_cols, status_rows, status_vals;

    MtxDataBookkeeping<ST> send_bk, recv_bk;
    MPI_Datatype bk_type;

    define_bookkeeping_type<IT>(&send_bk, &bk_type);

    IT msg_length;

    if (*my_rank == 0)
    {
        // Segment global row pointers, and place into an array
        seg_work_sharing_arr<VT, IT>(total_mtx, work_sharing_arr, seg_method, comm_size);

        // Eventhough we're iterting through the ranks, this loop is
        // (in the present implementation) executing sequentially on the root proc
        // STRONG CONTENDER FOR PRAGMA PARALELL
        for (IT loop_rank = 0; loop_rank < *comm_size; ++loop_rank)
        { // NOTE: This loop assumes we're using all ranks 0 -> comm_size-1

            std::vector<IT> local_I;
            std::vector<IT> local_J;
            std::vector<VT> local_vals; // an attempt to not have so much resizing in seg_mtx_struct

            // Assign rows, columns, and values to process local vectors
            if(config->log_prof && *my_rank == 0) {log("Begin seg_mtx_struct");}
            double begin_smtxs_time = MPI_Wtime();
            seg_mtx_struct<VT, IT>(total_mtx, &local_I, &local_J, &local_vals, work_sharing_arr, loop_rank);
            if(config->log_prof && *my_rank == 0) {log("Finish seg_mtx_struct", begin_smtxs_time, MPI_Wtime());}

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
                    local_vals.size(),
                    config->sort_matrix,
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
    else if (*my_rank != 0)
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

    // convert local_mtx to local_scs
    if(config->log_prof && *my_rank == 0) {log("Begin convert_to_scs");}
    double begin_ctscs_time = MPI_Wtime();
    convert_to_scs<VT, IT>(&local_mtx, config->chunk_size, config->sigma, local_scs);
    if(config->log_prof && *my_rank == 0) {log("Finish convert_to_scs", begin_ctscs_time, MPI_Wtime());}

    // HERE: adjust halo col idx

    // Broadcast work sharing array to other processes
    MPI_Bcast(work_sharing_arr,
              *comm_size + 1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    MPI_Type_free(&bk_type);

    std::vector<IT> local_needed_heri;
    if(config->log_prof && *my_rank == 0) {log("Begin collect_local_needed_heri");}
    double begin_clnh_time = MPI_Wtime();
    collect_local_needed_heri<VT, IT>(&local_needed_heri, local_scs, work_sharing_arr, my_rank, comm_size);
    if(config->log_prof && *my_rank == 0) {log("Finish collect_local_needed_heri", begin_clnh_time, MPI_Wtime());}


    // if(config->log_prof && *my_rank == 0) {log("Begin adjust_halo_col_idxs");}
    // double begin_ahci_time = MPI_Wtime();
    // adjust_halo_col_idxs<VT, IT>(local_scs, work_sharing_arr, my_rank, comm_size);
    // if(config->log_prof && *my_rank == 0) {log("Finish adjust_halo_col_idxs", begin_ahci_time, MPI_Wtime());}


    IT local_needed_heri_size = local_needed_heri.size();
    IT global_needed_heri_size;

    // TODO: Is this actually necessary?
    MPI_Allreduce(&local_needed_heri_size,
                  &global_needed_heri_size,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD
    );

    IT *global_needed_heri = new IT[global_needed_heri_size];

    for (IT i = 0; i < global_needed_heri_size; ++i)
    {
        global_needed_heri[i] = IT{};
    }

    // "to_send_heri" are all halo elements that this process is to send
    std::vector<IT> to_send_heri;
    if(config->log_prof && *my_rank == 0) {log("Begin collect_to_send_heri");}
    double begin_ctsh_time = MPI_Wtime();
    collect_to_send_heri<IT>(
        &to_send_heri,
        &local_needed_heri,
        global_needed_heri,
        my_rank,
        comm_size
    );
    if(config->log_prof && *my_rank == 0) {log("Finish collect_to_send_heri", begin_ctsh_time, MPI_Wtime());}


    // The shift array is used in the tag-generation scheme in halo communication.
    // the row idx is the "from_proc", the column is the "to_proc", and the element is the shift
    // after the local element index to make for the incoming halo elements
    std::vector<IT> shift_arr((*comm_size) * (*comm_size), 0);
    std::vector<IT> incidence_arr((*comm_size) * (*comm_size), 0);

    if(config->log_prof && *my_rank == 0) {log("Begin calc_heri_shifts");}
    double begin_chs_time = MPI_Wtime();
    calc_heri_shifts<IT>(
        global_needed_heri, 
        &global_needed_heri_size, 
        &shift_arr, 
        &incidence_arr, 
        comm_size
    ); // NOTE: always symmetric?
    if(config->log_prof && *my_rank == 0) {log("Finish calc_heri_shifts", begin_chs_time, MPI_Wtime());}

    local_context->local_needed_heri = local_needed_heri;
    local_context->to_send_heri = to_send_heri;
    local_context->shift_vec = shift_arr;
    local_context->incidence_vec = incidence_arr;
    local_context->amnt_local_elems = work_sharing_arr[*my_rank + 1] - work_sharing_arr[*my_rank];
    local_context->scs_padding = (IT)(local_scs->n_rows_padded - local_scs->n_rows);

    delete[] global_needed_heri;
}
#endif