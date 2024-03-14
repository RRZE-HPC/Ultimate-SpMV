#include "classes_structs.hpp"

//////////////////////////// M-big test data ////////////////////////////
MtxData<double, int> M_big {
    10, // n_rows
    10, // n_cols
    18, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,2,2,2,3,4,5,5,5,6,7,7,7,8,9}, // I
    std::vector<int> {0,3,4,1,0,1,2,3,4,5,8,9,6,5,6,7,8,9}, // J
    std::vector<double> {.11, 14, 15, .22, 31, 32, .33, 44, 55, .66, 69, .610, 77, 86, 87, 88, .99, 1010} // values
};

//////////////////////////// M-big-hp test data ////////////////////////////
MtxData<float, int> exp_M_big_lp {
    10, // n_rows
    10, // n_cols
    6, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1,2,5,5,8}, // I
    std::vector<int> {0,1,2,5,9,8}, // J
    std::vector<float> {.11, .22, .33, .66, .610, .99} // values
};

ScsData<float, int> exp_M_big_lp_scs_1_2 {
    1, // C
    2, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    6, // n_elements
    6, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(6), // col_idxs
    V<float, int>(6), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_2 {
    std::vector<int> {0,1,2,3,3,5,5,5,5,6,6}, //explicit_chunk_ptrs
    std::vector<int> {1,1,1,0,2,0,0,0,1,0}, //explicit_chunk_lengths
    std::vector<int> {0,1,2,5,9,8}, //explicit_col_idxs
    std::vector<float> {.11,.22,.33,.66,.610,.99}, //explicit_values
    std::vector<int> {0,1,2,3,5,4,6,7,8,9}, //explicit_old_to_new_idx
    std::vector<int> {0,1,2,3,5,4,6,7,8,9} //explicit_new_to_old_idx
};

ScsData<float, int> exp_M_big_lp_scs_1_2_compressed {
    1, // C
    2, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    6, // n_elements
    6, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(6), // col_idxs
    V<float, int>(6), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_2_compressed {
    std::vector<int> {0,1,2,3,3,5,5,5,5,6,6}, //explicit_chunk_ptrs
    std::vector<int> {1,1,1,0,2,0,0,0,1,0}, //explicit_chunk_lengths
    std::vector<int> {0,1,2,5,9,8}, //explicit_col_idxs
    std::vector<float> {.11,.22,.33,.66,.610,.99}, //explicit_values
    std::vector<int> {0,1,2,3,5,4,6,7,8,9}, //explicit_old_to_new_idx
    std::vector<int> {0,1,2,3,5,4,6,7,8,9} //explicit_new_to_old_idx
};

ScsData<float, int> exp_M_big_lp_scs_1_128 {
    1, // C
    128, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    6, // n_elements
    6, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(6), // col_idxs
    V<float, int>(6), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_128 {
    std::vector<int> {0,2,3,4,5,6,6,6,6,6,6}, //explicit_chunk_ptrs
    std::vector<int> {2,1,1,1,1,0,0,0,0,0}, //explicit_chunk_lengths
    std::vector<int> {5,9,0,1,2,8}, //explicit_col_idxs
    std::vector<float> {.66, .610, .11, .22, .33, .99}, //explicit_values
    std::vector<int> {1,2,3,5,6,0,7,8,4,9}, //explicit_old_to_new_idx
    std::vector<int> {5,0,1,2,8,3,4,6,7,9} //explicit_new_to_old_idx
};

ScsData<float, int> exp_M_big_lp_scs_1_128_compressed {
    1, // C
    128, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    6, // n_elements
    6, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(6), // col_idxs
    V<float, int>(6), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_128_compressed {
    std::vector<int> {0,2,3,4,5,6,6,6,6,6,6}, //explicit_chunk_ptrs
    std::vector<int> {2,1,1,1,1,0,0,0,0,0}, //explicit_chunk_lengths
    std::vector<int> {5,9,0,1,2,8}, //explicit_col_idxs
    std::vector<float> {.66, .610, .11, .22, .33, .99}, //explicit_values
    std::vector<int> {1,2,3,5,6,0,7,8,4,9}, //explicit_old_to_new_idx
    std::vector<int> {5,0,1,2,8,3,4,6,7,9} //explicit_new_to_old_idx
};

//////////////////////////// M-big-hp test data ////////////////////////////
MtxData<double, int> exp_M_big_hp{
    10, // n_rows
    10, // n_cols
    12, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,2,2,3,4,5,6,7,7,7,9}, // I
    std::vector<int> {3,4,0,1,3,4,8,6,5,6,7,9}, // J
    std::vector<double> {14, 15, 31, 32, 44, 55, 69, 77, 86, 87, 88, 1010} // values
};

ScsData<double, int> exp_M_big_hp_scs_1_2 {
    1, // C
    2, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    12, // n_elements
    12, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(12), // col_idxs
    V<double, int>(12), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_2 {
    std::vector<int> {0,2,2,4,5,6,7,10,11,12,12}, //explicit_chunk_ptrs
    std::vector<int> {2,0,2,1,1,1,3,1,1,0}, //explicit_chunk_lengths
    std::vector<int> {3,4,0,1,3,4,8,5,6,7,6,9}, //explicit_col_idxs
    std::vector<double> {14,15,31,32,44,55,69,86,87,88,77,1010}, //explicit_values
    std::vector<int> {0,1,2,3,4,5,7,6,9,8}, //explicit_old_to_new_idx
    std::vector<int> {0,1,2,3,4,5,7,6,9,8} //explicit_new_to_old_idx
};

ScsData<double, int> exp_M_big_hp_scs_1_2_compressed {
    1, // C
    2, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    12, // n_elements
    12, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(12), // col_idxs
    V<double, int>(12), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_2_compressed {
    std::vector<int> {0,2,2,4,5,6,7,10,11,12,12}, //explicit_chunk_ptrs
    std::vector<int> {2,0,2,1,1,1,3,1,1,0}, //explicit_chunk_lengths
    std::vector<int> {3,4,0,1,3,4,8,5,6,7,6,9}, //explicit_col_idxs
    std::vector<double> {14,15,31,32,44,55,69,86,87,88,77,1010}, //explicit_values
    std::vector<int> {0,1,2,3,4,5,7,6,9,8}, //explicit_old_to_new_idx
    std::vector<int> {0,1,2,3,4,5,7,6,9,8} //explicit_new_to_old_idx
};

ScsData<double, int> exp_M_big_hp_scs_1_128 {
    1, // C
    128, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    12, // n_elements
    12, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(12), // col_idxs
    V<double, int>(12), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_128 {
    std::vector<int> {0,3,5,7,8,9,10,11,12,12,12}, //explicit_chunk_ptrs
    std::vector<int> {3,2,2,1,1,1,1,1,0,0}, //explicit_chunk_lengths
    std::vector<int> {5,6,7,3,4,0,1,3,4,8,6,9}, //explicit_col_idxs
    std::vector<double> {86,87,88,14,15,31,32,44,55,69,77,1010}, //explicit_values
    std::vector<int> {1,8,2,3,4,5,6,0,9,7}, //explicit_old_to_new_idx
    std::vector<int> {7,0,2,3,4,5,6,9,1,8} //explicit_new_to_old_idx
};

ScsData<double, int> exp_M_big_hp_scs_1_128_compressed {
    1, // C
    128, // sigma
    10, // n_rows
    10, // n_cols
    10, // n_rows_padded
    10, // n_chunks
    12, // n_elements
    12, // nnz

    V<int, int>(11), // chunk_ptrs
    V<int, int>(10), // chunk_lengths
    V<int, int>(12), // col_idxs
    V<double, int>(12), // values
    V<int, int>(10), // old_to_new_idx
    std::vector<int> (10) // new_to_old_idx 
};

ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_128_compressed {
    std::vector<int> {0,3,5,7,8,9,10,11,12,12,12}, //explicit_chunk_ptrs
    std::vector<int> {3,2,2,1,1,1,1,1,0,0}, //explicit_chunk_lengths
    std::vector<int> {5,6,7,3,4,0,1,3,4,8,6,9}, //explicit_col_idxs
    std::vector<double> {86,87,88,14,15,31,32,44,55,69,77,1010}, //explicit_values
    std::vector<int> {1,8,2,3,4,5,6,0,9,7}, //explicit_old_to_new_idx
    std::vector<int> {7,0,2,3,4,5,6,9,1,8} //explicit_new_to_old_idx
};