#include "classes_structs.hpp"

//////////////////////////// M0 test data ////////////////////////////
MtxData<double, int> M0 {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {0.,0.,0.,0.,0.,0.,0.,0.,0.} // values
};

//////////////////////////// M1 test data ////////////////////////////
MtxData<double, int> M1 {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {1.5,0.5,0.25,-0.1,1.0,1.1,-1.0,0.999999999999,10101.1} // values
};

//////////////////////////// M1-hp test data ////////////////////////////
MtxData<double, int> exp_M1_hp {
    3, // n_rows
    3, // n_cols
    5, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1,1,2,2}, // I
    std::vector<int> {0,1,2,0,2}, // J
    std::vector<double> {1.5,1.0,1.1,-1.0,10101.1} // values
};

//////////////////////////// M1-hp_scs test data ////////////////////////////
ScsData<double, int> exp_M1_hp_scs_1_1 {
    1, // C
    1, // sigma
    3, // n_rows
    3, // n_cols
    3, // n_rows_padded
    3, // n_chunks
    5, // n_elements
    5, // nnz

    V<int, int>(4), // chunk_ptrs
    V<int, int>(3), // chunk_lengths
    V<int, int>(5), // col_idxs
    V<double, int>(5), // values
    V<int, int>(3), // old_to_new_idx
    std::vector<int> (3) // new_to_old_idx 
};

ScsExplicitData<double, int> explicit_exp_M1_hp_scs_1_1 {
    std::vector<int> {0,1,3,5}, //explicit_chunk_ptrs
    std::vector<int> {1,2,2}, //explicit_chunk_lengths
    std::vector<int> {0,1,2,0,2}, //explicit_col_idxs
    std::vector<double> {1.5, 1.0, 1.1, -1.0, 10101.1}, //explicit_values
    std::vector<int> {0,1,2}, //explicit_old_to_new_idx
    std::vector<int> {0,1,2} //explicit_new_to_old_idx
};

//////////////////////////// M1-lp test data ////////////////////////////
MtxData<float, int> exp_M1_lp {
    3, // n_rows
    3, // n_cols
    4, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,1,2}, // I
    std::vector<int> {1,2,0,1}, // J
    std::vector<float> {0.5,0.25,-0.1,0.999999999999} // values
};

//////////////////////////// M1-lp_scs test data ////////////////////////////
ScsData<float, int> exp_M1_lp_scs_1_1 {
    1, // C
    1, // sigma
    3, // n_rows
    3, // n_cols
    3, // n_rows_padded
    3, // n_chunks
    4, // n_elements
    4, // nnz

    V<int, int>(4), // chunk_ptrs
    V<int, int>(3), // chunk_lengths
    V<int, int>(4), // col_idxs
    V<float, int>(4), // values
    V<int, int>(3), // old_to_new_idx
    std::vector<int> (3) // new_to_old_idx 
};

ScsExplicitData<float, int> explicit_exp_M1_lp_scs_1_1 {
    std::vector<int> {0,2,3,4}, //explicit_chunk_ptrs
    std::vector<int> {2,1,1}, //explicit_chunk_lengths
    std::vector<int> {1,2,0,1}, //explicit_col_idxs
    std::vector<float> {0.5, 0.25, -0.1, 0.999999999999}, //explicit_values
    std::vector<int> {0,1,2}, //explicit_old_to_new_idx
    std::vector<int> {0,1,2} //explicit_new_to_old_idx
};

//////////////////////////// M1_scs test data ////////////////////////////
ScsData<double, int> exp_M1_scs_1_1 {
    1, // C
    1, // sigma
    3, // n_rows
    3, // n_cols
    3, // n_rows_padded
    3, // n_chunks
    9, // n_elements
    9, // nnz

    V<int, int>(4), // chunk_ptrs
    V<int, int>(3), // chunk_lengths
    V<int, int>(9), // col_idxs
    V<double, int>(9), // values
    V<int, int>(3), // old_to_new_idx
    std::vector<int> (3) // new_to_old_idx 
};

ScsExplicitData<double, int> explicit_exp_M1_scs_1_1 {
    std::vector<int> {0,3,6,9}, //explicit_chunk_ptrs
    std::vector<int> {3,3,3}, //explicit_chunk_lengths
    std::vector<int> {0,1,2,0,1,2,0,1,2}, //explicit_col_idxs
    std::vector<double> {1.5,0.5,0.25,-0.1,1.0,1.1,-1.0,0.999999999999,10101.1}, //explicit_values
    std::vector<int> {0,1,2}, //explicit_old_to_new_idx
    std::vector<int> {0,1,2} //explicit_new_to_old_idx
};