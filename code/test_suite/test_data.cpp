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

//////////////////////////// M1-mp test data ////////////////////////////
MtxData<double, int> M1_mp {
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