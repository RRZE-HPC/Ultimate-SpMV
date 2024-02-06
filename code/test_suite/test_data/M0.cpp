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