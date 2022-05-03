/*
    The idea here, is to have the root process read the .mtx file to a MtxData struct like normal. But afterwards,
    each processes would take certain parts of root MtxData struct, and each make their own...?
*/
using VT = double;
using ST = long;
using IT = int;

void
log(const char * format, ...)
{}

#include "mtx-reader.h"

#include <iostream>

struct Config
{
    long n_els_per_row { -1 };      // ell
    long chunk_size    { -1 };      // sell-c-sigma
    long sigma         { -1 };      // sell-c-sigma

    // Initialize rhs vector with random numbers.
    bool random_init_x { true };
    // Override values of the matrix, read via mtx file, with random numbers.
    bool random_init_A { false };

    // No. of repetitions to perform. 0 for automatic detection.
    unsigned long n_repetitions{};

    // Verify result of SpVM.
    bool verify_result{ true };

    // Verify result against solution of COO kernel.
    bool verify_result_with_coo{ false };

    // Print incorrect elements from solution.
    bool verbose_verification{ true };

    // Sort rows/columns of sparse matrix before
    // converting it to a specific format.
    bool sort_matrix{ true };
};

static std::string
file_base_name(const char * file_name)
{
    if (file_name == nullptr) {
        return std::string{};
    }

    std::string file_path(file_name);
    std::string file;

    size_t pos_slash = file_path.rfind('/');

    if (pos_slash == file_path.npos) {
        file = std::move(file_path);
    }
    else {
        file = file_path.substr(pos_slash + 1);
    }

    size_t pos_dot = file.rfind('.');

    if (pos_dot == file.npos) {
        return file;
    }
    else {
        return file.substr(0, pos_dot);
    }
}
template <typename IT>
IT getIndex(std::vector<IT> v, int K)
{
    auto it = find(v.begin(), v.end(), K);
 
    // If element was found
    if (it != v.end())
    {
     
        // calculating the index
        // of K
        int index = it - v.begin();
        return index;
    }
    else {
        // If the element is not
        // present in the vector
        return -1; // TODO: implement better error
    }
}


int main(int argc, char **argv){
    // Method 1. Segment by number of rows
    // if myRank == 0 {
    Config config;
    const char * file_name{};
    file_name = argv[1];
    int commSize = 3;
    int myRank = 2;

    std::string matrix_name = file_base_name(file_name);
    MtxData<VT, IT> mtx = read_mtx_data<double, int>(file_name, config.sort_matrix); 
    int rowsPerProc = mtx.n_rows / commSize;

    // How to get I, J and values for each process?
    std::vector<IT> procLocalI;
    std::vector<IT> procLocalJ;
    std::vector<VT> procLocalValues;
    int startingIdx, runningIdx, finishingIdx;

    // proc P gets all rows (i.e. all values corresponding to) rowsPerProc*myRank
    for(int row = myRank * rowsPerProc; row < (myRank + 1) * rowsPerProc; row++){
        startingIdx = getIndex<IT>(mtx.I, row); // this index is used to get corresponding col and val
        runningIdx = startingIdx;
        finishingIdx = getIndex<IT>(mtx.I, row + 1); // unnecessary?
        while(runningIdx != finishingIdx){
            procLocalI.push_back(mtx.I[runningIdx]);
            procLocalJ.push_back(mtx.J[runningIdx]);
            procLocalValues.push_back(mtx.values[runningIdx]);
            ++runningIdx;
        }
    }         
    // }

    // for(int myRank = 1; myRank < 2; ++myRank){
    //     printf("I'm rank %i, and my values are:\n", myRank);
    //     for(int i = 0; i < procLocalValues.size(); ++i){
    //         std::cout << "(" << procLocalJ[i] + 1 << "," <<  procLocalI[i] + 1 << ")" << "\n" ;
    //     }
    // }

    // MtxData<double, int> procLocalStruct = {rowsPerProc, mtx.n_cols, -1, config.sort_matrix, 0, procLocalI, procLocalJ, procLocalValues};
    // same number of columns as full mtx
    // wont be symmetric, won' be square


    // proc 1 gets first "rowsPerProc" rows, then proc 2 gets next "rowsPerProc" rows, etc.
    // last proc gets "rowsPerProc" rows plus leftover (= mtx.n_rows % rowsPerProc) 
    // }

    // Method 2. Segment by number of non-zero elements
}