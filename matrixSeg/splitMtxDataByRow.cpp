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
#include <set>
#include <iostream>
#include <mpi.h>
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

template <typename VT, typename IT>
struct MtxDataBookkeeping
{
    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    bool is_sorted{};
    bool is_symmetric{};
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
    /*
    Gets the index of the first instance of the element K in the vector v.
    */
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
    int myRank, commSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Status statusBK, statusCols, statusRows, statusValues;
    MtxDataBookkeeping<VT, IT> sendBookkeeping, recvBookkeeping;
    MtxData<VT, IT> procLocalMtxStruct;

    // Generate MPI Datatype
    int blockLengthArrBK[2];
    MPI_Aint displacementArrBK[2], firstAddressBK, secondAddressBK;

    MPI_Datatype typeArrBK[2], bookkeepingType;
    typeArrBK[0] = MPI_LONG; typeArrBK[1] = MPI_CXX_BOOL;
    blockLengthArrBK[0] = 3; // using 3 int elements
    blockLengthArrBK[1] = 2; // and 2 bool elements
    MPI_Get_address(&sendBookkeeping.n_rows, &firstAddressBK);
    MPI_Get_address(&sendBookkeeping.is_sorted, &secondAddressBK);

    displacementArrBK[0] = (MPI_Aint) 0; // calculate displacements from addresses
    displacementArrBK[1] = MPI_Aint_diff(secondAddressBK, firstAddressBK);
    MPI_Type_create_struct(2, blockLengthArrBK, displacementArrBK, typeArrBK, &bookkeepingType);
    MPI_Type_commit(&bookkeepingType);

    int msgLength;

    if (myRank == 0 ){
        Config config;
        const char * file_name{};
        file_name = argv[1];

        std::string matrix_name = file_base_name(file_name);
        // NOTE: Matrix will be read in as SORTED
        MtxData<VT, IT> mtx = read_mtx_data<double, int>(file_name, config.sort_matrix);
        // Evenly split the number of rows
        int rowsPerProc = mtx.n_rows / commSize;

        // Eventhough we're iterting through the ranks, 
        // this loop is still executing sequentially on the root proc
        for(IT rank = 0; rank < commSize; ++rank){ // NOTE: This loop assumes we're using all ranks 0 -> commSize-1
            std::vector<IT> procLocalI;
            std::vector<IT> procLocalJ;
            std::vector<VT> procLocalValues;
            int startingIdx, runningIdx, finishingIdx;
            int nextRow;

            // MAIN LOOP. Assigns rows, columns, and values to process local vectors
            // proc 1 gets first "rowsPerProc" rows, then proc 2 gets next "rowsPerProc" rows, etc.
            for(int row = rank * rowsPerProc; row < (rank + 1) * rowsPerProc; ++row){
                nextRow = row + 1;

                // Return the first instance of that row present in mtx.
                startingIdx = getIndex<IT>(mtx.I, row);

                // once we have the index of the first instance of the row, 
                // we calculate the index of the first instance of the next row
                if (nextRow != mtx.n_rows){
                    // meaning if we're in the "normal" regime
                    finishingIdx = getIndex<IT>(mtx.I, nextRow);
                }
                else{
                    // for the "last row" case, just set finishingIdx to the number of non zeros in mtx
                    // NOTE: if there are remiander rows, this block will not be entered
                    finishingIdx = mtx.nnz;
                }
                runningIdx = startingIdx;

                // This while loop will go "across the rows", basically filling the process local vectors
                // with the appropiate data
                // printf("With starting idx: %i, and end index %i.\n", startingIdx, finishingIdx);
                // TODO: is this better than a plain while loop here?
                do{
                    procLocalI.push_back(mtx.I[runningIdx]);
                    procLocalJ.push_back(mtx.J[runningIdx]);
                    procLocalValues.push_back(mtx.values[runningIdx]);
                    ++runningIdx;
                } while(runningIdx != finishingIdx);
            }

            // REMAINDER LOOP. Adds remaining rows to last processes
            // last proc gets "rowsPerProc" rows plus leftover (= mtx.n_rows % rowsPerProc) 
            if (rank == commSize - 1){
                for(int row = (rank + 1) * rowsPerProc; row < mtx.n_rows; ++row){
                    // picks up where the above loop leaves off, at row "commSize * rowsPerProc"
                    nextRow = row + 1;
                    startingIdx = getIndex<IT>(mtx.I, row);

                    if (nextRow != mtx.n_rows){
                        finishingIdx = getIndex<IT>(mtx.I, nextRow);
                    }
                    else{
                        finishingIdx = mtx.nnz;
                    }
                    runningIdx = startingIdx;
                    do{
                        procLocalI.push_back(mtx.I[runningIdx]);
                        procLocalJ.push_back(mtx.J[runningIdx]);
                        procLocalValues.push_back(mtx.values[runningIdx]);
                        ++runningIdx;
                    } while(runningIdx != finishingIdx);
                }
            }

            // Count the number of rows in each processes
            int procLocalRowCount = std::set<IT>( procLocalI.begin(), procLocalI.end() ).size();

            // Here, we segment data for the root process
            if (rank == 0){
                procLocalMtxStruct = {
                    procLocalRowCount,
                    mtx.n_cols,
                    procLocalValues.size(),
                    config.sort_matrix,
                    0,
                    procLocalI,
                    procLocalJ,
                    procLocalValues
                };
            }
            // Here, we segment and send data to another proc
            else{
                sendBookkeeping = {
                    procLocalRowCount, 
                    mtx.n_cols, // will need to remove reference to mtx, since all procs wont have view of it 
                    procLocalValues.size(), 
                    config.sort_matrix, 
                    0
                };

                // First, send BK struct
                MPI_Send(&sendBookkeeping, 1, bookkeepingType, rank, 99, MPI_COMM_WORLD);

                // Next, send three arrays, which will need to be probed on recieving process
                MPI_Send(&procLocalI[0], procLocalI.size(), MPI_INT, rank, 42, MPI_COMM_WORLD);
                MPI_Send(&procLocalJ[0], procLocalJ.size(), MPI_INT, rank, 43, MPI_COMM_WORLD);
                MPI_Send(&procLocalValues[0], procLocalValues.size(), MPI_DOUBLE, rank, 44, MPI_COMM_WORLD);
            }
        }
    }          
    else if(myRank != 0){
        // First, recieve BK struct
        MPI_Recv(&recvBookkeeping, 1, bookkeepingType, 0, 99, MPI_COMM_WORLD, &statusBK);

        // Next, probe single message (since all arrays same length) and allocate space for incoming arrays
        MPI_Probe(0, 42, MPI_COMM_WORLD, &statusRows);
        MPI_Get_count(&statusRows, MPI_INT, &msgLength);
        IT *recvBufRowCoords = new IT [msgLength];
        IT *recvBufColCoords = new IT [msgLength]; 
        VT *recvBufValues = new VT [msgLength]; // TODO: not conducive to templates?

        // Next, recieve 3 arrays that we've allocated space for
        MPI_Recv(recvBufRowCoords, msgLength, MPI_INT, 0, 42, MPI_COMM_WORLD, &statusRows);
        MPI_Recv(recvBufColCoords, msgLength, MPI_INT, 0, 43, MPI_COMM_WORLD, &statusCols);
        MPI_Recv(recvBufValues, msgLength, MPI_DOUBLE, 0, 44, MPI_COMM_WORLD, &statusValues);

        // TODO: Just how bad is this?... Are we copying array -> vector?
        std::vector<IT> vRows(recvBufRowCoords, recvBufRowCoords + msgLength);
        std::vector<IT> vCols(recvBufColCoords, recvBufColCoords + msgLength);
        std::vector<VT> vValues(recvBufValues, recvBufValues + msgLength);

        procLocalMtxStruct = {
            recvBookkeeping.n_rows,
            recvBookkeeping.n_cols,
            recvBookkeeping.nnz,
            recvBookkeeping.is_sorted,
            recvBookkeeping.is_symmetric,
            vRows,
            vCols,
            vValues
        };
    }
    MPI_Barrier(MPI_COMM_WORLD); // TODO: Is this needed?
    
    print_mtx(procLocalMtxStruct);

    MPI_Finalize();
}