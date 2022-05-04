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

template <typename VT, typename IT>
struct MtxDataValues
{
    std::vector<IT> I;
    std::vector<IT> J;
    std::vector<VT> values;
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
    MPI_Status status;
    MtxDataBookkeeping<VT, IT> sendBookkeeping, recvBookkeeping;
    MtxDataValues<VT, IT> sendData, recvData;

    int blockLengthArr[2];
    MPI_Aint displacementArr[2], firstAddress, secondAddress;

    MPI_Datatype typeArr[2], bookkeepingType;
    typeArr[0] = MPI_LONG;
    typeArr[1] = MPI_CXX_BOOL;
    blockLengthArr[0] = 3; // using 3 int elements
    blockLengthArr[1] = 2; // using 2 bool elements
    MPI_Get_address(&sendBookkeeping.n_rows, &firstAddress); // get addresses of "nodeSend" members
    MPI_Get_address(&sendBookkeeping.is_sorted, &secondAddress);

    displacementArr[0] = (MPI_Aint) 0; // calculate displacements
    displacementArr[1] = MPI_Aint_diff(secondAddress, firstAddress);
    MPI_Type_create_struct(2, blockLengthArr, displacementArr, typeArr, &bookkeepingType);
    MPI_Type_commit(&bookkeepingType);

    // Method 1. Segment by number of rows
    if (myRank == 0 ){
        Config config;
        const char * file_name{};
        file_name = argv[1];
        // int commSize = 50;
        // int myRank = 4;

        std::string matrix_name = file_base_name(file_name);
        MtxData<VT, IT> mtx = read_mtx_data<double, int>(file_name, config.sort_matrix); 
        int rowsPerProc = mtx.n_rows / commSize; //24
        int rowsRemainder = mtx.n_rows % rowsPerProc;

        // printf("Total number of rows in mtx: %li\n", mtx.n_rows);
        // printf("Number of procs: %i, and number of rows per proc: %i\n", commSize, rowsPerProc);
        // printf("Remainder rows: %i\n", rowsRemainder);
        // exit(0);

        // Initialize and populate vector with our ranks
        std::vector<IT> rankVec;
        for(int i = 0; i < commSize; ++i){
            rankVec.push_back(i);
        }

        // int tempCnt = 0;


        // idea, for each rank in the rank array
        //  -make this struct
        //  -send the struct
        for(IT rank : rankVec){
            std::vector<IT> procLocalI;
            std::vector<IT> procLocalJ;
            std::vector<VT> procLocalValues;
            int startingIdx, runningIdx, finishingIdx;
            int nextRow;

            // Main loop. Assigns rows, columns, and values to process local vectors
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
                // NOTE: is this better than a plain while loop here?
                do{
                    procLocalI.push_back(mtx.I[runningIdx]);
                    procLocalJ.push_back(mtx.J[runningIdx]);
                    procLocalValues.push_back(mtx.values[runningIdx]);
                    ++runningIdx;
                } while(runningIdx != finishingIdx);
            }

            // TODO: DRY. Integrate with above loop.
            // Remainder loop. Adds remaining rows to last processes
            if (rank == commSize - 1){
                for(int row = (rank + 1) * rowsPerProc; row < mtx.n_rows; row++){
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
        

            // For testing/verification
            // for(int myRank = 0; myRank < 1; ++myRank){
            //     printf("I'm rank %i, and my values are:\n", myRank);
            //     for(int i = 0; i < procLocalValues.size(); ++i){
            //         std::cout << "(" << procLocalJ[i] + 1 << "," <<  procLocalI[i] + 1 << ")" << "\n" ;
            //     }
            // }

            // Count the number of rows in each processes
            int procLocalRowCount = std::set<IT>( procLocalI.begin(), procLocalI.end() ).size();
            // std::cout << uniqueCount << std::endl;

            sendBookkeeping = {
                procLocalRowCount, 
                mtx.n_cols, // will need to remove reference to mtx, since all procs wont have view of it 
                procLocalValues.size(), 
                config.sort_matrix, 
                0
            };

            // We send the data over two MPI_Send commands:



            // NOTE: only sending one of these
            MPI_Send(&sendBookkeeping, 1, bookkeepingType, 1, 42, MPI_COMM_WORLD);
            // std::cout << procLocalStruct.values.size() << std::endl;
            // // print_mtx(procLocalStruct);
            // // printf("I'm rank %i, and my values are:\n", rank);
            // for(int i = 0; i < procLocalStruct.values.size(); ++i){
            //     // std::cout << "(" << procLocalStruct.J[i] + 1 << "," <<  procLocalStruct.I[i] + 1 << ") -> " << procLocalStruct.values[i] << "\n" ;
            //     std::cout << procLocalStruct.values[i] - mtx.values[tempCnt]<< "\n" ;
            //     tempCnt++;

            // }
            // printf("\n");
        }
    }          
    else if(myRank == 1){
        MPI_Recv(&recvBookkeeping, 1, bookkeepingType, 0, 42, MPI_COMM_WORLD, &status);

        // printf("Status: %d\n", status.MPI_ERROR);

        printf("[%li, %li, %li], (%d, %d)\n", recvBookkeeping.n_rows, recvBookkeeping.n_cols, recvBookkeeping.nnz, recvBookkeeping.is_sorted, recvBookkeeping.is_symmetric);
    }
    


    // proc 1 gets first "rowsPerProc" rows, then proc 2 gets next "rowsPerProc" rows, etc.
    // last proc gets "rowsPerProc" rows plus leftover (= mtx.n_rows % rowsPerProc) 
    // }

    // Method 2. Segment by number of non-zero elements
    // int nnzPerProc = mtx.nnz / commSize;
    // int nnzRemainder = mtx.nnz % nnzPerProc;

    // // Still need index of nnz
    // for(int nz = myRank * nnzPerProc; nz < (myRank + 1) * nnzPerProc; ++nz){
    //     // I dont think this idea works here
    //     getIndex<IT>(mtx.values, nz)

        //there needs to be some flag that states:
        // if nnzPerProc is reached, just give rest of row to processes

    // }
    MPI_Finalize();
}