#include "spmv.h"
#include <mpi.h>
#include <iostream>


// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Kernel for CSR format.
 */
template <typename VT, typename IT>
static void
spmv_omp_csr(const ST num_rows,
             const IT * RESTRICT row_ptrs,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; ++row) {
        VT sum{};
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            sum += values[j] * x[col_idxs[j]];
        }
        y[row] = sum;
    }
}


/**
 * Kernel for ELL format, data structures use row major (RM) layout.
 */
template <typename VT, typename IT>
static void
spmv_omp_ell_rm(
        const ST num_rows,
        const ST nelems_per_row,
        const IT * RESTRICT col_idxs,
        const VT * RESTRICT values,
        const VT * RESTRICT x,
        VT * RESTRICT y)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; row++) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; i++) {
            VT val = values[  row * nelems_per_row + i];
            IT col = col_idxs[row * nelems_per_row + i];

            sum += val * x[col];
        }
        y[row] = sum;
    }
}

/**
 * Kernel for ELL format, data structures use column major (CM) layout.
 */
template <typename VT, typename IT>
static void
spmv_omp_ell_cm(
        const ST num_rows,
        const ST nelems_per_row,
        const IT * RESTRICT col_idxs,
        const VT * RESTRICT values,
        const VT * RESTRICT x,
        VT * RESTRICT y)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; row++) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; i++) {
            VT val = values[row + i * num_rows];
            IT col = col_idxs[row + i * num_rows];

            sum += val * x[col];
        }
        y[row] = sum;
    }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 * 
 * Every process will fill it's procLocalY, which root collects in a MPI_Gather()
 */
template <typename VT, typename IT>
static void
spmv_omp_scs(const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{
    // called again here as to not pass more arguements into template,
    // which is also a possibility
    int myRank, commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // We can only choose a number of processors that are a divisor of trunc(n_chunks / C)
    // int procChoices =  n_chunks / C;

    // printf("%f", (float)n_chunks/commSize);
    // exit(1);

    // we know C = 8
    // if (myRank == 0 && procChoices % commSize != 0){
    //     std::cout << "Please choose a divisor of: " << procChoices << "for your number of MPI processes:" << std::endl;
    //     for (int i = 1; i <= procChoices; i++)
    //         if (procChoices % i == 0)
    //             std::cout <<" " << i;
    //     printf("\n");
    //     MPI_Finalize();
    //     exit(1);
    // }

    // allows last process to take remainder chunks
    int chunksInProc;

    if (myRank == 0 && commSize == 0)
        chunksInProc = n_chunks - (n_chunks / commSize) * (commSize - 1); //edge case, single process
    else if (myRank == commSize - 1)
        chunksInProc = (n_chunks - (n_chunks / commSize) * (commSize - 1));// + (n_chunks / commSize); //remainder case
    else
        chunksInProc = n_chunks / commSize; //normal case

    
    // printf("I'm proc %i and I will be working on %i chunks.\n", myRank, chunksInProc);
    // printf("%i", commSize - 1);

    const int cChunksInProc = chunksInProc;


    VT procLocalY[C * cChunksInProc];

    for (ST i = 0; i < C * chunksInProc; ++i) {
        procLocalY[i] = VT{};
    }

    // each process has it's own segment counter
    int segment = 0;
    // Chunk are distributed accross process via. MPI
    // all chunk sizes are constant in this case
    if (myRank != commSize - 1){
        for (ST c = myRank * chunksInProc; c < (myRank + 1) * chunksInProc; ++c){
            // printf("I'm proc %i and I'm working on chunk %i.\n", myRank, c);
            VT tmp[C];
            for (ST i = 0; i < C; ++i) {
                tmp[i] = VT{};
            }

            IT cs = chunk_ptrs[c];

            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < (IT)C; ++i) {
                    tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
                }
            }

            // basically append to proc-local buffer
            for(int i = 0; i < C; ++i){
                procLocalY[C * segment + i] = tmp[i];
            }
        // move to the next segment to place the tmp buffer within procLocalY
        // TODO: not a good solution
        ++segment;
        }
    }
    else{ // for the last rank process
        segment = 0;
        for (ST c = n_chunks - chunksInProc; c < n_chunks; ++c){
            // printf("I'm proc %i and I'm working on chunk %i.\n", myRank, c);
            VT tmp[C];
            for (ST i = 0; i < C; ++i) {
                tmp[i] = VT{};
            }

            IT cs = chunk_ptrs[c];

            for (IT j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < (IT)C; ++i) {
                    tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
                }
            }

            // basically append to proc-local buffer
            for(int i = 0; i < C; ++i){
                procLocalY[C * segment + i] = tmp[i];
        }
        // move to the next segment to place the tmp buffer within procLocalY
        // TODO: not a good solution
        ++segment;
        }
    }

    // chunk size different here
    if (myRank == 0){
        // Allocate space on root to recieve each procLocalY, in total, same size as y
        MPI_Gatherv(procLocalY, C * chunksInProc, MPI_DOUBLE, y, C * chunksInProc, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else{
        MPI_Gatherv(procLocalY, C * chunksInProc, MPI_DOUBLE, NULL, 0, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}


//          name      function         is_gpu format
REG_KERNELS("scs",    spmv_omp_scs,    false, MatrixFormat::SellCSigma);
REG_KERNELS("ell-rm", spmv_omp_ell_rm, false, MatrixFormat::EllRm);
REG_KERNELS("ell-cm", spmv_omp_ell_cm, false, MatrixFormat::EllCm);
REG_KERNELS("csr",    spmv_omp_csr,    false, MatrixFormat::Csr);

// OpenMP Ideas

// // OpenMP threads are distributed accross chunk-local columns
// #pragma omp parallel num_threads(2)
// {
//     IT procLocalThreads = 2; // do we actually require num threads divides chunk length?
//     VT privateTmp[C];
//     for (ST i = 0; i < C; ++i) {
//         privateTmp[i] = VT{};
//     }
//     #pragma omp for
//     for (IT n = 0; n < 2; n++){
//         for (IT j = (n/2)*chunk_lengths[c]; j < ((n+1)/2)*chunk_lengths[c]; ++j) {
//             for (IT i = 0; i < (IT)C; ++i) {
//                 privateTmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
//                 // tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
//             }
//         }
//     }
//     // TODO: does this need to be run serially?
//     #pragma omp critical //basically a reduction on the tmp array
//     {
//         for(IT i = 0; i < procLocalThreads; ++i)
//             tmp[i] += privateTmp[i];
//     }
// }

// or trivial for testing?
// #pragma omp parallel
// {
//     #pragma omp single
//     {
//         for (IT j = 0; j < chunk_lengths[c]; ++j) {
//             for (IT i = 0; i < (IT)C; ++i) {
//                 tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
//             }
//         }
//     }
// }
