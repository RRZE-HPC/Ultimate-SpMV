# Ultimate-SpMV
MPI+X SpMV with SELL-C-sigma format

In this second attempt, we've implemented two methods for the segmentation our sparse matrix. This is to act as a load balancing "top layer", to be performed before the benchmark bench_spmv call. The purpose of the segmentation is so that each MPI process does not read and store the entire sparse matrix, but only the "submatrix" with which it operates on.

There is a "splitting" layer between the reading of the .mtx file, and actually using the spmv-omp on the matrix. This layer segment the MtxData struct into smaller structs, over which each process reads and then performs spMVM with. This splitting and disribution is done on the basis of rows (-seg-by-rows flag), or non-zero elements (-seg-by-nnz flag).

Also, we have a naïve optimization on the x-vector by which the proc-local matrix is multiplied, so that each process does not store an entire copy of x. We take the lowest and highest column index of the proc-local matrix, and only store values of the x-vector between these indicies.

An Allgather is then used to collect the proc-local results to every other processes, essentailly reconstructing the result vector on each process. The reason for this, is so that the full x and y vectors are of compatible size, and we gave the option to "swap" them. We want to have this option so that can eventually embed Ultimate-SpMV into iterative solvers, such as Krylov subspace methods.