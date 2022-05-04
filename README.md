# Ultimate-SpMV
MPI+X SpMV with SELL-C-sigma format

In this second attempt, we've implemented two methods for the segmentation and distribution of our sparse matrix. This is to act as a preprocessing top layer, to be performed before the benchmark. The purpose of the segmentation is so that each MPI process does not read and store the entire sparse matrix, but only the segment with which it operates on.

There is a "splitting" layer between the reading of the .mtx file, and actually using the spmv-omp on the matrix. This layer attempts to segment the MtxData struct into smaller structs, over which each process reads and then performs spMVM with. This splitting and disribution is done on the basis of rows (method 1), or non-zero elements (method 2).

The method to gather the results from individual processes is not finished. 
