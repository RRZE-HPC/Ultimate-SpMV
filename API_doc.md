# Ultimate-SpMV interface documentation

## Introduction
The interface for USpMV is a single header file, interface.hpp, which contains everything you need. 
As it currently stands, the functions in the API don't have or need any awareness of MPI. The organization of communication information should be done by the host application, and these routines are used on each MPI process locally.

## Structs
- <strong>MtxData</strong> is the COO matrix structure, e.g. which is read into from an .mtx file.
- <strong>ScsData</strong> is the SELL-C-sigma matrix structure, e.g. which returned after a COO matrix has been processed with the <strong>convert_to_scs</strong> function.

## Functions
- <strong>convert_to_scs</strong>
	- Main routine. Converts a COO matrix (in this case, stored in the MtxData struct) into a Sell-C-sigma matrix. Parametarized by C and sigma.
- <strong>partition_precisions</strong>
	- Separate an MtxData struct into other, "smaller" MtxData structs to facilitate adaptive precision. 
- <strong>apply_permutation</strong>
	- Useful function to apply a permutation (e.g. the permutation vector obtained from permuting the rows/columns of your Sell-C-simga matrix) to a vector (e.g. the x vector for computing Ax=y) 
- <strong>permute_scs_cols</strong>
	- Permute the columns of your Sell-C-sigma matrix, i.e. for symmetric permutations
- <strong>uspmv_scs_(cpu/gpu)</strong>
	- The Sell-C-sigma SpMV kernel.
- <strong>uspmv_csr_(cpu/gpu)</strong>
	- A CRS format SpMV kernel to use as a reference.
- <strong>execute_spmv</strong>
	- Wrapper function to facilitate adaptive precision SpMV. 