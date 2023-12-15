# Ultimate-SpMV
MPI+X SpMV with SELL-C-sigma format

Select compiler in Makefile (icc, icx, gcc)

Dependencies:
	mkl (used for validation of results)
	intelmpi

Example:
	mpirun -n 4 ./uspmv matrix.mtx <options>

Options:
	-c (chunk size)
	-s (sigma)
	-rev (number of back-to-back revisions to perform)
	-rand_x (0/1 random x vector option)
	-sp/dp (numerical precision for spmv)
	-seg_nnz/seg_rows/seg_metis (global matrix partitioning)
	-validate (0/1 check result against mkl option)
	-verbose (0/1 verbose validation of results)
	-mode (s/b either in solve mode or bench mode)
	-comm_halos (0/1 communicate halo elements option)
