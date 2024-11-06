# Ultimate-SpMV
## MPI+X SpMV with SELL-C-sigma format

Can be run as a standalone benchmarking harness, or as a library. See API_doc.md for information about the interface.

Examples:\
	```mpirun -n 4 ./uspmv <matrix_name>.mtx <kernel_format> <options>```\
	```./uspmv <matrix_name>.mtx scs -c 16 -s 512 -mode b```\
	```./uspmv <matrix_name>.mtx crs -mode s -ap[dp_sp] -ap_threshold_1 1000.0 -verbose 1```

- kernel_format can be any one of: crs, scs (and by extention: ell and sell-p)

Options:
- -c (int: chunk size (required for scs))
- -s (int: sigma (required for scs))
- -rev (int: number of back-to-back revisions to perform)
- -rand_x (0/1: random x vector option)
- -dp / sp / hp / ap[dp_sp] / ap[dp_hp] / ap[sp_hp] / ap[dp_sp_hp] (numerical precision of matrix data)
- -seg_nnz/seg_rows/seg_metis (global matrix partitioning for MPI)
- -validate (0/1: check result against MKL option)
- -verbose (0/1: verbose validation of results)
- -mode ('s'/'b': either in solve mode or bench mode)
- -bench_time (float: minimum number of seconds for SpMV benchmark)
- -ba_synch (0/1: synch processes each benchmark loop)
- -comm_halos (0/1: communicate halo elements each benchmark loop)
- -par_pack (0/1: pack elements contigously for MPI_Isend in parallel)
- -ap_threshold_1 (float: threshold for two-way matrix partitioning for adaptive precision `-ap`)
- -ap_threshold_2 (float: threshold for three-way matrix partitioning for adaptive precision `-ap`)
- -dropout (0/1: enable dropout of elements below theh designated threshold)
- -dropout_threshold (float: remove matrix elements below this range)
- -equilibrate (0/1: normalize rows of matrix)
 
 
Notes:
- This is a work in progress. Please email any problems or suggestions to `dane.c.lacey at fau.de`
- The -c and -s options are only relevant when the scs kernel is selected
- Dependencies:
	- mkl (used for validation of results)
- Select compiler in Makefile (gcc, icc, icx, llvm, nvcc)
	- icc is legacy, and not advised
- VECTOR_LENGTH for SIMD instructions is also defined at the top of the Makefile, useful for non-SELL_C_SIGMA kernels
- If using AVX512 on icelake, I currently get around downfall perf bug with the icx compiler from OneAPI 2023.2.0
- The par_pack option yields better performance for MPI+Openmp runs with poorly load balanced matrices
- Thresholds for adaptive precision are expected in the order `0---TH2---TH1---\infty`