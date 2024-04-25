# Ultimate-SpMV
## MPI+X SpMV with SELL-C-sigma format

Can be run as a standalone benchmarking harness, or as a library. See API_doc.md for information about the interface.

Example:\
	```mpirun -n 4 ./uspmv <matrix_name>.mtx <kernel_format> <options>```

- kernel_format can be any one of: crs, scs, ell

Options:
- -c (chunk size)
- -s (sigma)
- -rev (number of back-to-back revisions to perform)
- -rand_x (0/1 random x vector option)
- -sp/dp (numerical precision for spmv)
- -seg_nnz/seg_rows/seg_metis (global matrix partitioning)
- -validate (0/1 check result against mkl option)
- -verbose (0/1 verbose validation of results)
- -mode (s/b either in solve mode or bench mode)
- -bench_time (minimum number of seconds for SpMV benchmark)
- -ba_synch (0/1 synch processes each benchmark loop)
- -comm_halos (0/1 communication halo elements each benchmark loop)
- -par_pack (0/1 pack elements contigously for MPI_Isend in parallel)
 
 
Notes:
- kernel_format can be any one of: crs, scs, ell
- The -c and -s options are only relevant when the scs kernel is selected
- Dependencies:
	- mkl (used for validation of results)
	- intelmpi
- Select compiler in Makefile (icc, icx, gcc)
	- icc not advised (depreciated, known to seg fault in SCS kernel sometimes)
- VECTOR_LENGTH for SIMD instructions is also defined at the top of the Makefile, useful for non-SELL_C_SIGMA kernels
- If using AVX512 on icelake, I currently get around downfall perf bug with newest icx compiler using:
	- module use -a ~unrz139/.modules/modulefiles
	- module load oneapi/2023.2.0
	- module load compiler
	- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/saturn/unrz/unrz139/.modules/oneapi-2023.2.0/compiler/2023.2.0/linux/compiler/lib/intel64
- The par_pack option yields better performance for MPI+Openmp runs with poorly load balanced matrices