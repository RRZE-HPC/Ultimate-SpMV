# Ultimate-SpMV
## MPI+X SpMV with SELL-C-sigma format

Example:\
	```mpirun -n 4 ./uspmv matrix.mtx <options>```

Options:
- c (chunk size)
- s (sigma)
- rev (number of back-to-back revisions to perform)
- rand_x (0/1 random x vector option)
- sp/dp (numerical precision for spmv)
- seg_nnz/seg_rows/seg_metis (global matrix partitioning)
- validate (0/1 check result against mkl option)
- verbose (0/1 verbose validation of results)
- mode (s/b either in solve mode or bench mode)
- ba_synch (0/1 synch processes each benchmark loop)
- comm_halos (0/1 communication halo elements each benchmark loop)
 
Notes:
- Dependencies:
	- mkl (used for validation of results)
	- intelmpi
- Select compiler in Makefile (icc, icx, gcc)
- For each external library, make sure you are adding the file paths for both the header file and library in the Makefile
- VECTOR_LENGTH for SIMD instructions is also defined at the top of the Makefile, useful for non-SELL_C_SIGMA kernels
- If using AVX512 on icelake, I currently get around downfall perf bug with newest icx compiler using:
	- module use -a ~unrz139/.modules/modulefiles
	- module load oneapi/2023.2.0
	- module load compiler
	- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/saturn/unrz/unrz139/.modules/oneapi-2023.2.0/compiler/2023.2.0/linux/compiler/lib/intel64

TODO features:
- CMake building for automatic file path detection
- SCAMAC (seq and parallel)
- kernel picker (with no perf degredation)
- local copies of structs to avoid data placement problems with large matrices
