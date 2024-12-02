# [gcc, icc, icx, llvm, nvcc]
COMPILER = icx
# [int] (CPU only)
SIMD_LENGTH = 4
# [colwise, rowwise] (If using single vector SpMV, use "colwise")
BLOCK_VECTOR_LAYOUT = colwise
# [1/0]
DEBUG_MODE = 0
# [1/0]
DEBUG_MODE_FINE = 0
# [1/0]
OUTPUT_SPARSITY = 0
# [c++14]
CPP_VERSION = c++14
# [none, a40, a100] (GPU only)
GPGPU_ARCH = none
# [int] (GPU only)
THREADS_PER_BLOCK=128

### External Libraries ###
# [1/0]
USE_MKL = 0
# MKL_INC = 
# MKL_LIB = 

# [1/0]
USE_MPI = 0
# MPI_INC = 
# MPI_LIB = 

# [1/0]
USE_METIS = 0
# METIS_INC = 
# METIS_LIB = 
# GK_INC = 
# GK_LIB =

# [1/0]
USE_LIKWID = 0
# LIKWID_INC =
# LIKWID_LIB = 

# [1/0]
USE_CUSPARSE = 0
# NVJITPATH = 
