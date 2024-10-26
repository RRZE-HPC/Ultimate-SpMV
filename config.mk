# [gcc, icc, icx, llvm, nvcc]
COMPILER = icx
# [int]
VECTOR_LENGTH = 32 # CPU only
# [1/0]
DEBUG_MODE = 0
# [1/0]
DEBUG_MODE_FINE = 0
# [1/0]
OUTPUT_SPARSITY = 0
# [c++14]
CPP_VERSION = c++23

# [none, a40, a100]
GPGPU_ARCH = a100
# [int]
THREADS_PER_BLOCK=128 # GPU only

### External Libraries ###
# [1/0]
USE_MPI = 0

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
