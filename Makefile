include config.mk

# apply user-defined variables
CXXFLAGS += -DVECTOR_LENGTH=$(VECTOR_LENGTH) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK)


# Validate Makefile options (sanity checks)
ifneq ($(COMPILER),nvcc)
GPGPU_ARCH = none
endif

ifneq ($(GPGPU_ARCH),none)
ifneq ($(GPGPU_ARCH),a40)
ifneq ($(GPGPU_ARCH),a100)
$(info GPGPU_ARCH=$(GPGPU_ARCH))
$(error Please select a GPGPU architecture in: [none, a40, a100])
endif
endif
endif

ifneq ($(COMPILER),gcc)
ifneq ($(COMPILER),nvcc)
ifneq ($(COMPILER),icx)
ifneq ($(COMPILER),icc)
$(info COMPILER=$(COMPILER))
$(error Please select a compiler in: [gcc, icc, icx, nvcc])
endif
endif
endif
endif

ifeq ($(COMPILER),nvcc)
ifeq ($(USE_MPI),1)
$(error MPI is not currently supported for GPUs)
endif
endif

# compiler options
ifeq ($(COMPILER),gcc)
  CXX       = g++
  MPICXX     = mpicxx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
  CXXFLAGS += $(OPT_LEVEL) -std=$(CPP_VERSION) -Wall -fopenmp $(MKL) $(OPT_ARCH)

  LIBS += -L${MKLROOT}/lib/intel64 
endif

ifeq ($(COMPILER),icc)
  CXX       = icpc
  MPICXX     = mpiicpc
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -qmkl

  CXXFLAGS += $(OPT_LEVEL) -std=$(CPP_VERSION) -Wall -fopenmp $(MKL) $(OPT_ARCH)
endif

ifeq ($(COMPILER),icx)
  CXX       = icpx
  MPICXX     = mpiicpc -cxx=icpx
  OPT_LEVEL = -O3
  OPT_ARCH  = -xhost
  MKL = -qmkl
  AVX512_fix= -Xclang -target-feature -Xclang +prefer-no-gather -xCORE-AVX512 -qopt-zmm-usage=high

  CXXFLAGS += $(OPT_LEVEL) -std=$(CPP_VERSION) -Wall -fopenmp $(MKL) $(AVX512_fix) $(OPT_ARCH)
endif

ifeq ($(COMPILER),nvcc)
	ifneq ($(GPGPU_ARCH),none)
		ifeq ($(GPGPU_ARCH),a40)
			GPGPU_ARCH_FLAGS += -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp
		endif
		ifeq ($(GPGPU_ARCH),a100)
			GPGPU_ARCH_FLAGS += -gencode arch=compute_80,code=sm_80 -Xcompiler -fopenmp
		endif
	endif
	MKL += -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
	CXXFLAGS += $(MKL)
endif

ifeq ($(DEBUG_MODE),1)
	DEBUGFLAGS += -g -DDEBUG_MODE
ifneq ($(GPGPU_ARCH),none)
  DEBUGFLAGS += -G
endif
endif

ifeq ($(DEBUG_MODE_FINE),1)
	DEBUGFLAGS += -g -DDEBUG_MODE -DDEBUG_MODE_FINE
ifneq ($(GPGPU_ARCH),none)
  DEBUGFLAGS += -G
endif
endif

ifeq ($(OUTPUT_SPARSITY),1)
  CXXFLAGS += -DOUTPUT_SPARSITY
endif

ifeq ($(USE_METIS),1)
  ifeq ($(METIS_INC),)
    $(error USE_METIS selected, but no include path given in METIS_INC)
  endif
  ifeq ($(METIS_LIB),)
    $(error USE_METIS selected, but no library path given in METIS_LIB)
  endif

  CXXFLAGS  += -DUSE_METIS $(METIS_INC) $(GK_INC)
  LIBS += $(METIS_LIB) $(GK_LIB)
endif

ifeq ($(USE_LIKWID),1)
  # !!! include your own file paths !!! (I'm just loading module, which comes with file paths)
  ifeq ($(LIKWID_INC),)
    $(error USE_LIKWID selected, but no include path given in LIKWID_INC)
  endif
  ifeq ($(LIKWID_LIB),)
    $(error USE_LIKWID selected, but no library path given in LIKWID_LIB)
  endif
  CXXFLAGS  += -DUSE_LIKWID -DLIKWID_PERFMON $(LIKWID_INC) $(LIKWID_LIB) -llikwid
endif

ifeq ($(USE_CUSPARSE),1)
	CUDA_TOOLKIT=$(shell dirname $$(command -v nvcc))/..
	CUSPARSE_FLAGS += -I$(CUDA_TOOLKIT)/include
	GPGPU_ARCH_FLAGS += -lcusparse -DUSE_CUSPARSE
  CXXFLAGS += $(CUSPARSE_FLAGS) #$(NVJITPATH)
endif

ifeq ($(USE_MPI),1)
  CXXFLAGS  += -DUSE_MPI
else
  MPICXX = $(CXX)
endif

# Further memory debugging options
ifeq ($(ASAN),1)
  CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
endif

ifeq ($(UBSAN),1)
  CXXFLAGS += -fsanitize=undefined -g
endif

# Also rebuild when following files change.
REBUILD_DEPS = $(MAKEFILE_LIST) code/vectors.h code/timing.h code/classes_structs.hpp code/utilities.hpp code/kernels.hpp code/mpi_funcs.hpp code/write_results.hpp code/mmio.h

.PHONY: all
all: uspmv

uspmv: code/main.o code/mmio.o code/timing.o $(REBUILD_DEPS)
ifeq ($(COMPILER),nvcc)
	nvcc $(CXXFLAGS) $(GPGPU_ARCH_FLAGS) $(DEBUGFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)
else
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)
endif

code/main.o: code/main.cpp $(REBUILD_DEPS)
ifeq ($(COMPILER),nvcc)
	nvcc -x cu $(CXXFLAGS) $(GPGPU_ARCH_FLAGS) $(DEBUGFLAGS) -o $@ -c $<
else
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<
endif

code/timing.o: code/timing.c $(REBUILD_DEPS)
ifeq ($(COMPILER),nvcc)
	nvcc -x cu $(CXXFLAGS) $(GPGPU_ARCH_FLAGS) $(DEBUGFLAGS) -o $@ -c $<
else
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<
endif

code/mmio.o: code/mmio.cpp code/mmio.h
ifeq ($(COMPILER),nvcc)
	nvcc -x cu $(CXXFLAGS) $(GPGPU_ARCH_FLAGS) $(DEBUGFLAGS) -o $@ -c $<
else
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<
endif

TEST_INC_DIR = $(PWD)/code
TEST_PREFIX = code/test_suite

tests: $(TEST_PREFIX)/catch.o $(TEST_PREFIX)/tests.o $(TEST_PREFIX)/test_data/M0.o $(TEST_PREFIX)/test_data/M1.o $(TEST_PREFIX)/test_data/M_big.o $(TEST_PREFIX)/catch.hpp $(REBUILD_DEPS)
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) $(TEST_PREFIX)/tests.o $(TEST_PREFIX)/catch.o $(TEST_PREFIX)/test_data/M0.o $(TEST_PREFIX)/test_data/M1.o $(TEST_PREFIX)/test_data/M_big.o -o $(TEST_PREFIX)/tests

code/test_suite/catch.o: $(TEST_PREFIX)/catch.cpp test_suite/catch.hpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c $(TEST_PREFIX)/catch.cpp -o $(TEST_PREFIX)/catch.o

code/test_suite/tests.o: $(TEST_PREFIX)/tests.cpp $(TEST_PREFIX)/catch.hpp $(TEST_PREFIX)/test_data/test_data.hpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c $(TEST_PREFIX)/tests.cpp -o $(TEST_PREFIX)/tests.o

code/test_suite/test_data/M_big.o: $(TEST_PREFIX)/test_data/M_big.cpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c $(TEST_PREFIX)/test_data/M_big.cpp -o $(TEST_PREFIX)/test_data/M_big.o

code/test_suite/test_data/M0.o: $(TEST_PREFIX)/test_data/M0.cpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c $(TEST_PREFIX)/test_data/M0.cpp -o $(TEST_PREFIX)/test_data/M0.o

code/test_suite/test_data/M1.o: $(TEST_PREFIX)/test_data/M1.cpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c $(TEST_PREFIX)/test_data/M1.cpp -o $(TEST_PREFIX)/test_data/M1.o

.PHONY: clean
clean:
	-rm code/*.o

.PHONY: tests_clean
tests_clean:
	-rm $(TEST_PREFIX)/test_data/M0.o $(TEST_PREFIX)/test_data/M1.o $(TEST_PREFIX)/tests.o

.PHONY: rm
rm:
	-rm uspmv

