COMPILER = gcc
VECTOR_LENGTH = 4
DEBUG_MODE = 0
DEBUG_MODE_FINE = 0
OUTPUT_SPARSITY = 0
CPP_VERSION=c++14

# 0/1 library usage
USE_MPI = 0
USE_METIS = 0
USE_LIKWID = 0

# compiler options
ifeq ($(COMPILER),gcc)
  CXX       = g++
  MPICXX     = mpicxx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
  CXXFLAGS += $(OPT_LEVEL) -std=$(CPP_VERSION) -Wall -fopenmp $(MKL) $(OPT_ARCH) -DVECTOR_LENGTH=$(VECTOR_LENGTH)

  LIBS += -L${MKLROOT}/lib/intel64 
endif

ifeq ($(COMPILER),icc)
  CXX       = icpc
  MPICXX     = mpiicpc
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -qmkl

  CXXFLAGS += $(OPT_LEVEL) -std=$(CPP_VERSION) -Wall -fopenmp $(MKL) $(OPT_ARCH) -DVECTOR_LENGTH=$(VECTOR_LENGTH)
endif

ifeq ($(COMPILER),icx)
  CXX       = icpx
  MPICXX     = mpiicpc -cxx=icpx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -xhost
  MKL = -qmkl
  AVX512_fix= -Xclang -target-feature -Xclang +prefer-no-gather -xCORE-AVX512 -qopt-zmm-usage=high

  CXXFLAGS += $(OPT_LEVEL) -std=$(CPP_VERSION) -Wall -fopenmp $(MKL) $(AVX512_fix) $(OPT_ARCH) -DVECTOR_LENGTH=$(VECTOR_LENGTH)
endif

ifeq ($(DEBUG_MODE),1)
  DEBUGFLAGS += -g -DDEBUG_MODE
endif

ifeq ($(DEBUG_MODE_FINE),1)
  DEBUGFLAGS += -g -DDEBUG_MODE -DDEBUG_MODE_FINE
endif

ifeq ($(OUTPUT_SPARSITY),1)
  CXXFLAGS += -DOUTPUT_SPARSITY
endif

ifeq ($(USE_METIS),1)
  # !!! include your own file paths !!!
  # For example: (I've had better luck with static libraries)
  # METIS_INC = -I/home/hpc/k107ce/k107ce17/install/include/
  # METIS_LIB = /home/hpc/k107ce/k107ce17/install/lib/libmetis.a 
  # GK_INC = -I/home/hpc/k107ce/k107ce17/install/include/
  # GK_LIB = /home/hpc/k107ce/k107ce17/install/lib/libGKlib.a
  METIS_INC = 
  METIS_LIB = 
  GK_INC = 
  GK_LIB = 
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
  # LIKWID_INC =
  # LIKWID_LIB = 
  ifeq ($(LIKWID_INC),)
    $(error USE_LIKWID selected, but no include path given in LIKWID_INC)
  endif
  ifeq ($(LIKWID_LIB),)
    $(error USE_LIKWID selected, but no library path given in LIKWID_LIB)
  endif
  CXXFLAGS  += -DUSE_LIKWID -DLIKWID_PERFMON $(LIKWID_INC) $(LIKWID_LIB) -llikwid
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

CXXFLAGS += $(HEADERS)
# Also rebuild when following files change.
REBUILD_DEPS = $(MAKEFILE_LIST) code/vectors.h code/classes_structs.hpp code/utilities.hpp code/kernels.hpp code/mpi_funcs.hpp code/write_results.hpp code/mmio.h

.PHONY: all
all: uspmv_no_mpi

uspmv_no_mpi: code/main.o code/mmio.o code/timing.o $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)

code/main.o: code/main.cpp $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

code/timing.o: code/timing.c $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

code/mmio.o: code/mmio.cpp code/mmio.h
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

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

