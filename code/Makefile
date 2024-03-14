COMPILER = icx
VECTOR_LENGTH = 4
USE_MPI = 0
DEBUG_MODE = 0
DEBUG_MODE_FINE = 0
OUTPUT_SPARSITY = 0

# 0/1 library usage
USE_METIS = 
# USE_SPMP = 0
USE_LIKWID = 0

# compiler options
ifeq ($(COMPILER),gcc)
  CXX       = g++
  MPICXX     = mpicxx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -Wall -fopenmp $(MKL) $(OPT_ARCH) -DVECTOR_LENGTH=$(VECTOR_LENGTH)

  LIBS += -L${MKLROOT}/lib/intel64 
endif

ifeq ($(COMPILER),icc)
  CXX       = icpc
  MPICXX     = mpiicpc
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -qmkl

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -Wall -fopenmp $(MKL) $(OPT_ARCH) -DVECTOR_LENGTH=$(VECTOR_LENGTH)
endif

ifeq ($(COMPILER),icx)
  CXX       = icpx
  MPICXX     = mpiicpc -cxx=icpx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -xhost
  MKL = -qmkl
  AVX512_fix= #-Xclang -target-feature -Xclang +prefer-no-gather -xCORE-AVX512 -qopt-zmm-usage=high

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -Wall -fopenmp $(MKL) $(AVX512_fix) $(OPT_ARCH) -DVECTOR_LENGTH=$(VECTOR_LENGTH)
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
  METIS_INC = -I/home/hpc/ihpc/ihpc062h/install/include/
  METIS_LIB = /home/hpc/ihpc/ihpc062h/install/lib/libmetis.a
  ifeq ($(METIS_INC),)
    $(error USE_METIS selected, but no include path given in METIS_INC)
  endif
  ifeq ($(METIS_LIB),)
    $(error USE_METIS selected, but no library path given in METIS_LIB)
  endif

  CXXFLAGS  += -DUSE_METIS $(METIS_INC)
  LIBS += $(METIS_LIB)
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

# ifeq ($(USE_SPMP),1)
#   SPMP_INC =
#   SPMP_LIB = /home/hpc/ihpc/ihpc062h/HPC_HiWi/thesis/SPMP/SpMP/libspmp.a
#   ifeq ($(SPMP_INC),)
#     $(error USE_SPMP selected, but no include path given in SPMP_INC)
#   endif
#   ifeq ($(SPMP_LIB),)
#     $(error USE_SPMP selected, but no library path given in SPMP_LIB)
#   endif
#   CXXFLAGS  += -DUSE_SPMP $(SPMP_INC) $(SPMP_LIB)
# endif

# Further memory debugging options
ifeq ($(ASAN),1)
  CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
endif

ifeq ($(UBSAN),1)
  CXXFLAGS += -fsanitize=undefined -g
endif

CXXFLAGS += $(HEADERS)
# Also rebuild when following files change.
REBUILD_DEPS = $(MAKEFILE_LIST) vectors.h classes_structs.hpp utilities.hpp kernels.hpp mpi_funcs.hpp write_results.hpp mmio.h

.PHONY: all
all: omp

.PHONY: omp
omp: uspmv

# uspmv: main.o mmio.o timing.o $(REBUILD_DEPS)
# 	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)

# main.o: main.cpp $(REBUILD_DEPS)
# 	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

# timing.o: timing.c $(REBUILD_DEPS)
# 	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

# mmio.o: mmio.cpp mmio.h
# 	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

uspmv: main.o mmio.o timing.o $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)

main.o: main.cpp $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

timing.o: timing.c $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

mmio.o: mmio.cpp mmio.h
	$(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ -c $<

TEST_INC_DIR = $(PWD)

tests: test_suite/catch.o test_suite/tests.o test_suite/test_data/M0.o test_suite/test_data/M1.o test_suite/test_data/M_big.o test_suite/catch.hpp $(REBUILD_DEPS)
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) test_suite/tests.o test_suite/catch.o test_suite/test_data/M0.o test_suite/test_data/M1.o test_suite/test_data/M_big.o -o test_suite/tests

test_suite/catch.o: test_suite/catch.cpp test_suite/catch.hpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c test_suite/catch.cpp -o test_suite/catch.o

test_suite/tests.o: test_suite/tests.cpp test_suite/catch.hpp test_suite/test_data/test_data.hpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c test_suite/tests.cpp -o test_suite/tests.o

test_suite/test_data/M_big.o: test_suite/test_data/M_big.cpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c test_suite/test_data/M_big.cpp -o test_suite/test_data/M_big.o

test_suite/test_data/M0.o: test_suite/test_data/M0.cpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c test_suite/test_data/M0.cpp -o test_suite/test_data/M0.o

test_suite/test_data/M1.o: test_suite/test_data/M1.cpp
	$(CXX) $(DEBUGFLAGS) -I$(TEST_INC_DIR) -c test_suite/test_data/M1.cpp -o test_suite/test_data/M1.o

.PHONY: clean
clean:
	-rm *.o

.PHONY: tests_clean
tests_clean:
	-rm test_suite/test_data/M0.o test_suite/test_data/M1.o test_suite/tests.o

.PHONY: rm
rm:
	-rm uspmv

