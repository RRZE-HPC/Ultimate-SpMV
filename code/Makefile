COMPILER ?= icc

ifeq ($(COMPILER),gcc)
  CXX       = g++
  MPICXX     = mpic++

  # OPT_LEVEL = -O0 -g -ggdb3
  # OPT_LEVEL = -Ofast -g -ggdb3
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -g -Wall -fopenmp $(OPT_ARCH)

  ifeq ($(ASAN),1)
    CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
  endif

  ifeq ($(UBSAN),1)
    CXXFLAGS += -fsanitize=undefined -g
  endif
endif

ifeq ($(COMPILER),icc)
  CXX       = icpc
  MPICXX     = mpiicpc

  # OPT_LEVEL = -O0 -g -ggdb3
  # OPT_LEVEL = -Ofast -g -ggdb3
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -g -Wall -fopenmp $(OPT_ARCH)

  ifeq ($(ASAN),1)
    CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
  endif

  ifeq ($(UBSAN),1)
    CXXFLAGS += -fsanitize=undefined -g
  endif
endif

ifeq ($(LISTINGS),1)
  CXXFLAGS_LISTINGS += -Wa,-alnds=$@.lst -g
  ifeq ($(COMPILER),clang)
    CXXFLAGS_LISTINGS += -no-integrated-as
  else
  ifeq ($(COMPILER),intel-ng)
    CXXFLAGS_LISTINGS += -no-integrated-as
  endif
  endif
endif

ifeq ($(USE_LIKWID),1)
  CXXFLAGS  += -DUSE_LIKWID $(LIKWID_INC) $(LIKWID_LIB)
  LIBS      += -llikwid
endif

# Also rebuild when following files change.
REBUILD_DEPS = $(MAKEFILE_LIST) vectors.h mtx_reader.h classes_structs.hpp utilities.hpp kernels.hpp mpi_funcs.hpp write_results.hpp

.PHONY: all
all: omp

.PHONY: omp
omp: spmv-omp

spmv-omp: main.o $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -qmkl -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS) 

main.o: main.cpp $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -o $@ -c $<

# main.o: main.cpp $(REBUILD_DEPS)
# 	$(MPICXX) $(CXXFLAGS) $(MKLLINKS)-o $@ -c $<

.PHONY: clean
clean:
	-rm *.o

.PHONY: rm
rm:
	-rm spmv-omp

