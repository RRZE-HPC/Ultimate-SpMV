COMPILER ?= gcc

ifeq ($(COMPILER),gcc)
  CXX       = g++
  MPICXX     = mpic++

  # OPT_LEVEL = -O0 -g -ggdb3
  # OPT_LEVEL = -Ofast -g -ggdb3
  OPT_LEVEL = -O1
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
REBUILD_DEPS = $(MAKEFILE_LIST) spmv.h vectors.h mtx-reader.h utilities.hpp kernels.hpp results.hpp

.PHONY: all
all: omp

.PHONY: omp
omp: spmv-omp

# .PHONY: omp-adv
# omp-adv: spmv-omp-adv

spmv-omp: main-omp.o spmv-omp-adv.o $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)

# spmv-omp-adv: main-omp.o spmv-omp-adv.o splitSendMtxData.o $(REBUILD_DEPS)
# 	$(MPICXX) $(CXXFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)

main-omp.o: main.cpp $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -o $@ -c $<


spmv-omp.o: spmv-ompOG.cpp $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -o $@ -c $< $(CXXFLAGS_LISTINGS)

spmv-omp-adv.o: spmv-omp-adv.cpp $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -o $@ -c $< $(CXXFLAGS_LISTINGS)


.PHONY: clean
clean:
	-rm *.o

.PHONY: rm
rm:
	-rm spmv-omp
	-rm spmv-omp-adv

