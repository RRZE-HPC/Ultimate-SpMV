COMPILER ?= icx

ifeq ($(COMPILER),gcc)
  CXX       = g++
  MPICXX     = mpicxx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -g -Wall -fopenmp $(MKL) $(OPT_ARCH)

  LIBS += -L${MKLROOT}/lib/intel64 

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
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  MKL = -qmkl
  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -g -Wall -fopenmp $(MKL) $(OPT_ARCH)

  ifeq ($(ASAN),1)
    CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
  endif

  ifeq ($(UBSAN),1)
    CXXFLAGS += -fsanitize=undefined -g
  endif
endif

ifeq ($(COMPILER),icx)
  CXX       = icpx
  MPICXX     = mpiicpc -cxx=icpx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -xhost
  MKL = -qmkl
  AVX512_fix= -Xclang -target-feature -Xclang +prefer-no-gather -xCORE-AVX512 -qopt-zmm-usage=high

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -Wall -fopenmp -g $(MKL) $(AVX512_fix) $(OPT_ARCH) 

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
REBUILD_DEPS = $(MAKEFILE_LIST) vectors.h classes_structs.hpp utilities.hpp kernels.hpp mpi_funcs.hpp write_results.hpp mmio.h

.PHONY: all
all: omp

.PHONY: omp
omp: uspmv

uspmv: main.o mmio.o $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS) 

main.o: main.cpp $(REBUILD_DEPS)
	$(MPICXX) $(CXXFLAGS) -o $@ -c $<

mmio.o: mmio.cpp mmio.h
	$(MPICXX) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean
clean:
	-rm *.o

.PHONY: rm
rm:
	-rm uspmv

