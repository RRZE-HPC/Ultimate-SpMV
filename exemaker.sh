#!/bin/bash -l

module load intel
module load intelmpi
module load mkl

exedir="./EXE"

if [ ! -d "$exedir" ]
then
    mkdir -p $exedir
fi


OLD_COMPILER=$(grep -i "COMPILER =" config.mk | awk '{print $NF}')  
OLD_SIMD_LENGTH=$(grep -i "SIMD_LENGTH =" config.mk | awk '{print $NF}')  
OLD_BLOCK_VECTOR_LAYOUT=$(grep -i "BLOCK_VECTOR_LAYOUT =" config.mk | awk '{print $NF}')  
OLD_DEBUG_MODE=$(grep -i "DEBUG_MODE =" config.mk | awk '{print $NF}')  
OLD_DEBUG_MODE_FINE=$(grep -i "DEBUG_MODE_FINE =" config.mk | awk '{print $NF}')  
OLD_OUTPUT_SPARSITY=$(grep -i "OUTPUT_SPARSITY =" config.mk | awk '{print $NF}')  
OLD_CPP_VERSION=$(grep -i "CPP_VERSION =" config.mk | awk '{print $NF}')  
OLD_GPGPU_ARCH=$(grep -i "GPGPU_ARCH =" config.mk | awk '{print $NF}')  
OLD_THREADS_PER_BLOCK=$(grep -i "THREADS_PER_BLOCK =" config.mk | awk '{print $NF}')   # GPU only
OLD_USE_MKL=$(grep -i "USE_MKL =" config.mk | awk '{print $NF}')  
OLD_USE_MPI=$(grep -i "USE_MPI =" config.mk | awk '{print $NF}')  
OLD_USE_METIS=$(grep -i "USE_METIS =" config.mk | awk '{print $NF}')  
OLD_USE_LIKWID=$(grep -i "USE_LIKWID =" config.mk | awk '{print $NF}')  
OLD_USE_CUSPARSE=$(grep -i "USE_CUSPARSE =" config.mk | awk '{print $NF}')  

sed -i 's/^\(# \?\)\?USE_MKL = .*/USE_MKL = '"1"'/' config.mk

#MPI off mkl on
sed -i 's/^\(# \?\)\?USE_MPI = .*/USE_MPI = '"0"'/' config.mk
make
mv uspmv ./EXE/uspmv_mkl1_mpi0

#MPI on mkl on
sed -i 's/^\(# \?\)\?USE_MPI = .*/USE_MPI = '"1"'/' config.mk
make
mv uspmv ./EXE/uspmv_mkl1_mpi1




sed -i 's/^\(# \?\)\?COMPILER = .*/COMPILER = '"$OLD_COMPILER"'/' config.mk
sed -i 's/^\(# \?\)\?SIMD_LENGTH = .*/SIMD_LENGTH = '"$OLD_SIMD_LENGTH"'/' config.mk
sed -i 's/^\(# \?\)\?BLOCK_VECTOR_LAYOUT = .*/BLOCK_VECTOR_LAYOUT = '"$OLD_BLOCK_VECTOR_LAYOUT"'/' config.mk
sed -i 's/^\(# \?\)\?DEBUG_MODE = .*/DEBUG_MODE = '"$OLD_DEBUG_MODE"'/' config.mk
sed -i 's/^\(# \?\)\?DEBUG_MODE_FINE = .*/DEBUG_MODE_FINE = '"$OLD_DEBUG_MODE_FINE"'/' config.mk
sed -i 's/^\(# \?\)\?OUTPUT_SPARSITY = .*/OUTPUT_SPARSITY = '"$OLD_OUTPUT_SPARSITY"'/' config.mk
sed -i 's/^\(# \?\)\?CPP_VERSION = .*/CPP_VERSION = '"$OLD_CPP_VERSION"'/' config.mk
sed -i 's/^\(# \?\)\?GPGPU_ARCH = .*/GPGPU_ARCH = '"$OLD_GPGPU_ARCH"'/' config.mk
sed -i 's/^\(# \?\)\?THREADS_PER_BLOCK = .*/THREADS_PER_BLOCK = '"$OLD_THREADS_PER_BLOCK"'/' config.mk
sed -i 's/^\(# \?\)\?USE_MKL = .*/USE_MKL = '"$OLD_USE_MKL"'/' config.mk
sed -i 's/^\(# \?\)\?USE_MPI = .*/USE_MPI = '"$OLD_USE_MPI"'/' config.mk
sed -i 's/^\(# \?\)\?USE_METIS = .*/USE_METIS = '"$OLD_USE_METIS"'/' config.mk
sed -i 's/^\(# \?\)\?USE_LIKWID = .*/USE_LIKWID = '"$OLD_USE_LIKWID"'/' config.mk
sed -i 's/^\(# \?\)\?USE_CUSPARSE = .*/USE_CUSPARSE = '"$OLD_USE_CUSPARSE"'/' config.mk
