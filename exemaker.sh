#!/bin/bash -l

module load intel
module load intelmpi
module load likwid
module load mkl

RED="\e[31m"
GREEN="\e[32m"
RESET="\e[0m"

exedir="./EXE"

[ ! -d "$exedir" ] && mkdir -p $exedir

get_config_value() {
  grep -i "$1 =" config.mk | awk '{print $NF}'
}

update_config_value() {
  local key=$1
  local value=$2
  sed -i 's/^\(# \?\)\?'"$key"' = .*/'"$key"' = '"$value"'/' config.mk
}

#collecting all old values
OLD_COMPILER=$(get_config_value "COMPILER")
OLD_SIMD_LENGTH=$(get_config_value "SIMD_LENGTH")
OLD_BLOCK_VECTOR_LAYOUT=$(get_config_value "BLOCK_VECTOR_LAYOUT")
OLD_MPI_MODE=$(get_config_value "MPI_MODE")
OLD_DEBUG_MODE=$(get_config_value "DEBUG_MODE")
OLD_DEBUG_MODE_FINE=$(get_config_value "DEBUG_MODE_FINE")
OLD_OUTPUT_SPARSITY=$(get_config_value "OUTPUT_SPARSITY")
OLD_CPP_VERSION=$(get_config_value "CPP_VERSION")
OLD_GPGPU_ARCH=$(get_config_value "GPGPU_ARCH")
OLD_THREADS_PER_BLOCK=$(get_config_value "THREADS_PER_BLOCK") # GPU only
OLD_USE_OPENMP=$(get_config_value "USE_OPENMP")
OLD_USE_MKL=$(get_config_value "USE_MKL")
OLD_USE_MPI=$(get_config_value "USE_MPI")
OLD_USE_METIS=$(get_config_value "USE_METIS")
OLD_USE_LIKWID=$(get_config_value "USE_LIKWID")
OLD_USE_CUSPARSE=$(get_config_value "USE_CUSPARSE")
OLD_USE_SCAMAC=$(get_config_value "USE_SCAMAC")

#setting to default values of config.mk
update_config_value "COMPILER" "icx"
update_config_value "SIMD_LENGTH" "4"
update_config_value "BLOCK_VECTOR_LAYOUT" "rowwise"
update_config_value "MPI_MODE" "singlevec"
update_config_value "DEBUG_MODE" "0"
update_config_value "DEBUG_MODE_FINE" "0"
update_config_value "OUTPUT_SPARSITY" "0"
update_config_value "CPP_VERSION" "c++14"
update_config_value "GPGPU_ARCH" "none"
update_config_value "THREADS_PER_BLOCK" "128"
update_config_value "USE_OPENMP" "0"
update_config_value "USE_MKL" "0"
update_config_value "USE_MPI" "0"
update_config_value "USE_METIS" "0"
update_config_value "USE_LIKWID" "0"
update_config_value "USE_CUSPARSE" "0"
update_config_value "USE_SCAMAC" "0"

declare -a mpi_modes=("singlevec" "bulkvec") # "multivec"  # removed it due to issues in intelmpi implementation issues
declare -a active_modes=(1)
declare -a deactive_modes=(0)
declare -a binarymodes=(0)

for mpi_mode in "${mpi_modes[@]}"; do
  update_config_value "MPI_MODE" "$mpi_mode"
  for mkl_set in "${binarymodes[@]}"; do
    update_config_value "USE_MKL" "$mkl_set"
    for mpi_set in "${binarymodes[@]}"; do
      update_config_value "USE_MPI" "$mpi_set"
      exename=uspmv_${mpi_mode}_MKL${mkl_set}_MPI${mpi_set}
      make clean >/dev/null 2>&1 && make -j$(nproc) >/dev/null 2>&1
      if [ $? -eq 0 ]; then
        mv uspmv $exedir/$exename
        echo -e "$exename executable creation ${GREEN}sucess${RESET}"
      else
        echo -e "$exename executable creation ${RED}failed${RESET}"
      fi
    done
  done
done

make clean >/dev/null 2>&1

#reseting to old value of config.mk
update_config_value "COMPILER" "$OLD_COMPILER"
update_config_value "SIMD_LENGTH" "$OLD_SIMD_LENGTH"
update_config_value "BLOCK_VECTOR_LAYOUT" "$OLD_BLOCK_VECTOR_LAYOUT"
update_config_value "MPI_MODE" "$OLD_MPI_MODE"
update_config_value "DEBUG_MODE" "$OLD_DEBUG_MODE"
update_config_value "DEBUG_MODE_FINE" "$OLD_DEBUG_MODE_FINE"
update_config_value "OUTPUT_SPARSITY" "$OLD_OUTPUT_SPARSITY"
update_config_value "CPP_VERSION" "$OLD_CPP_VERSION"
update_config_value "GPGPU_ARCH" "$OLD_GPGPU_ARCH"
update_config_value "THREADS_PER_BLOCK" "$OLD_THREADS_PER_BLOCK"
update_config_value "USE_OPENMP" "$OLD_USE_OPENMP"
update_config_value "USE_MKL" "$OLD_USE_MKL"
update_config_value "USE_MPI" "$OLD_USE_MPI"
update_config_value "USE_METIS" "$OLD_USE_METIS"
update_config_value "USE_LIKWID" "$OLD_USE_LIKWID"
update_config_value "USE_CUSPARSE" "$OLD_USE_CUSPARSE"
update_config_value "USE_SCAMAC" "$OLD_USE_SCAMAC"
