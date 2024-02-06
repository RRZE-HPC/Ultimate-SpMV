#!/bin/bash -l
#SBATCH -J node_perf_tests
#SBATCH -N 1
#SBATCH --cpus-per-task=18
#SBATCH -t 08:00:00
#SBATCH --exclusive
#SBATCH --output=/home/hpc/ihpc/ihpc062h/HPC_HiWi/batchScripts/results/%j_%x.out

unset SLURM_EXPORT_ENV 

# new modules!, gather-less intel compiler
module use -a ~unrz139/.modules/modulefiles
module load oneapi/2023.2.0
module load compiler
module load intelmpi
module load mkl
module load cmake
module load python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc/ihpc/ihpc062h/install/lib/RACE/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc/ihpc/ihpc062h/HPC_HiWi/RACE/build/hwloc-inst/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/saturn/unrz/unrz139/.modules/oneapi-2023.2.0/compiler/2023.2.0/linux/compiler/lib/intel64

# 72 physical cores/node, 2 sockets/node, 2 ccNuma domain/socket => 72/4 = 18 cores/process if pinning to ccNuma
export I_MPI_PIN=1
export I_MPI_PIN_PROCESSOR_LIST="allcores"
export I_MPI_PIN_DOMAIN="18:compact"

export OMP_NUM_THREADS=18 # 18 procs/node with 4 threads each => 72 physical cores
export OMP_PLACES=cores
export OMP_PROC_BIND=close

declare -a matrices=(
    # "/home/vault/ihpc/ihpc062h/bench_matrices/thermal2.mtx" # 1_228_045, 8_580_313, 8_580_313, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/crankseg_1.mtx" # 52_804, 10_614_210, 10_,614_210, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/pwtk.mtx" # 217_918, 11_524_432, 11_634_424, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/rajat31.mtx" # 4_690_002, 20_316_253, 20_316_253, N
    # "/home/vault/ihpc/ihpc062h/bench_matrices/gsm_106857.mtx" # 589_446, 21_758_924, 21_758_924, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/F1.mtx" # 343_791, 26_837_113, 26_837_113, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/cage14.mtx" # 1_505_785, 27_130_349, 27_130_349, N
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Fault_639.mtx" # 638_802, 27_245_944, 28_614_564, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/inline_1.mtx" # 503_712, 36_816_170, 36_816_342, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Emilia_923.mtx" # 923_136, 40_373_538, 41_005_206, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/ldoor.mtx" # 952_203, 42_493_817, 46_522_475, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/af_shell10.mtx" # 1_508_065, 52_259_885, 52_672_325, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Hook_1498.mtx" # 1_498_023, 59_374_451, 60_917_445, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Geo_1438.mtx" # 1_437_960, 60_236_322, 63_156_690, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Serena.mtx" # 1_391_349, 64_131_971, 64_531_701, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/bone010.mtx" # 986_703, 47_851_783, 71_666_325, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/audikw_1.mtx" # 943_695, 77_651_847, 77_651_847, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/channel-500x100x100-b050.mtx" # 4_802_000, 85_362_744, 85_362_744, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Long_Coup_dt0.mtx" # 1,470,152 84,422,970 87,088,992
    # "/home/vault/ihpc/ihpc062h/bench_matrices/dielFilterV3real.mtx" # 1_102_824, 89_306_020, 89_306_020, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/nlpkkt120.mtx" # 3_542_400, 95_117_792, 96_845_792, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/ML_Geer.mtx" # 1_504_002, 110_686_677, 110_879_972, N
    # "/home/woody/unrz/unrz002h/matrices/Heart_simulation_Langguth/Lynx68.mtx" # 6,811,350 ?  111,560,826
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Flan_1565.mtx" # 1_564_794, 114_165_372, 117_406_044, Y
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Cube_Coup_dt0.mtx" # 2,164,760 124,406,070 127,206,144
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Bump_2911.mtx" # 2,911,419 127,729,899 127,729,899 
    # "/home/vault/ihpc/ihpc062h/bench_matrices/vas_stokes_4M.mtx" # 4,382,246 131,577,616 131,577,616
    # "/home/vault/ihpc/ihpc062h/bench_matrices/HV15R.mtx" # 2,017,169 283,073,458 283,073,458
    # "/home/vault/ihpc/ihpc062h/bench_matrices/Queen_4147.mtx" # 4,147,110 316,548,962 329,499,284
    # "/home/vault/ihpc/ihpc062h/bench_matrices/stokes.mtx" # 11,449,533 349,321,980 349,321,980
    # "/home/vault/ihpc/ihpc062h/bench_matrices/nlpkkt200.mtx" # 16,240,000 440,225,632, 448,225,632
    # "/home/vault/ihpc/ihpc062h/bench_matrices/nlpkkt240.mtx" # 27,993,600 760,648,352 774,472,352
    # "/home/woody/unrz/unrz002h/matrices/Heart_simulation_Langguth/Lynx649_reordered.mtx" # 64,950,632 ? 978,866,282 
    # # "/home/woody/unrz/unrz002h/matrices/Heart_simulation_Langguth/Lynx1151.mtx"
)