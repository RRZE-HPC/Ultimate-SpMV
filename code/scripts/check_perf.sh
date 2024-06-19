#!/bin/bash -l
#SBATCH -J node_perf_tests
#SBATCH -N 1
#--cpus-per-task=18
#SBATCH -t 03:00:00
#SBATCH --exclusive
#SBATCH -c 72
#SBATCH --constraint=hwperf
#SBATCH --cpu-freq=2000000-2000000:performance 
#SBATCH --output=/home/hpc/k107ce/k107ce17/Ultimate-SpMV/code/scripts/results/%j_%x.out

unset SLURM_EXPORT_ENV 

# new modules!, gather-less intel compiler
module load intel
module load likwid

# 72 physical cores/node, 2 sockets/node, 2 ccNuma domain/socket => 72/4 = 18 cores/process if pinning to ccNuma
# export I_MPI_PIN=1
# export I_MPI_PIN_PROCESSOR_LIST="allcores"
# export I_MPI_PIN_DOMAIN="18:compact"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

declare -a matrices=(
    "/home/vault/k107ce/k107ce17/bench_matrices/Long_Coup_dt0.mtx"
    "/home/vault/k107ce/k107ce17/bench_matrices/ML_Geer.mtx"
    "/home/vault/k107ce/k107ce17/bench_matrices/af_shell10.mtx"
)

# declare -a datatypes=("-dp" "mp")

declare -a datatypes=("-dp")

# Validate Accuracy
# for matrix in "${matrices[@]}";
# do
#     export OMP_NUM_THREADS=18
#     srun --cpu-freq=2000000-2000000 ./../../uspmv_cpu $matrix crs -mode s $datatype
# done

# Bench: Scale only in one numa domain
for matrix in "${matrices[@]}";
do
    for datatype in "${datatypes[@]}";
    do
        # for core_count in {18..1};
        # do
            export OMP_NUM_THREADS=1
        #     for med_loop in {1..5};
        #     do
                # srun --cpu-freq=2000000-2000000:performance ./../../uspmv_icx_seq_ofast_0_simd_1 $matrix crs -rand_x 0 -mode s $datatype
                # srun --cpu-freq=2000000-2000000:performance ./../../uspmv_icx_seq_ofast_m_simd_1 $matrix crs -rand_x m -mode s $datatype
                # srun --cpu-freq=2000000-2000000:performance ./../../uspmv_icx_seq_ofast_1_simd_1 $matrix crs -rand_x 1 -mode s $datatype
                srun --cpu-freq=2000000-2000000:performance ./../../uspmv_icx_seq_ofast_0_simd_4 $matrix crs -rand_x 0 -mode s $datatype
                srun --cpu-freq=2000000-2000000:performance ./../../uspmv_icx_seq_ofast_m_simd_4 $matrix crs -rand_x m -mode s $datatype
                srun --cpu-freq=2000000-2000000:performance ./../../uspmv_icx_seq_ofast_1_simd_4 $matrix crs -rand_x 1 -mode s $datatype
                # srun --cpu-freq=2000000-2000000:performance ./../../uspmv_cpu_ap $matrix crs -mode b $datatype  -bucket_size 3.08e7

        #     done
        # done
    done
done

# for matrix in "${matrices[@]}";
# do
#     for datatype in "${datatypes[@]}";
#     do
#         core_count=17
#         # for core_count in {0..17};
#         # do
#             # Not necessary with likwid
#             # export OMP_NUM_THREADS=$core_count
#             # for med_loop in {1..3};
#             # do
# #                 srun --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g FLOPS_DP -m ./../../uspmv_cpu_likwid $matrix crs -mode b $datatype >> uspmv_cpu_likwid_flops.txt
# #                 srun --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g FLOPS_SP -m ./../../uspmv_cpu_ap_new_new_likwid $matrix crs -bucket_size 3.08e7 -mode b $datatype >> uspmv_cpu_ap_new_likwid_flops.txt
#                 srun -c 72 --cpu-freq=2100000-2100000:performance likwid-perfctr -C M0:0-$core_count -g TMA -m ./../../uspmv_cpu_ap_new_new_likwid $matrix crs -bucket_size 3.08e7 -mode b $datatype >> check_cache.txt

#             # done
#         # done
#     done
# done