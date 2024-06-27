#!/bin/bash -l
#SBATCH -J node_perf_tests
#SBATCH -N 1
#--cpus-per-task=18
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH -c 72
#SBATCH --constraint=hwperf
#SBATCH --cpu-freq=2000000-2000000:performance 
#SBATCH --output=/home/hpc/k107ce/k107ce17/Ultimate-SpMV/code/scripts/results/%j_%x.out

unset SLURM_EXPORT_ENV 

# new modules!, gather-less intel compiler
module load intel
module load likwid
module load mkl

# 72 physical cores/node, 2 sockets/node, 2 ccNuma domain/socket => 72/4 = 18 cores/process if pinning to ccNuma
# export I_MPI_PIN=1
# export I_MPI_PIN_PROCESSOR_LIST="allcores"
# export I_MPI_PIN_DOMAIN="18:compact"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

declare -a matrices=(
    "/home/vault/k107ce/k107ce17/bench_matrices/Long_Coup_dt0.mtx"
    # "/home/vault/k107ce/k107ce17/bench_matrices/ML_Geer.mtx"
    # "/home/vault/k107ce/k107ce17/bench_matrices/af_shell10.mtx"
)

# declare -a datatypes=("-dp" "mp")

# declare -a dat atypes=("-dp")

# Validate Accuracy
# for matrix in "${matrices[@]}";
# do
#     export OMP_NUM_THREADS=18
#     srun --cpu-freq=2000000-2000000 ./../../uspmv_cpu $matrix crs -mode s $datatype
# done

# Bench: Scale only in one numa domain
# for matrix in "${matrices[@]}";
# do
#     # for datatype in "${datatypes[@]}";
#     # do
#         for core_count in {18..1};
#         do
#             export OMP_NUM_THREADS=$core_count
#             for med_loop in {1..5};
#             do
#                 srun -c 72 --cpu-freq=2000000-2000000:performance ./../../uspmv_dp $matrix crs -mode b -dp
#                 # srun -c 72 --cpu-freq=2000000-2000000:performance ./../../uspmv_sp $matrix crs -mode b -sp
#                 # srun -c 72 --cpu-freq=2000000-2000000:performance ./../../uspmv_with_lp_x $matrix crs -bucket_size 30846172.3366031 -mode b -mp
#                 # srun -c 72 --cpu-freq=2000000-2000000:performance ./../../uspmv_with_hp_x $matrix crs -bucket_size 30846172.3366031 -mode b -mp
#             done
#         done
#     # done
# done

for matrix in "${matrices[@]}";
do
    core_count=17
    # for core_count in {0..17};
    # do
        #### Not necessary with likwid ####
        # export OMP_NUM_THREADS=$core_count+1
        ###################################

        # for med_loop in {1..3};
        # do
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Long_Coup_dt0.mtx crs -mode b -dp >> data_movement_tests/dp_data_movement_LC.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Long_Coup_dt0.mtx crs -mode b -sp >> data_movement_tests/sp_data_movement_LC.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Long_Coup_dt0.mtx crs -mode b -mp -bucket_size 30846172.3366031 >> data_movement_tests/ap_data_movement_LC.txt

            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Flan_1565.mtx crs -mode b -dp >> data_movement_tests/dp_data_movement_Flan.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Flan_1565.mtx crs -mode b -sp >> data_movement_tests/sp_data_movement_Flan.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Flan_1565.mtx crs -mode b -mp -bucket_size 0.131937082292698 >> data_movement_tests/ap_data_movement_Flan.txt

            srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Emilia_923.mtx crs -mode b -dp >> data_movement_tests/dp_data_movement_Emilia.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Emilia_923.mtx crs -mode b -sp >> data_movement_tests/sp_data_movement_Emilia.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_likwid /home/vault/k107ce/k107ce17/bench_matrices/Emilia_923.mtx crs -mode b -mp -bucket_size 3886214.46894891 >> data_movement_tests/ap_data_movement_Emilia.txt
        # done
    # done
done