#!/bin/bash -l
#SBATCH -J node_perf_tests
#SBATCH -N 1
#--cpus-per-task=18
#SBATCH -t 02:00:00
#SBATCH --exclusive
#SBATCH -c 72
#SBATCH --constraint=hwperf
#SBATCH --cpu-freq=2000000-2000000:performance 
#SBATCH --output=/home/hpc/k107ce/k107ce17/Ultimate-SpMV/code/scripts/results/%j_%x.out

unset SLURM_EXPORT_ENV 

# new modules!, gather-less intel compiler
# module use -a ~unrz139/.modules/modulefiles
# module load intel
# module load likwid/spr
# module load mkl
module load likwid mkl intel

# 72 physical cores/node, 2 sockets/node, 2 ccNuma domain/socket => 72/4 = 18 cores/process if pinning to ccNuma
# export I_MPI_PIN=1
# export I_MPI_PIN_PROCESSOR_LIST="allcores"
# export I_MPI_PIN_DOMAIN="18:compact"

# export OMP_PLACES=cores
# export OMP_PROC_BIND=close

declare -a matrices=(
    # "/home/vault/k107ce/k107ce17/bench_matrices/dlr1.mtx"
    # "/home/vault/k107ce/k107ce17/bench_matrices/Long_Coup_dt0.mtx"
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

# for matrix in "${matrices[@]}";
# do
    # core_count=0
    for core_count in {17..0};
    do
        #### Not necessary with likwid ####
        # export OMP_NUM_THREADS=$core_count+1
        ###################################

        for med_loop in {1..3};
        do
            # Compact pinning
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_icx_fixed_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/compact/fixed_yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_icx_fixed_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/compact/fixed_yequals_CRS_AP_LC_data_vol_test.txt
            srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_icx_fixed_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/compact/fixed_yequals_CRS_SP_LC_data_vol_test.txt

            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/compact/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/compact/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C M0:0-$core_count -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/compact/yplusequals_CRS_SP_LC_data_vol_test.txt

            # # scatter pinning
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,2,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,3,2,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,4,1,3,2 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,5,1,4,2,3 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,6,1,5,2,3,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,7,1,6,2,5,3,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,8,1,7,2,6,3,5,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,9,1,8,2,7,3,6,4,5 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,10,1,9,2,8,3,7,4,6,5 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,11,1,10,2,9,3,8,4,7,5,6 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,12,1,11,2,10,3,9,4,8,5,7,6 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,13,1,12,2,11,3,10,4,9,5,8,6,7 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,14,1,13,2,12,3,11,4,10,5,9,6,8,7 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,15,1,14,2,13,3,12,4,11,5,10,6,9,7,8 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,17,1,16,2,15,3,14,4,13,5,12,6,11,7,10,8,9 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_DP_LC_data_vol_test.txt

            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,2,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,3,2,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,4,1,3,2 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,5,1,4,2,3 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,6,1,5,2,3,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,7,1,6,2,5,3,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,8,1,7,2,6,3,5,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,9,1,8,2,7,3,6,4,5 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,10,1,9,2,8,3,7,4,6,5 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,11,1,10,2,9,3,8,4,7,5,6 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,12,1,11,2,10,3,9,4,8,5,7,6 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,13,1,12,2,11,3,10,4,9,5,8,6,7 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,14,1,13,2,12,3,11,4,10,5,9,6,8,7 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,15,1,14,2,13,3,12,4,11,5,10,6,9,7,8 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,17,1,16,2,15,3,14,4,13,5,12,6,11,7,10,8,9 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_AP_LC_data_vol_test.txt

            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,2,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,3,2,1 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,4,1,3,2 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,5,1,4,2,3 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,6,1,5,2,3,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,7,1,6,2,5,3,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,8,1,7,2,6,3,5,4 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,9,1,8,2,7,3,6,4,5 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,10,1,9,2,8,3,7,4,6,5 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,11,1,10,2,9,3,8,4,7,5,6 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,12,1,11,2,10,3,9,4,8,5,7,6 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,13,1,12,2,11,3,10,4,9,5,8,6,7 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,14,1,13,2,12,3,11,4,10,5,9,6,8,7 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,15,1,14,2,13,3,12,4,11,5,10,6,9,7,8 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,17,1,16,2,15,3,14,4,13,5,12,6,11,7,10,8,9 -g MEM -m ./../../uspmv_icx_y_equals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yequals_CRS_SP_LC_data_vol_test.txt

            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,2,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,3,2,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,4,1,3,2 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,5,1,4,2,3 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,6,1,5,2,3,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,7,1,6,2,5,3,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,8,1,7,2,6,3,5,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,9,1,8,2,7,3,6,4,5 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,10,1,9,2,8,3,7,4,6,5 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,11,1,10,2,9,3,8,4,7,5,6 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,12,1,11,2,10,3,9,4,8,5,7,6 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,13,1,12,2,11,3,10,4,9,5,8,6,7 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,14,1,13,2,12,3,11,4,10,5,9,6,8,7 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,15,1,14,2,13,3,12,4,11,5,10,6,9,7,8 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,17,1,16,2,15,3,14,4,13,5,12,6,11,7,10,8,9 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -dp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_DP_LC_data_vol_test.txt

            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,2,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,3,2,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,4,1,3,2 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,5,1,4,2,3 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,6,1,5,2,3,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,7,1,6,2,5,3,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,8,1,7,2,6,3,5,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,9,1,8,2,7,3,6,4,5 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,10,1,9,2,8,3,7,4,6,5 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,11,1,10,2,9,3,8,4,7,5,6 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,12,1,11,2,10,3,9,4,8,5,7,6 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,13,1,12,2,11,3,10,4,9,5,8,6,7 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,14,1,13,2,12,3,11,4,10,5,9,6,8,7 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,15,1,14,2,13,3,12,4,11,5,10,6,9,7,8 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,17,1,16,2,15,3,14,4,13,5,12,6,11,7,10,8,9 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -mp -bucket_size 30846172.3366031 >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_AP_LC_data_vol_test.txt

            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,2,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,3,2,1 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,4,1,3,2 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,5,1,4,2,3 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,6,1,5,2,3,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,7,1,6,2,5,3,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,8,1,7,2,6,3,5,4 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,9,1,8,2,7,3,6,4,5 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,10,1,9,2,8,3,7,4,6,5 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,11,1,10,2,9,3,8,4,7,5,6 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,12,1,11,2,10,3,9,4,8,5,7,6 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,13,1,12,2,11,3,10,4,9,5,8,6,7 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,14,1,13,2,12,3,11,4,10,5,9,6,8,7 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,15,1,14,2,13,3,12,4,11,5,10,6,9,7,8 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt
            # srun -c 72 --cpu-freq=2000000-2000000:performance likwid-perfctr -C 0,17,1,16,2,15,3,14,4,13,5,12,6,11,7,10,8,9 -g MEM -m ./../../uspmv_icx_y_plusequals /home/hpc/k107ce/k107ce17/Ultimate-SpMV/Long_Coup_dt0.mtx crs -sp >> ../../data_movement_tests/georg_bump_tests/scatter/yplusequals_CRS_SP_LC_data_vol_test.txt

        done
    done
# done