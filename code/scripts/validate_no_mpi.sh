#!/bin/bash -l
#SBATCH -J validation_tests_no_mpi
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --output=/home/hpc/k107ce/k107ce17/linking_it_solve/Ultimate-SpMV/code/scripts/results/%j_%x.out

module load intel
module load intelmpi
module load mkl

export I_MPI_PIN=1
export I_MPI_PIN_PROCESSOR_LIST="allcores"

declare -a other_kernel_formats=("crs")
declare -a seg_types=("-seg_rows") # "-seg_nnz") # TODO: metis?

declare -a Cs=("1" "2" "3" "4" "8" "10" "16" "32" "64")
declare -a sigmas=("1" "2" "3" "4" "8" "10" "16" "32" "64")

# # Config 1. Check non-scs kernels are working, MPI off
# export I_MPI_PIN_DOMAIN="72:compact"
# export OMP_NUM_THREADS=72
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close
# for kernel_format in "${other_kernel_formats[@]}";
# do
#     for seg_method in "${seg_types[@]}";
#     do
#         for rand_opt in 0 1;
#         do
#             ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/impcol_e.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             ./../../uspmv_no_mpi ../../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#         done
#     done
# done

# # Config 3. Check scs kernel with sigma=1 fixed, MPI off
# export I_MPI_PIN_DOMAIN="72:compact"
# export OMP_NUM_THREADS=72
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close
# for C in "${Cs[@]}";
# do
#     for seg_method in "${seg_types[@]}";
#     do
#         for rand_opt in 0 1;
#         do
#             ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/impcol_e.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#         done
#     done
# done

# # Config 5. Check scs kernel with C=1 fixed, MPI off
# export I_MPI_PIN_DOMAIN="72:compact"
# export OMP_NUM_THREADS=72
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close
# for sigma in "${sigmas[@]}";
# do
#     for seg_method in "${seg_types[@]}";
#     do
#         for rand_opt in 0 1;
#         do
#             ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/impcol_e.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#         done
#     done
# done

# # Config 7. Check scs kernel with C>1, sigma>C, MPI off
# export I_MPI_PIN_DOMAIN="72:compact"
# export OMP_NUM_THREADS=72
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close
# for C in "${Cs[@]}";
# do
#     for sigma in "${sigmas[@]}";
#     do
#         for seg_method in "${seg_types[@]}";
#         do
#             for rand_opt in 0 1;
#             do
#                 if [ "$sigma" -gt "$C" ];
#                 then
#                     ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     ./../../uspmv_no_mpi ../../matrices/impcol_e.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                 fi
#             done
#         done
#     done
# done

# # Config 9. Check scs kernel with C>1, C>=sigma, MPI off
# export I_MPI_PIN_DOMAIN="72:compact"
# export OMP_NUM_THREADS=72
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close
# for C in "${Cs[@]}";
# do
#     for sigma in "${sigmas[@]}";
#     do
#         for seg_method in "${seg_types[@]}";
#         do
#             for rand_opt in 0 1;
#             do
#                 if [ "$sigma" -le "$C" ];
#                 then
#                     ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     ./../../uspmv_no_mpi ../../matrices/impcol_e.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                 fi
#             done
#         done
#     done
# done

# Config 11. Check scs kernel with C>1, sigma>1 MPI off
export I_MPI_PIN_DOMAIN="72:compact"
export OMP_NUM_THREADS=72
export OMP_PLACES=cores
export OMP_PROC_BIND=close
for C in "${Cs[@]}";
do
    for sigma in "${sigmas[@]}";
    do
        for seg_method in "${seg_types[@]}";
        do
            for rand_opt in 0 1;
            do
                export I_MPI_PIN_DOMAIN="36:compact"
                export OMP_NUM_THREADS=36
                export OMP_PLACES=cores
                export OMP_PROC_BIND=close
                ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                ./../../uspmv_no_mpi ../../matrices/impcol_e.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
                ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
            done
        done
    done
done

# Bench!
export I_MPI_PIN_DOMAIN="72:compact"
export OMP_NUM_THREADS=72
export OMP_PLACES=cores
export OMP_PROC_BIND=close
for C in "${Cs[@]}";
do
    for sigma in "${sigmas[@]}";
    do
        for seg_method in "${seg_types[@]}";
        do
            for rand_opt in 0 1;
            do
                export I_MPI_PIN_DOMAIN="36:compact"
                export OMP_NUM_THREADS=36
                export OMP_PLACES=cores
                export OMP_PROC_BIND=close
                ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma  -mode b $seg_method -rand_x $rand_opt -dp
                ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma  -mode b $seg_method -rand_x $rand_opt -dp
                ./../../uspmv_no_mpi ../../matrices/impcol_e.mtx scs -c $C -s $sigma  -mode b $seg_method -rand_x $rand_opt -dp
                ./../../uspmv_no_mpi ../../matrices/FDM-2d-16.mtx scs -c $C -s $sigma  -mode b $seg_method -rand_x $rand_opt -sp
                ./../../uspmv_no_mpi ../../matrices/matrix1.mtx scs -c $C -s $sigma  -mode b $seg_method -rand_x $rand_opt -sp
            done
        done
    done
done