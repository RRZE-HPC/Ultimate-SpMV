#!/bin/bash -l
#SBATCH -J validation_tests
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH --exclusive
#SBATCH --output=/home/hpc/ihpc/ihpc062h/HPC_HiWi/fresh_USpMV/Ultimate-SpMV/code/scripts/results/%j_%x.out

module load intel
module load intelmpi
module load mkl

export I_MPI_PIN=1
export I_MPI_PIN_PROCESSOR_LIST="allcores"

declare -a other_kernel_formats=("crs")
declare -a seg_types=("-seg_rows") # "-seg_nnz") # TODO: metis?

declare -a Cs=("1" "2" "3" "4" "8" "16" "32" "64" "128")
declare -a sigmas=("1" "2" "3" "4" "8" "16" "32" "64" "128")

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
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#         done
#     done
# done

# # Config 2. Check non-scs kernels are working, MPI on
# for kernel_format in "${other_kernel_formats[@]}";
# do
#     for seg_method in "${seg_types[@]}";
#     do
#         for rand_opt in 0 1;
#         do
#             export I_MPI_PIN_DOMAIN="36:compact"
#             export OMP_NUM_THREADS=36
#             export OMP_PLACES=cores
#             export OMP_PROC_BIND=close
#             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

#             export I_MPI_PIN_DOMAIN="18:compact"
#             export OMP_NUM_THREADS=18
#             export OMP_PLACES=cores
#             export OMP_PROC_BIND=close
#             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

#             # Doesn't exactly disable OMP, bust still does what I want
#             export OMP_NUM_THREADS=1
#             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx $kernel_format -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
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
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#         done
#     done
# done

# # Config 4. Check scs kernel with sigma=1 fixed, MPI on
# for C in "${Cs[@]}";
# do
#     for seg_method in "${seg_types[@]}";
#     do
#         for rand_opt in 0 1;
#         do
#             export I_MPI_PIN_DOMAIN="36:compact"
#             export OMP_NUM_THREADS=36
#             export OMP_PLACES=cores
#             export OMP_PROC_BIND=close
#             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

#             export I_MPI_PIN_DOMAIN="18:compact"
#             export OMP_NUM_THREADS=18
#             export OMP_PLACES=cores
#             export OMP_PROC_BIND=close
#             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

#             export OMP_NUM_THREADS=1
#             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s 1 -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

#         done
#     done
# done

# Config 5. Check scs kernel with C=1 fixed, MPI off
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
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#             mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#             mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#         done
#     done
# done

# # # Config 6. Check scs kernel with C=1 fixed, MPI on
# # for sigma in "${sigmas[@]}";
# # do
# #     for seg_method in "${seg_types[@]}";
# #     do
# #         for rand_opt in 0 1;
# #         do
# #             export I_MPI_PIN_DOMAIN="36:compact"
# #             export OMP_NUM_THREADS=36
# #             export OMP_PLACES=cores
# #             export OMP_PROC_BIND=close
# #             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
# #             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
# #             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
# #             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
# #             mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
# #             mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

# #             export I_MPI_PIN_DOMAIN="18:compact"
# #             export OMP_NUM_THREADS=18
# #             export OMP_PLACES=cores
# #             export OMP_PROC_BIND=close
# #             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
# #             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
# #             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
# #             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
# #             mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
# #             mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
        
# #             export OMP_NUM_THREADS=1
# #             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
# #             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
# #             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
# #             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
# #             mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
# #             mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c 1 -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

# #         done
# #     done
# # done

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
#                     mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                 fi
#             done
#         done
#     done
# done

# # Config 8. Check scs kernel with C>1, sigma>C, MPI on
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
#                     export I_MPI_PIN_DOMAIN="36:compact"
#                     export OMP_NUM_THREADS=36
#                     export OMP_PLACES=cores
#                     export OMP_PROC_BIND=close
#                     mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

#                     export I_MPI_PIN_DOMAIN="18:compact"
#                     export OMP_NUM_THREADS=18
#                     export OMP_PLACES=cores
#                     export OMP_PROC_BIND=close
#                     mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
                
#                     export OMP_NUM_THREADS=1
#                     mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                 fi
#             done
#         done
#     done
# done

# Config 9. Check scs kernel with C>1, C>=sigma, MPI off
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
#                     mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 1 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 1 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                 fi
#             done
#         done
#     done
# done

# # Config 10. Check scs kernel with C>1, C>=sigma, MPI on
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
#                     export I_MPI_PIN_DOMAIN="36:compact"
#                     export OMP_NUM_THREADS=36
#                     export OMP_PLACES=cores
#                     export OMP_PROC_BIND=close
#                     mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

#                     export I_MPI_PIN_DOMAIN="18:compact"
#                     export OMP_NUM_THREADS=18
#                     export OMP_PLACES=cores
#                     export OMP_PROC_BIND=close
#                     mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
                
#                     export OMP_NUM_THREADS=1
#                     mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
#                     # mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
#                     # mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                     # mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
#                 fi
#             done
#         done
#     done
# done

# Config 11. Check scs kernel with C>1, sigma>1 MPI on
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
                mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                # mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
                # mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
                # mpirun -n 2 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
                # mpirun -n 2 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp

                export I_MPI_PIN_DOMAIN="18:compact"
                export OMP_NUM_THREADS=18
                export OMP_PLACES=cores
                export OMP_PROC_BIND=close
                mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                # mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
                # mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
                # mpirun -n 4 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
                # mpirun -n 4 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
            
                export OMP_NUM_THREADS=1
                mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -dp
                # mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
                # mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -sp
                # mpirun -n 72 ./../uspmv ../matrices/FDM-2d-16.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
                # mpirun -n 48 ./../uspmv ../matrices/matrix1.mtx scs -c $C -s $sigma -mode s $seg_method -rand_x $rand_opt -rev 3 -mp
            done
        done
    done
done