#!/bin/bash -l
#
#SBATCH -J SPMMV_bott_crs
#SBATCH -p singlenode
#SBATCH --time=20:00:00
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load intel intelmpi likwid

folname=$(realpath $1)
matfile=$(realpath $2)
maxblockvec=${3:-30}
startcores=${4:-72}
endcores=${5:-59}

[ $startcores -le 1 ] && startcores=1
[ $endcores -le 1 ] && endcores=1

cd $folname

exe=$(ls | grep "uspmv")

for np in $(seq $startcores -1 $endcores); do
    for bvecsize in $(seq $maxblockvec -1 1); do
        echo "number of process $np and blockvecsize $bvecsize | start @ $(date +"%T")"
        SECONDS=0
        likwid-mpirun -mpi intelmpi -n $np \
            ./$exe \
            $matfile \
            crs \
            -mode b \
            -block_vec_size $bvecsize \
            -verbose 1
        echo "number of process $np and blockvecsize $bvecsize | elapsed $SECONDS sec "
    done
done
