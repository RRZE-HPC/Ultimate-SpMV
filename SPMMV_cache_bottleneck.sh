#!/bin/bash -l
#
#SBATCH -J SPMMV_cache_crsbneck
#SBATCH -p singlenode
#SBATCH --constraint=hwperf
#SBATCH --time=1-00:00:00
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
        likwid-mpirun -mpi intelmpi -n $np -g L3 -m \
            ./$exe $matfile \
            crs \
            -mode b \
            -block_vec_size $bvecsize >temp.txt
        cat temp.txt >>L3_likwid.txt
        L3bandwidth=$(cat temp.txt | grep -i "L3 bandwidth \[MBytes/s\] STAT" | awk -F '|' '{print $3}')
        likwid-mpirun -mpi intelmpi -n $np -g L2 -m \
            ./$exe $matfile \
            crs \
            -mode b \
            -block_vec_size $bvecsize >temp.txt
        cat temp.txt >>L2_likwid.txt
        L2loadbandwidth=$(cat temp.txt | grep -i "L2D load bandwidth \[MBytes/s\] STAT" | awk -F '|' '{print $3}')
        likwid-mpirun -mpi intelmpi -n $np -g MEM_DP -m \
            ./$exe $matfile \
            crs \
            -mode b \
            -block_vec_size $bvecsize >temp.txt
        cat temp.txt >>MEM_likwid.txt
        likwidoutflopps=$(cat temp.txt | grep -i "  DP \[MFLOP/s\] STAT" | awk -F '|' '{print $3}')
        likwidoutbandwidth=$(cat temp.txt | grep -i "Memory bandwidth \[MBytes/s\] STAT" | awk -F '|' '{print $3}')
        echo "$np, $bvecsize, $likwidoutflopps, $likwidoutbandwidth, $L2loadbandwidth, $L3bandwidth" >>spmv_cache_data.txt
        echo "number of process $np and blockvecsize $bvecsize | elapsed $SECONDS sec "
    done
done
