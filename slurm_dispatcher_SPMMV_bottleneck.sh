#!/bin/bash -l

exefile=$(realpath $1)
matrixfile=$(realpath $2)
maxcores=${3:-72}
stepcores=${4:-12}
maxblockvec=${5:-30}

newfoldername=bench_SpMMV_$(date "+%Y%m%d_%H%M%S")

mkdir $newfoldername

cp $exefile $newfoldername

startcores=$maxcores #72
endcores=$((maxcores - stepcores)) #72-5 = 67
ID=$(sbatch SPMMV_bottleneck.sh $newfoldername $matrixfile $maxblockvec $startcores  $endcores | grep "^Submitted" | cut -f 4 -d ' ')
echo "$newfoldername | $matrixfile | $maxblockvec | $startcores | $endcores | $ID" >$newfoldername/slurmids.txt

for loopcorestart in $(seq $endcores $((-stepcores - 1)) 2); do
    lscore=$((loopcorestart - 1))
    lecore=$((lscore - stepcores ))
    ID=$(sbatch -d afterok:$ID SPMMV_bottleneck.sh $newfoldername $matrixfile $maxblockvec $lscore $lecore | grep "^Submitted" | cut -f 4 -d ' ')
    echo "$newfoldername | $matrixfile | $maxblockvec | $lscore | $lecore | $ID" >>$newfoldername/slurmids.txt
done

echo "list of schduled jobs : "
echo "newfoldername | matrixfile | maxblockvec | startcores | endcores | ID" 
cat $newfoldername/slurmids.txt
