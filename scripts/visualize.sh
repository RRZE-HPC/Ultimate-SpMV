#!/bin/bash -l

matrices=$(ls ./matrices/ | while read file; do realpath "matrices/$file"; done)

module load python

for matrix in $matrices;
do
    echo $matrix
    python ./scripts/mm2sparsityPattern.py $matrix 0
done