#!/usr/bin/env bash

#n_features=$1
in_dir=$1
out_dir=$2
seed=$3
task=$4 

PYTHONPATH=$PYTHONPATH:. python3 tool/main.py -i ${in_dir} -o ${out_dir} -n -d ${task} -D -I -1 -rs ${seed}
