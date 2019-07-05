#!/bin/bash
#SBATCH -J lstm_cnn_merge
#SBATCH -C knl
#SBATCH -N 4
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -o log_no_preprocess.out
#SBATCH -A m3194

root_dir=$HOME/plasma-profile-predictor
. $root_dir/scripts/setup.sh
config=$root_dir/configs/lstm_cnn_merge_joe.yaml
srun -l python $root_dir/train.py $config --distributed

