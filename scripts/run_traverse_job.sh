#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00

root_dir=$HOME/plasma-profile-predictor

module load anaconda
conda activate tensorflow_gpu

python $root_dir/minimal_test.py
