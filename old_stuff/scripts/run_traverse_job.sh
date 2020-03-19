#!/bin/bash
#SBATCH -N 1
#SBARCH -c 16
#SBATCH -t 02:00:00

root_dir=$HOME/plasma-profile-predictor

module load anaconda
conda activate tensorflow_cpu

python $root_dir/train_traverse.py
