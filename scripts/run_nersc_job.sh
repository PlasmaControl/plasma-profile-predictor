#!/bin/bash
#SBATCH -J lstm_cnn_merge
#SBATCH -C knl
#SBATCH -N 2
#SBATCH -q regular
#SBATCH -t 0:20:00
#SBATCH -o log
#SBATCH -A m3194

root_dir=$HOME/plasma-profile-predictor
. $root_dir/scripts/setup.sh
srun -l python $root_dir/train_NERSC_parallel_hvd.py

