#!/bin/bash
#SBATCH -J log
#SBATCH -C knl
#SBATCH -N 2
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -o log_please_work
#SBATCH -A m3194

root_dir=$HOME/plasma-profile-predictor
. $root_dir/scripts/setup.sh
srun -l python $root_dir/train_NERSC.py

