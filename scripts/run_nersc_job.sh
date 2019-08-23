#!/bin/bash
#SBATCH -J log_1_10000
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -o log_1_10000
#SBATCH -A m3194

root_dir=$HOME/plasma-profile-predictor
. $root_dir/scripts/setup.sh
python $root_dir/train_NERSC_1.py

