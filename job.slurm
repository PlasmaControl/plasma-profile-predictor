#!/bin/bash 
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -o /home/wconlin/run_results_11_2/log0.out 
#SBATCH -t 3:00:00
root_dir=$HOME/plasma-profile-predictor 
module load anaconda 
conda activate tfgpu 
python $root_dir/train_traverse.py
exit
