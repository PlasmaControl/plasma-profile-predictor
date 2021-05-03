#!/bin/bash 
#SBATCH -N 1 
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -G 1
#SBATCH -o /home/aiqtidar/run_results_04_12/log0.out 
#SBATCH -t 6:00:00
root_dir=$HOME 
module load anaconda 
conda activate tfgpu 
python $root_dir/plasma-profile-predictor/Controls/untitled.py
exit