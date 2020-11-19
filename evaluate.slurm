#!/bin/bash 
#SBATCH -N 1 
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -G 1
#SBATCH -o /home/jabbate/plasma-profile-predictor/evaluate_time_log_test.out 
#SBATCH -t 96:00:00
root_dir=$HOME/plasma-profile-predictor 
module load anaconda 
conda activate tfgpu 
cd $root_dir
python evaluate.py
exit
