#!/bin/bash 
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --mem 48G
#SBATCH -G 0
#SBATCH -o /home/wconlin/plasma-profile-predictor/log.out 
#SBATCH -t 24:00:00
root_dir=$HOME/plasma-profile-predictor 
module load anaconda 
conda activate tfgpu 
cd $root_dir
python ensemble_evaluate.py
exit
