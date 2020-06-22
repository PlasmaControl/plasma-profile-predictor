#!/bin/bash 

#SBATCH -N 1 
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -G 1
#SBATCH -o /home/wconlin/plasma-profile-predictor/log.out 
#SBATCH -t 48:00:00

root_dir=$HOME/plasma-profile-predictor 
module load anaconda 
conda activate tfgpu 
cd $root_dir
python train_traverse.py

exit
