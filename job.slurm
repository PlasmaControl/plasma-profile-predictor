#!/bin/bash 
#SBATCH -N 1 
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -G 1
#SBATCH -o /home/wconlin/run_results_12_01/log0.out 
#SBATCH -t 0:20:00
root_dir=$HOME/plasma-profile-predictor 
module load anaconda 
conda activate tfgpu 
cd $root_dir
python train_autoencoder.py
exit
