#!/bin/bash
#SBATCH -J lstm_cnn_merge
#SBATCH -C knl
#SBATCH -N 32
#SBATCH -q regular
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out
#SBATCH -A m3194

root_dir = $HOME/plasma_profiles_predictor/
. root_dir/scripts/setup.sh
config=root_dir/configs/lstm_cnn_merge_final_dense_activation.yaml
srun -l python train.py $config --distributed
