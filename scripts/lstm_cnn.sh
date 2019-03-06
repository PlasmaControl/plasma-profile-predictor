#!/bin/bash
#SBATCH -J lstm_cnn
#SBATCH -C knl
#SBATCH -N 4
#SBATCH -q regular
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out
#SBATCH -A m3194

. scripts/setup.sh
config=configs/lstm_cnn.yaml
srun -l python train.py $config --distributed --verbose
