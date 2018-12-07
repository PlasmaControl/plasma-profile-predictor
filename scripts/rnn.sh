#!/bin/bash
#SBATCH -J rnn
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/rnn.yaml
srun python train.py $config -d False
