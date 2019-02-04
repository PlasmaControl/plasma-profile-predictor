#!/bin/bash
#SBATCH -J rnn
#SBATCH -C knl
#SBATCH -N 2
#SBATCH -q regular
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/rnn.yaml
srun -l python train.py $config --distributed
