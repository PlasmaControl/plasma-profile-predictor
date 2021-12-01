import os
import re
import datetime


def make_bash_scripts(number, output_dir, ncpu, ngpu, req_mem, times, mode="traverse"):
    """Creates slurm scripts for batch submit jobs.

    Args:
        number (int): how many to create
        output_dir (str): where to create the scripts
        ncpu (int): how many CPU to request
        nhpu (int): how many GPU to request
        req_mem (int): how much memory to request (in GB)
        times (list): list of estimated runtimes for jobs, in minutes)
    """

    # make the directory
    os.makedirs(output_dir, exist_ok=True)

    for i in range(number):
        with open(os.path.join(output_dir, "driver" + str(i) + ".sh"), "w+") as f:
            f.write("#!/bin/bash \n")
            if mode == "autoencoder":
                f.write("#SBATCH -J LRAN " + str(i) + " \n")
            else:
                f.write("#SBATCH -J CONV " + str(i) + " \n")
            f.write("#SBATCH --mail-type=begin \n")
            f.write("#SBATCH --mail-type=end \n")
            f.write("#SBATCH --mail-user=wconlin@princeton.edu \n")
            f.write("#SBATCH -N 1 \n")
            f.write("#SBATCH -c " + str(ncpu) + "\n")
            f.write("#SBATCH --mem " + str(req_mem) + "G\n")
            if ngpu > 0:
                f.write("#SBATCH -G " + str(ngpu) + "\n")
            f.write(
                "#SBATCH -o " + os.path.join(output_dir, "log" + str(i) + ".out \n")
            )
            f.write("#SBATCH -t " + str(datetime.timedelta(minutes=times[i])) + "\n")

            f.write("root_dir=$HOME/plasma-profile-predictor \n")
            f.write("module load anaconda \n")
            f.write("conda activate tf2-gpu \n")
            if mode == "traverse":
                f.write("python $root_dir/train_traverse.py " + str(i) + "\n")
            elif mode == "autoencoder":
                f.write("python $root_dir/train_autoencoder.py " + str(i) + "\n")
            f.write("exit")
