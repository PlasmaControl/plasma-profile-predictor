import os
import datetime


def make_bash_scripts(number, output_dir, ncpu, ngpu, mem, times, mode="traverse"):
    """Creates slurm scripts for batch submit jobs.

    Parameters
    ----------
    number : int
        how many to create
    output_dir : str
        where to create the scripts
    ncpu : int
        how many CPU to request
    nhpu : int
        how many GPU to request
    mem : int
        how much memory to request (in GB)
    times : list
        list of estimated runtimes for jobs, in minutes
    """

    # make the directory
    os.makedirs(output_dir, exist_ok=True)

    for i in range(number):
        command = [
            "root_dir=$HOME/plasma-profile-predictor",
            "module load anaconda",
            "conda activate tf2-gpu",
        ]
        if mode == "traverse":
            command.append("python $root_dir/train_traverse.py " + str(i))
            job = "#SBATCH -J CONV " + str(i)
        elif mode == "autoencoder":
            command.append("python $root_dir/train_autoencoder.py " + str(i))
            job = "#SBATCH -J LRAN_" + str(i)

        slurm_script(
            os.path.join(output_dir, "driver" + str(i) + ".sh"),
            command=command,
            job_name=job,
            ncpu=ncpu,
            ngpu=ngpu,
            mem=mem,
            time=times[i],
        )


def slurm_script(
    file_path,
    command,
    job_name=None,
    ncpu=1,
    ngpu=0,
    mem=32,
    time=60,
    user="",
):
    """Create a SLURM scripts

    Parameters
    ----------
    file_path : path-like
        where to create file
    command : str or list of str
        command to run within slurm script
    ncpu : int
        number of CPU cores
    cgpu : int
        number of GPUs
    mem : int
        amount of memory, in GB
    time : int
        time, in minutes
    user : str
        username for mailto, assumes princeton netID. Defaults to $USER
    """
    file_path = str(file_path)

    with open(file_path, "w+") as f:
        f.write("#!/bin/bash \n")
        f.write(
            "#SBATCH -J "
            + str(job_name if job_name is not None else file_path.split("/")[-1])
            + " \n"
        )
        if user == "":
            user = os.environ["USER"]
        if user is not None:
            f.write("#SBATCH --mail-type=begin \n")
            f.write("#SBATCH --mail-type=end \n")
            f.write("#SBATCH --mail-user={}@princeton.edu \n".format(user))
        f.write("#SBATCH -N 1 \n")
        f.write("#SBATCH -c " + str(ncpu) + "\n")
        f.write("#SBATCH --mem " + str(mem) + "G\n")
        if ngpu > 0:
            f.write("#SBATCH -G " + str(ngpu) + "\n")
        f.write("#SBATCH -o " + file_path, +".log  \n")
        f.write("#SBATCH -t " + str(datetime.timedelta(minutes=time)) + "\n")
        if isinstance(command, (list, tuple)):
            for c in command:
                f.write(c + " \n")
        else:
            f.write(command + "\n")
        f.write("exit")
