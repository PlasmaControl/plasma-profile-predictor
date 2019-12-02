import os
import yaml
import re
import datetime

def make_bash_scripts(number, output_dir, ncpu, ngpu, req_mem, times, mode='traverse'):
       # make the directory
    os.makedirs(output_dir, exist_ok=True)

    for i in range(number):
        with open(os.path.join(output_dir, 'driver' + str(i) + '.sh'), 'w+') as f:
            f.write('#!/bin/bash \n')
            f.write('#SBATCH -N 1 \n')
            f.write('#SBATCH -c ' + str(ncpu) + '\n')
            f.write('#SBATCH --mem ' + str(req_mem) + 'G\n')
            if ngpu>0:
                f.write('#SBATCH -G ' + str(ngpu) + '\n')
            f.write('#SBATCH -o ' +
                    os.path.join(output_dir, 'log' + str(i) + '.out \n'))
            f.write('#SBATCH -t ' + str(datetime.timedelta(minutes=times[i])) + '\n')

            f.write('root_dir=$HOME/plasma-profile-predictor \n')
            f.write('module load anaconda \n')
            f.write('conda activate tfgpu \n')
            if mode=='traverse':
                f.write('python $root_dir/train_traverse.py ' + str(i) + '\n')
            elif mode=='autoencoder':
                f.write('python $root_dir/train_autoencoder.py ' + str(i) + '\n')
            f.write('exit')


def make_folder_contents(input_conf, input_script, output_dir, changes_array):
    # make the directory
    os.makedirs(output_dir, exist_ok=True)

    # make the new conf in the directory
    conf = os.path.join(output_dir, 'conf.yaml')
    with open(input_conf) as f:
        data = yaml.load(f)

    for arr in changes_array:
        data[arr[0]][arr[1]] = arr[2]

    data['output_dir'] = output_dir

    with open(conf, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    # make the new driver
    driver = os.path.join(output_dir, 'driver.sh')
    with open(input_script, 'r') as f:
        script = f.read()

    with open(driver, 'w') as f:
        script = re.sub('config=.*', 'config='+conf, script)
        script = re.sub(
            '#SBATCH -o.*', '#SBATCH -o {}'.format(os.path.join(output_dir, 'log.out')), script)
        f.write(script)
