import os
import yaml
import re


def make_bash_scripts(number, output_dir):
       # make the directory
    os.makedirs(output_dir, exist_ok=True)

    for i in range(number):
        with open(os.path.join(output_dir, 'driver' + str(i) + '.sh'), 'w+') as f:
            f.write('#!/bin/bash')
            f.write('#SBATCH -N 1')
            f.write('#SBATCH -c 16')
            f.write('#SBATCH -G 1')
            f.write('#SBATCH -o ' +
                    os.path.join(output_dir, 'log' + str(i) + '.out'))
            f.write('#SBATCH -t 04:00:00')

            f.write('root_dir=$HOME/plasma-profile-predictor')
            f.write('module load anaconda')
            f.write('conda activate tfgpu')

            f.write('python $root_dir/train_traverse.py ' + str(i))


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
