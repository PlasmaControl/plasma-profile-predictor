import os
import yaml
import re

def make_folder_contents(input_conf, input_script, output_dir, changes_array):
    # make the directory
    os.makedirs(output_dir, exist_ok=True)

    # make the new conf in the directory
    conf=os.path.join(output_dir, 'conf.yaml')
    with open(input_conf) as f:
        data = yaml.load(f)

    for arr in changes_array:
        data[arr[0]][arr[1]] = arr[2]

    data['output_dir']=output_dir

    with open(conf, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    # make the new driver
    driver=os.path.join(output_dir, 'driver.sh')
    with open(input_script, 'r') as f:
        script = f.read()
    with open(driver, 'w') as f:
        script=re.sub('config=.*','config='+conf, script)
        script=re.sub('#SBATCH -o.*', '#SBATCH -o {}'.format(os.path.join(output_dir,'log.out')), script)
        f.write(script)


#output_dir=os.path.join('/global/cscratch1/sd/abbatej/autoruns',subfolder,new_dirname)
#make_new_conf(input_conf, input_script, output_dir, changes_array)

#os.system('sbatch '+os.path.join(output_dir,'driver.sh'))
