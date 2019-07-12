from helpers.hyperparam_helpers import make_folder_contents
import numpy as np
import os

import yaml
input_conf='configs/trend_plus_actuators.yaml'
input_script='scripts/trend_plus_actuators.sh'

subfolder='trend_plus_actuators_min_normalized'

lookback_=np.arange(5,10)
delay_=np.arange(2,7)
rnn_size_=np.arange(10,80)
dense_final_size_=np.arange(10,80)
num_final_layers_=np.arange(1,4)
num_rnn_layers_=np.arange(1,4)

def choose_one(arr):
    return int(np.random.choice(arr))

for i in range(20):
    new_dirname = str(i)
    changes_array = [['data_and_model','lookback', choose_one(lookback_)],
                     ['data_and_model','delay',choose_one(delay_)],
                     ['model', 'rnn_size', choose_one(rnn_size_)],
                     ['model', 'dense_final_size', choose_one(dense_final_size_)],
                     ['model', 'num_final_layers', choose_one(num_final_layers_)],
                     ['model', 'num_rnn_layers', choose_one(num_rnn_layers_)]]

    output_dir=os.path.join('/global/cscratch1/sd/abbatej/autoruns',subfolder,new_dirname)
    make_folder_contents(input_conf, input_script, output_dir, changes_array)
    os.system('sbatch {}'.format(os.path.join(output_dir,'driver.sh')))
