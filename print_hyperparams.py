import yaml
import os
import re
import numpy as np


#base_dir='/global/cscratch1/sd/al34/autoruns/different_padding'
base_dir='/global/cscratch1/sd/abbatej/autoruns/trend_plus_actuators_1/'

for dir_name in ['4_old']:#os.listdir(base_dir):
    def load_config(base_file):
        with open(base_file) as f:
            config = yaml.load(f)
        return config

    config=load_config(os.path.join(base_dir,dir_name,'conf.yaml'))
    print(config)
    history = np.load(os.path.join(base_dir,dir_name,'history.npz'))
    val_mae = min(history['val_mean_absolute_error'])

    with open(os.path.join(base_dir,dir_name,'log.out')) as f:
        now = False
        for line in f:
            if now:
                m = re.search('0\..*',line)
                baseline_mae = m.group(0)
                break
            if re.search('(?<=0: baseline mae average:)',line):
                now = True

    print('{}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}'.format(base_dir,
                                                                          dir_name, 
                                                                          config['data_and_model']['sigs_0d'], 
                                                                          config['data_and_model']['sigs_1d'],
                                                                          config['data_and_model']['lookback'],
                                                                          config['data_and_model']['delay'],
                                                                          config['model']['rnn_type'],
                                                                          config['model']['rnn_size'],
                                                                          val_mae,
                                                                          baseline_mae))

    #except:
    #    pass
