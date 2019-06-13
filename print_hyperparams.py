import yaml
import os
import re
import numpy as np

base_dir='/global/cscratch1/sd/al34/autoruns/lookback_with_0d_lookahead'
for dir_name in os.listdir(base_dir):
    try:
        def load_config(base_file):
            with open(base_file) as f:
                config = yaml.load(f)
            return config

        config=load_config(os.path.join(base_dir,dir_name,'conf.yaml'))

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
                                                                              config['data']['sigs_0d'], 
                                                                              config['data']['sigs_1d'],
                                                                              config['data']['n_components'],
                                                                              config['data']['avg_window'],
                                                                              config['data']['lookback'],
                                                                              config['data']['delay'],
                                                                              config['model']['dense_pre_size'],
                                                                              config['model']['num_pre_layers'],
                                                                              config['model']['rnn_type'],
                                                                              config['model']['rnn_size'],
                                                                              val_mae,
                                                                              baseline_mae))
        
    except:
        pass
