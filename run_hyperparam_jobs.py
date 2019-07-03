
from helpers.hyperparam_helpers import make_folder_contents
import numpy as np
import os

import yaml
input_conf='configs/lstm_cnn_merge.yaml'
input_script='scripts/lstm_cnn_merge.sh'


subfolder='different_padding'

#all_changes = [(x,y,z) for x in [1,2,4] for y in ['linear','relu'] for z in ['linear','relu']]


for pad_1d_to in [-100, -200]:
    for num_final_layers in [0,2]:
        for lookback in [3, 4]:
            delay = 6
#             lookback = 3

            n_components = None

            dense_pre_layers = 2
            dense_pre_size = 30
            rnn_size = 30

    #         num_final_layers = 2

            # if len(sigs)>0:
            #     sig_name=sigs[-1]
            # else:
            #     sig_name='None'

            new_dirname = 'ncomponents_{}_lookback_{}_prelayers_{}_size_{}_rnnsize_{}_delay_{}_numfinallayers_{}_pad1dto_{}'.format(n_components,
                                                                                                            lookback, 
                                                                                                            dense_pre_layers,
                                                                                                            dense_pre_size,
                                                                                                            rnn_size, delay, num_final_layers, pad_1d_to)


            changes_array = [['data','lookback', lookback],
                                            ['model','num_pre_layers',dense_pre_layers],
                                            ['model','dense_pre_size',dense_pre_size],
                                            ['model','rnn_size',rnn_size],
                                            ['data','n_components',n_components],

                                            ['data', 'delay', delay],
                                            ['data', 'pad_1d_to', pad_1d_to],

                                            ['model', 'num_final_layers', num_final_layers]]




            # changes_array = [['model','num_final_layers',num_final_layers],
            #                  ['model','dense_final_activation',dens_final_act],
            #                  ['model','dense_final_size',dens_final_size]]




            output_dir=os.path.join('/global/cscratch1/sd/al34/autoruns',subfolder,new_dirname)
            make_folder_contents(input_conf, input_script, output_dir, changes_array)
            os.system('sbatch {}'.format(os.path.join(output_dir,'driver.sh')))
