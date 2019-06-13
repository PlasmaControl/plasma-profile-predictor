
from helpers.hyperparam_helpers import make_folder_contents
import numpy as np
import os

import yaml
print("run_hyperparams")
input_conf='configs/lstm_cnn_merge.yaml'
input_script='scripts/lstm_cnn_merge.sh'


subfolder='lookback_with_0d_lookahead'

#all_changes = [(x,y,z) for x in [1,2,4] for y in ['linear','relu'] for z in ['linear','relu']]



lookback = 3
n_components = 2


dense_pre_layers = int(np.random.choice([0,1,2,4]))
dense_pre_size = int(np.random.choice([10,20,30]))
rnn_size = int(np.random.choice([10,20,30,50]))

# if len(sigs)>0:
#     sig_name=sigs[-1]
# else:
#     sig_name='None'

new_dirname = 'ncomponents_{}_lookback_{}_prelayers_{}_size_{}_rnnsize_{}_'.format(n_components,
                                                                                lookback, 
                                                                                dense_pre_layers,
                                                                                dense_pre_size,
                                                                                rnn_size)

changes_array = [['data','lookback', lookback],
                    ['model','num_pre_layers',dense_pre_layers],
                    ['model','dense_pre_size',dense_pre_size],
                    ['model','rnn_size',rnn_size],
                    ['data','n_components',n_components]]


# changes_array = [['model','num_final_layers',num_final_layers],
#                  ['model','dense_final_activation',dens_final_act],
#                  ['model','dense_final_size',dens_final_size]]




output_dir=os.path.join('/global/cscratch1/sd/al34/autoruns',subfolder,new_dirname)
make_folder_contents(input_conf, input_script, output_dir, changes_array)
os.system('sbatch {}'.format(os.path.join(output_dir,'driver.sh')))
