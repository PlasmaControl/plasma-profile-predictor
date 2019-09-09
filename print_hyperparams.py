import yaml
import os
import re
import numpy as np
import pickle
import sys
import time

base_dir='/home/wconlin/run_results/'

regular_params=['model_path','model_type', 'epochs', 'actuator_names', 'scalar_input_names', 'flattop_only', 'input_profile_names', 'target_profile_names', 'batch_size', 'predict_deltas', 'processed_filename_base', 'rawdata_path', 'sig_names', 'normalization_method', 'window_length', 'window_overlap', 'lookbacks', 'lookahead', 'sample_step', 'uniform_normalization', 'train_frac', 'val_frac', 'nshots', 'efit_type', 'profile_lookback', 'actuator_lookback', 'basefilename', 'profile_downsample', 'model_kwargs', 'std_activation', 'hinge_weight', 'mse_weight_power', 'mse_weight_edge', 'mse_power', 'verbose', 'optimizer', 'profile_length', 'runname', 'steps_per_epoch', 'val_steps']

history_params=['val_loss', 'val_target_temp_loss', 'val_target_dens_loss', 'val_target_temp_denorm_MAE', 'val_target_temp_sgn_acc', 'val_target_temp_perBLMAE', 'val_target_dens_denorm_MAE', 'val_target_dens_sgn_acc', 'val_target_dens_perBLMAE', 'loss', 'target_temp_loss', 'target_dens_loss', 'target_temp_denorm_MAE', 'target_temp_sgn_acc', 'target_temp_perBLMAE', 'target_dens_denorm_MAE', 'target_dens_sgn_acc', 'target_dens_perBLMAE']

# print(regular_params)
# print(history_params)

all_lines=os.listdir(base_dir)
count=0
#add space in the beginning for easier copy-pasting
print('\n')
for line in all_lines:
    if re.search('.pkl',line):
        count+=1
        with open(os.path.join(base_dir,line),'rb') as f:
            params=pickle.load(f)
            # if 'history' not in params:
            #     continue
            #if 'model_path' in params and params['model_path']=='/home/wconlin/run_results/model-conv2d_profiles-thomson_dens_EFITRT1-thomson_temp_EFITRT1_act-pinj-curr-tinj-gasA-gasB-gasC-gasD_targ-temp-dens_profLB-1_actLB-6_norm-RobustScaler_activ-relu_nshots-12000_ftop-True_07Sep19-14-31_Scenario-34.h5':
            for regular_param in regular_params:
                if regular_param in params:
                    print(params[regular_param],end='\t')
                else:
                    print('',end='\t')
            if 'history' in params:
                for history_param in history_params:
                    print(max(params['history'][history_param]),end='\t')
            print('')

'''
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
'''
