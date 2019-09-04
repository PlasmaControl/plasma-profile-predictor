import pickle
import numpy as np
import os
from helpers.data_generator import process_data, DataGenerator

output_filename_base='/scratch/gpfs/jabbate/small_data' #_include_current_ramps/'

efit_type='EFITRT1'

#for efit_type in ['EFITRT1','EFITRT2','EFIT01','EFIT02']:
#for cer_type in ['cerreal','cerquick','cerauto']:
# avail_profiles = ['ffprime_{}'.format(efit_type), 'press_{}'.format(efit_type), 'ffprime_{}'.format(efit_type), 'pprime_{}'.format(efit_type),
#                   'dens', 'temp', 'idens', 'itemp', 'rotation',
#                  'thomson_dens_{}'.format(efit_type), 'thomson_temp_{}'.format(efit_type)]
# avail_actuators = ['curr', 'ech', 'gasA', 'gasB', 'gasC', 'gasD' 'gasE', 'pinj',
#                    'pinj_15L', 'pinj_15R', 'pinj_21L', 'pinj_21R', 'pinj_30L',
#                    'pinj_30R', 'pinj_33L', 'pinj_33R', 'tinj']

available_sigs = ['which_gas', 'cerauto_rotation_EFITRT1', 'cerauto_rotation_EFITRT2', 'cerquick_temp_EFITRT1', 'cerquick_temp_EFITRT2', 'zmagX_EFITRT2', 'press_EFIT02', 'target_source_pinj_15R', 'press_EFIT01', 'cerquick_rotation_EFITRT1', 'cerquick_rotation_EFITRT2', 'cerauto_temp_EFIT02', 'a_EFITRT1', 'a_EFITRT2', 'target_source_pinj_15L', 'a_EFIT01', 'cerauto_temp_EFIT01', 'rmagx_EFIT02', 'rmagx_EFIT01', 'target_pinj_15R', 'cerquick_included_channels', 'zmagX_EFITRT1', 'target_pinj_15L', 'rotation', 'gasD_voltage', 'target_pinj_30R', 'triangularity_bot_EFIT01', 'triangularity_bot_EFIT02', 'thomson_temp_EFITRT1', 'thomson_temp_EFITRT2', 'beam_feedback_switch', 'target_pinj_30L', 'pprime_EFIT02', 'pprime_EFIT01', 'a_EFIT02', 'pinj_30L', 'pinj_30R', 'beam_feedback_torque_target_quantity', 'ffprime_EFITRT2', 'betan_EFITRT1', 'target_source_pinj_33R', 'zmagX_EFIT02', 'target_pinj_21R', 'zmagX_EFIT01', 'target_pinj_21L', 'target_source_pinj_33L', 'gasA_voltage', 'gasA', 'pinj_15L', 'pinj_15R', 'cerquick_temp_EFIT01', 'cerauto_rotation_EFIT01', 'thomson_temp_EFIT01', 'target_pinj_33L', 'itemp', 'thomson_temp_EFIT02', 'target_source_pinj_30R', 'target_source_pinj_30L', 'density_estimate', 'beam_feedback_torque_target_value', 'kappa_EFIT02', 'kappa_EFIT01', 'target_pinj_33R', 'target_source_pinj_21R', 'beam_target_torque', 'drsep_EFIT02', 'drsep_EFIT01', 'beam_target_power', 'gas_feedback', 'tinj', 'target_source_pinj_21L', 'betan_EFIT01', 'betan_EFIT02', 'gasB_voltage', 'ffprime_EFIT01', 'ffprime_EFIT02', 'pinj_33R', 'cerauto_rotation_EFIT02', 'gasE_voltage', 'thomson_dens_EFITRT2', 'thomson_dens_EFITRT1', 'idens', 'temp', 'cerquick_temp_EFIT02', 'dens', 'ffprime_EFITRT1', 'realtime_betan', 'gasC_voltage', 'gasB', 'curr', 'beam_feedback_power_target_quantity', 'pinj', 'pprime_EFITRT2', 'pprime_EFITRT1', 'cerauto_included_channels', 'curr_target', 'kappa_EFITRT1', 'kappa_EFITRT2', 'rmagx_EFITRT1', 'pinj_33L', 'ip_flat_duration', 'rmagx_EFITRT2', 'cerquick_rotation_EFIT01', 'pinj_21L', 'cerquick_rotation_EFIT02', 'time', 'target_density', 't_ip_flat', 'thomson_dens_EFIT02', 'thomson_dens_EFIT01', 'beam_feedback_power_target_value', 'pinj_21R', 'cerauto_temp_EFITRT1', 'cerauto_temp_EFITRT2', 'triangularity_top_EFIT02', 'gasC', 'triangularity_top_EFIT01', 'gasE', 'gasD', 'betan_EFITRT2', 'drsep_EFITRT2', 'drsep_EFITRT1', 'gas_density_or_profile_algorithm'] 

#avail_profiles + avail_actuators + ['time']

input_profile_names = ['thomson_dens_{}'.format(efit_type), 'thomson_temp_{}'.format(efit_type),'temp', 'dens']
target_profile_names = ['temp', 'dens']
actuator_names = ['pinj', 'curr', 'tinj', 'gasA','gasB','gasC','gasD','target_density','density_estimate','gas_feedback']

rawdata_path='/scratch/gpfs/jabbate/small_data/final_data.pkl'
sig_names = input_profile_names + target_profile_names + actuator_names
normalization_method = 'RobustScaler'
window_length = 3
window_overlap = 0
profile_lookback = 1
actuator_lookback = 6
lookbacks = {}
for sig in input_profile_names + target_profile_names:
   lookbacks[sig] = profile_lookback
for sig in actuator_names:
   lookbacks[sig] = actuator_lookback
lookahead = 3
sample_step = 1
uniform_normalization = True
train_frac = 0.8
val_frac = 0.2
nshots = 12000 #TODO: replace with nbatches
flattop_only=True

assert(all(elem in available_sigs for elem in sig_names))

traindata, valdata, normalization_dict = process_data(rawdata_path, sig_names,
                                                      normalization_method, window_length,
                                                      window_overlap, lookbacks,
                                                      lookahead, sample_step,
                                                      uniform_normalization, train_frac,
                                                      val_frac, nshots,
                                                      flattop_only=flattop_only)

    
with open(os.path.join(output_filename_base,'train.pkl'),'wb') as f:
   pickle.dump(traindata,f)
with open(os.path.join(output_filename_base,'val.pkl'),'wb') as f:
   pickle.dump(valdata,f)

# train_shapes={}
# val_shapes={}
# for sig in traindata:
#     fp=np.memmap(os.path.join(output_filename_base,'{}_{}'.format('train',sig)), dtype='float64', mode='w+',shape=traindata[sig].shape)
#     fp[:]=traindata[sig][:]
#     train_shapes[sig]=traindata[sig].shape
# for sig in valdata:
#     fp=np.memmap(os.path.join(output_filename_base,'{}_{}'.format('val',sig)), dtype='float64', mode='w+',shape=valdata[sig].shape)
#     fp[:]=valdata[sig][:]
#     val_shapes[sig]=valdata[sig].shape

# with open(os.path.join(output_filename_base,'train_shapes.pkl'),'wb') as f:
#     pickle.dump(train_shapes,f)
# with open(os.path.join(output_filename_base,'val_shapes.pkl'),'wb') as f:
#     pickle.dump(val_shapes,f)

param_dict={'normalization_dict': normalization_dict,
            'rawdata_path': rawdata_path,
            'sig_names': sig_names,
            'normalization_method': normalization_method,
            'window_length': window_length,
            'window_overlap': window_overlap,
            'lookbacks': lookbacks,
            'lookahead': lookahead,
            'sample_step': sample_step,
            'uniform_normalization': uniform_normalization,
            'train_frac': train_frac,
            'val_frac': val_frac,
            'nshots': nshots,
            'flattop_only': flattop_only,
            'input_profile_names': input_profile_names,
            'target_profile_names': target_profile_names,
            'actuator_names': actuator_names,
            'efit_type': efit_type,
            'profile_lookback': profile_lookback,
            'actuator_lookback': actuator_lookback,
            'basefilename': output_filename_base}

with open(os.path.join(output_filename_base,'param_dict.pkl'),'wb') as f:
    pickle.dump(param_dict,f)
