import pickle
import numpy as np
import os
from helpers.data_generator import process_data, DataGenerator

output_filename_base='/scratch/gpfs/jabbate/data_with_flattop' #_include_current_ramps/'

efit_type='EFITRT1'

#for efit_type in ['EFITRT1','EFITRT2','EFIT01','EFIT02']:
#for cer_type in ['cerreal','cerquick','cerauto']:
avail_profiles = ['ffprime_{}'.format(efit_type), 'press_{}'.format(efit_type), 'ffprime_{}'.format(efit_type), 'pprime_{}'.format(efit_type),
                  'dens', 'temp', 'idens', 'itemp', 'rotation',
                 'thomson_dens_{}'.format(efit_type), 'thomson_temp_{}'.format(efit_type)]
avail_actuators = ['curr', 'ech', 'gasA', 'gasB', 'gasC', 'gasD' 'gasE', 'pinj',
                   'pinj_15L', 'pinj_15R', 'pinj_21L', 'pinj_21R', 'pinj_30L',
                   'pinj_30R', 'pinj_33L', 'pinj_33R', 'tinj']
available_sigs = avail_profiles + avail_actuators + ['time']

input_profile_names = ['thomson_dens_{}'.format(efit_type), 'thomson_temp_{}'.format(efit_type)]
target_profile_names = ['temp', 'dens']
actuator_names = ['pinj', 'curr', 'tinj', 'gasA']

rawdata_path='/scratch/gpfs/jabbate/small_data/final_data.pkl'
sig_names = input_profile_names + target_profile_names + actuator_names
normalization_method = 'StandardScaler'
window_length = 3
window_overlap = 0
profile_lookback = 1
actuator_lookback = 6
lookbacks = {'thomson_dens_{}'.format(efit_type): profile_lookback,
             'thomson_temp_{}'.format(efit_type): profile_lookback,
             'temp': profile_lookback,
             'dens': profile_lookback,
             'rotation': profile_lookback,
             'press_{}'.format(efit_type): profile_lookback,
             'itemp': profile_lookback,
             'ffprime_{}'.format(efit_type): profile_lookback,
             'pinj': actuator_lookback,
             'curr': actuator_lookback,
             'tinj': actuator_lookback,
             'gasA': actuator_lookback}
lookahead = 3
sample_step = 5
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
