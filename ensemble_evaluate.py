import pickle
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../'))
import helpers
from helpers.data_generator import process_data, DataGenerator
from helpers.custom_losses import normed_mse, mean_diff_sum_2, max_diff_sum_2, mean_diff2_sum2, max_diff2_sum2
import time
from time import strftime, localtime
import matplotlib
from matplotlib import pyplot as plt
import copy
from helpers.normalization import normalize, denormalize, renormalize
from tqdm import tqdm

t0 = time.time()

config = tf.ConfigProto(intra_op_parallelism_threads=16,
                            inter_op_parallelism_threads=16,
                            allow_soft_placement=True,
                            device_count={'CPU': 8,
                                          'GPU': 1})
session = tf.Session(config=config)
K.set_session(session)

results_path = os.path.expanduser('~/ensemble_results_02_13.pkl')

base_path = os.path.expanduser('~/')
folders = ['run_results_02_13/']

profiles = ['temp','dens','itemp','rotation','q']
actuators = ['target_density', 'pinj', 'tinj', 'curr_target']
scalars = ['density_estimate', 'li_EFIT01', 'volume_EFIT01', 'triangularity_top_EFIT01', 'triangularity_bot_EFIT01']
scenarios = []
model_paths = []
for folder in folders:
    files =  [foo for foo in os.listdir(base_path+folder) if foo.endswith('.pkl')]
    for file in files:
        file_path = base_path + folder + file
        with open(file_path, 'rb') as f:
            scenario = pickle.load(f, encoding='latin1')
        if set(scenario['input_profile_names']) == set(profiles) and \
        set(scenario['target_profile_names']) == set(profiles) and \
        set(scenario['actuator_names']) == set(actuators) and \
        set(scenario['scalar_input_names']) == set(scalars):
            scenarios.append(scenario)
            model_path = file_path[:-11] + '.h5'
            model_paths.append(model_path)
scenario = scenarios[0]


models = []
for model_path in model_paths:
    model = keras.models.load_model(model_path, compile=False)
    models.append(model)
print('loaded models, time={}'.format(time.time()-t0))
    
full_data_oath = '/scratch/gpfs/jabbate/full_data_with_error/train_data_full.pkl'
test_data_path = '/scratch/gpfs/jabbate/full_data_with_error/test_data.pkl' 
traindata, valdata, normalization_dict = helpers.data_generator.process_data(test_data_path,
                                                      scenario['sig_names'],
                                                      scenario['normalization_method'],
                                                      scenario['window_length'],
                                                      scenario['window_overlap'],
                                                      scenario['lookbacks'],
                                                      scenario['lookahead'],
                                                      scenario['sample_step'],
                                                      scenario['uniform_normalization'],
                                                      1, #scenario['train_frac'],
                                                      0, #scenario['val_frac'],
                                                      scenario['nshots'],
                                                      2, #scenario['verbose']
                                                      scenario['flattop_only'],
                                                      randomize=False,
                                                      pruning_functions=scenario['pruning_functions'],
                                                      excluded_shots = scenario['excluded_shots'],
                                                      delta_sigs = [],
                                                      uncertainties=False)


traindata = helpers.normalization.renormalize(helpers.normalization.denormalize(traindata.copy(),normalization_dict),scenario['normalization_dict'])
psi = np.linspace(0,1,scenario['profile_length'])

train_generator = DataGenerator(traindata,
                                scenario['batch_size'],
                                scenario['input_profile_names'],
                                scenario['actuator_names'],
                                scenario['target_profile_names'],
                                scenario['scalar_input_names'],
                                scenario['lookbacks'],
                                scenario['lookahead'],
                                scenario['predict_deltas'],
                                scenario['profile_downsample'],
                                False,
                                sample_weights = 'std',
                                return_uncertainties=False) #scenario['shuffle_generators'])

losses = {'mean_squared_error': keras.losses.mean_squared_error,
          'mean_absolute_error': keras.losses.mean_absolute_error,
          'normed_mse': normed_mse,
          'mean_diff_sum_2': mean_diff_sum_2,
          'max_diff_sum_2': max_diff_sum_2, 
          'mean_diff2_sum2': mean_diff2_sum2, 
          'max_diff2_sum2': max_diff2_sum2}
ensemble_evaluation_metrics = {}
for profile in model.output_names:
    for metric in losses.keys():
        ensemble_evaluation_metrics[profile+'_'+metric] = []
        
results_data = {i:{} for i in range(len(train_generator))}  

for index in range(len(train_generator)):
    t0 = time.time()
    inputs,targets,_ = train_generator[index]
    results_data[index]['targets'] = copy.deepcopy(targets)
    results_data[index]['inputs'] = copy.deepcopy(inputs)
    results_data[index]['shotnum'] = copy.deepcopy(train_generator.cur_shotnum)
    results_data[index]['times'] = copy.deepcopy(train_generator.cur_times)
    
    predictions = []
    for j, model in enumerate(models):
        pred = model.predict_on_batch(inputs)
        predictions.append(pred)
    
    uncertainties = np.std(predictions,axis=0)
    predictions = np.mean(predictions,axis=0)
    
    predictions = {name:p for name, p in zip(model.output_names,predictions)}
    uncertainties = {name:p for name, p in zip(model.output_names,uncertainties)}
    
    results_data[index]['predictions'] = copy.deepcopy(predictions)
    results_data[index]['uncertainties'] = copy.deepcopy(uncertainties)

        
    for profile in model.output_names:
        for name, metric in losses.items():
            ensemble_evaluation_metrics[profile+'_'+name].append(K.eval(metric(targets[profile],predictions[profile]))) 

    results_data['ensemble_metrics'] = ensemble_evaluation_metrics
    
    with open(results_path,'wb+') as f:
        pickle.dump(results_data,f)
    print('finished {}/{}'.format(index,len(train_generator)))
    print('time={}'.format(time.time()-t0))

            
            
for key,val in ensemble_evaluation_metrics.items():
    ensemble_evaluation_metrics[key] = np.mean(val)


for metric in losses:
    name = metric if isinstance(metric,str) else str(metric.__name__)
    s = 0
    for key,val in ensemble_evaluation_metrics.items():
        if name in key:
            s += val
    ensemble_evaluation_metrics[name] = s

for key, val in ensemble_evaluation_metrics.items():
    print('{}: {:.3e}'.format(key,val))