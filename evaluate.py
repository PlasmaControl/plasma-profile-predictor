import sys
import os
import pickle
import keras
import datetime
import matplotlib
import copy

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath('../'))
import helpers
from helpers.data_generator import process_data, DataGenerator, AutoEncoderDataGenerator
from helpers.normalization import normalize, denormalize, renormalize
# from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
# from helpers.custom_losses import percent_correct_sign, baseline_MAE, normed_mse, mean_diff_sum_2, max_diff_sum_2, mean_diff2_sum2, max_diff2_sum2

##########
# set tf session
##########
config = tf.ConfigProto(intra_op_parallelism_threads=16,
                            inter_op_parallelism_threads=16,
                            allow_soft_placement=True,
                            device_count={'CPU': 8,
                                          'GPU': 1})
session = tf.Session(config=config)
K.set_session(session)


##########
# metrics
##########
def mean_squared_error(true,pred):
    return np.mean((true-pred)**2)

def mean_absolute_error(true,pred):
    return np.mean(np.abs(true-pred))

def median_absolute_error(true,pred):
    return np.median(np.abs(true-pred))

def percentile25_absolute_error(true,pred):
    return np.percentile(np.abs(true-pred),25)

def percentile75_absolute_error(true,pred):
    return np.percentile(np.abs(true-pred),75)

def median_squared_error(true,pred):
    return np.median((true-pred)**2)

def percentile25_squared_error(true,pred):
    return np.percentile((true-pred)**2,25)

def percentile75_squared_error(true,pred):
    return np.percentile((true-pred)**2,75)

def huber_error(true,pred):
    return np.mean(np.where(np.abs(pred-true) < 0.7, 5*(pred - true)**2, 0.7*(np.abs(pred-true)-0.5*0.7)))

metrics = {'mean_squared_error':mean_squared_error,
          'mean_absolute_error':mean_absolute_error,
          'median_absolute_error':median_absolute_error,
          'percentile25_absolute_error':percentile25_absolute_error,
          'percentile75_absolute_error':percentile75_absolute_error,
          'median_squared_error':median_squared_error,
          'percentile25_squared_error':percentile25_squared_error,
          'percentile75_squared_error':percentile75_squared_error,
          'huber_error':huber_error
          }

##########
# load model and scenario
##########
base_path = os.path.expanduser('~')
folders = ['/test_run/']
           
for folder in folders:
    files =  [foo for foo in os.listdir(base_path+folder) if foo.endswith('.pkl')]
    for file in files:
        try:
            file_path = base_path + folder + file
            with open(file_path, 'rb') as f:
                scenario = pickle.load(f, encoding='latin1')

            #if 'evaluation_metrics' in scenario:
                #continue

            model_path = file_path[:-11] + '.h5'
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path, compile=False)
                print('loaded model: ' + model_path.split('/')[-1])
            else:
                print('no model for path:',model_path)
                continue

            full_data_path = '/scratch/gpfs/jabbate/full_data_with_error/train_data.pkl'# '/scratch/gpfs/jabbate/small_data.pkl'

            
            traindata, valdata, normalization_dict = helpers.data_generator.process_data(full_data_path,
                                                              scenario['sig_names'],
                                                              scenario['normalization_method'],
                                                              scenario['window_length'],
                                                              scenario['window_overlap'],
                                                              scenario['lookback'],
                                                              scenario['lookahead'],
                                                              scenario['sample_step'],
                                                              scenario['uniform_normalization'],
                                                              0.9, #scenario['train_frac'],
                                                              0.1, #scenario['val_frac'],
                                                              scenario['nshots'],
                                                              0, #scenario['verbose']
                                                              scenario['flattop_only'],
                                                              randomize=False,
                                                              pruning_functions=scenario['pruning_functions'],
                                                              excluded_shots = scenario['excluded_shots'],
                                                              delta_sigs = [],
                                                              invert_q=scenario.setdefault('invert_q',False),
                                                              val_idx = None)
            
            
            valdata = helpers.normalization.renormalize(
                helpers.normalization.denormalize(
                    valdata.copy(),normalization_dict, verbose=0),
                scenario['normalization_dict'],verbose=0)

            train_generator = AutoEncoderDataGenerator(valdata,
                                            scenario['batch_size'],
                                            scenario['profile_names'],
                                            scenario['actuator_names'],
                                            scenario['scalar_names'],
                                            scenario['lookback'],
                                            scenario['lookahead'],
                                            scenario['profile_downsample'],
                                            scenario['state_latent_dim'],
                                            shuffle =False)
            
    #         optimizer = keras.optimizers.Adam()
    #         loss = keras.metrics.mean_squared_error
    #         metrics = [keras.metrics.mean_squared_error, 
    #                    keras.metrics.mean_absolute_error, 
    #                    normed_mse, 
    #                    mean_diff_sum_2, 
    #                    max_diff_sum_2, 
    #                    mean_diff2_sum2, 
    #                    max_diff2_sum2]
    #         model.compile(optimizer, loss, metrics)

    #         outs = model.evaluate_generator(train_generator, verbose=0, workers=4, use_multiprocessing=True)
        
            predictions_arr = model.predict_generator(train_generator, verbose=0, workers=4, use_multiprocessing=True)
            
            output_names = ['u_residual','x_residual','linear_system_residual']
            

            predictions = {sig: arr for sig, arr in zip(output_names,predictions_arr)}            
            
#             for key,val in predictions.items():
#                 print(key,val.shape)

            true_output = {sig:[] for sig in output_names}
            for i in range(len(train_generator)):
                sample = train_generator[i]
                for sig in output_names:
                    true_output[sig].append(sample[1][sig])
            true_output = {sig:np.concatenate(true_output[sig],axis=0) for sig in output_names}
            
            for key,val in true_output.items():
                print(key,val.shape)

            evaluation_metrics = {}
            for metric_name,metric in metrics.items():
                s = 0
                for sig in output_names:
                    key = sig + '_' + metric_name
                    val = metric(true_output[sig],predictions[sig])
                    s += val
                    evaluation_metrics[key] = val
                    print(key)
                    print(val)
                evaluation_metrics[metric_name] = s
            
            # Want to additionally save the true output and predictions just in case for plotting
            scenario['true_output'] = true_output
            scenario['predictions'] = predictions
            print('Saved True output and Predictions')
            
            scenario['evaluation_metrics'] = evaluation_metrics
            if 'date' not in scenario:
                scenario['date'] = datetime.datetime.strptime(scenario['runname'].split('_')[-2],'%d%b%y-%H-%M')

            
            with open(file_path,'wb+') as f:
                pickle.dump(copy.deepcopy(scenario),f)

            print('saved evaluation metrics')
            print(evaluation_metrics)
        
        except Exception as e: 
            print(e)

    print('done')