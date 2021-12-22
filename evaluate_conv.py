import sys
import os
import pickle
import keras
import datetime
import matplotlib
import copy
import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath('../'))
import helpers
from helpers.data_generator import process_data, DataGenerator
from helpers.normalization import normalize, denormalize, renormalize
# from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
# from helpers.custom_losses import percent_correct_sign, baseline_MAE, normed_mse, mean_diff_sum_2, max_diff_sum_2, mean_diff2_sum2, max_diff2_sum2

excluded_year_only=False

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

def sigma(inp,true,prediction):
    eps=prediction-true
    
    num=np.linalg.norm(eps,axis=-1)
    denom=np.linalg.norm(true,axis=-1)
    
    included_inds=np.where(~np.isclose(denom,0))[0]
    return num[included_inds]/denom[included_inds]

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

# see 2008 TGLF paper for def of sigma and f
# def sigma(eps, T):
#     return np.linalg.norm(eps,axis=-1)/np.linalg.norm(T,axis=-1)

# def f(eps, T): 
#     return np.mean(eps,axis=-1)/np.linalg.norm(T,axis=-1)

metrics = {'mean_squared_error':mean_squared_error,
          'mean_absolute_error':mean_absolute_error,
          'median_absolute_error':median_absolute_error,
          'percentile25_absolute_error':percentile25_absolute_error,
          'percentile75_absolute_error':percentile75_absolute_error,
          'median_squared_error':median_squared_error,
          'percentile25_squared_error':percentile25_squared_error,
          'percentile75_squared_error':percentile75_squared_error}

##########
# load model and scenario
##########
base_path = '/projects/EKOLEMEN/profile_predictor/'
folders = ['run_results_rt'] #['run_results_bt_scan/','run_results_time_scan/']



for folder in folders:
    files =  [os.path.join(base_path,folder,foo) for foo in os.listdir(os.path.join(base_path,folder)) if foo.endswith('.pkl')]
    files=['/projects/EKOLEMEN/profile_predictor/run_results_campaign_year/model-conv2d_profiles-dens-temp-rotation-q_EFIT01-press_EFIT01_act-target_density-pinj-tinj-curr_target-bt_10Oct20-18-42_Scenario-14_params.pkl']
    # files = ['/projects/EKOLEMEN/profile_predictor/run_results_time_scan/model-conv2d_profiles-dens-temp-rotation-q_EFIT01-press_EFIT01_act-target_density-pinj-tinj-curr_target-bt_22Aug20-19-09_Scenario-170_params.pkl',
    #          '/projects/EKOLEMEN/profile_predictor/run_results_time_scan/model-conv2d_profiles-dens-temp-rotation-q_EFIT01-press_EFIT01_act-target_density-pinj-tinj-curr_target-bt_22Aug20-19-19_Scenario-171_params.pkl',
    #          '/projects/EKOLEMEN/profile_predictor/run_results_time_scan/model-conv2d_profiles-dens-temp-rotation-q_EFIT01-press_EFIT01_act-target_density-pinj-tinj-curr_target-bt_22Aug20-20-00_Scenario-174_params.pkl',
    #          '/projects/EKOLEMEN/profile_predictor/run_results_08-25/model-conv2d_profiles-dens-temp-rotation-q_EFIT01-press_EFIT01_act-target_density-pinj-tinj-curr_target-bt_25Aug20-18-08_Scenario-20_params.pkl',
    #          '/projects/EKOLEMEN/profile_predictor/run_results_time_scan/model-conv2d_profiles-dens-temp-rotation-q_EFIT01-press_EFIT01_act-target_density-pinj-tinj-curr_target-bt_22Aug20-04-40_Scenario-106_params.pkl','/projects/EKOLEMEN/profile_predictor/run_results_time_scan/model-conv2d_profiles-dens-temp-rotation-q_EFIT01-press_EFIT01_act-target_density-pinj-tinj-curr_target-bt_22Aug20-07-05_Scenario-118_params.pkl']
    for file_path in files:
        try:
            with open(file_path, 'rb') as f:
                scenario = pickle.load(f, encoding='latin1')
            # if 'evaluation_metrics' in scenario:
            #     continue

            model_path = file_path[:-11] + '.h5'
            prev_time=time.time()
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path, compile=False)
                print('loaded model: ' + model_path.split('/')[-1])
                print('took {}s'.format(time.time() - prev_time))
            else:
                print('no model for path:',model_path)
                continue

            full_data_path = '/scratch/gpfs/jabbate/full_data_with_error/small_data.pkl' #train_data.pkl'
            rt_data_path = '/scratch/gpfs/jabbate/test_rt/final_data.pkl'
            prev_time=time.time()
            # for test data
            if not excluded_year_only:
                traindata, valdata, normalization_dict = helpers.data_generator.process_data(full_data_path,
                                                                                             scenario['sig_names'],
                                                                                             scenario['normalization_method'],
                                                                                             scenario['window_length'],
                                                                                             scenario['window_overlap'],
                                                                                             scenario['lookbacks'],
                                                                                             scenario['lookahead'],
                                                                                             scenario['sample_step'],
                                                                                             scenario['uniform_normalization'],
                                                                                             scenario['train_frac'],
                                                                                             scenario['val_frac'],
                                                                                             scenario['nshots'],
                                                                                             1, #0, #scenario['verbose']
                                                                                             scenario['flattop_only'],
                                                                                             randomize=False,
                                                                                             pruning_functions=scenario['pruning_functions'],
                                                                                             excluded_shots = scenario['excluded_shots'],
                                                                                             delta_sigs = [],
                                                                                             invert_q=scenario.setdefault('invert_q',False),
                                                                                             val_idx=0) #scenario['val_idx']) #if 0, then the val set is really the test set
                # for testing on the campaign year that we excluded during training
            else:
                excluded_shots=copy.deepcopy(scenario['excluded_shots'])

                for year in range(2010,2020):
                    if 'year_{}'.format(year) in excluded_shots:
                        excluded_shots.remove('year_{}'.format(year))
                    else:
                        excluded_shots.append('year_{}'.format(year))

                traindata, valdata, normalization_dict = helpers.data_generator.process_data(full_data_path,
                                                                                             scenario['sig_names'],
                                                                                             scenario['normalization_method'],
                                                                                             scenario['window_length'],
                                                                                             scenario['window_overlap'],
                                                                                             scenario['lookbacks'],
                                                                                             scenario['lookahead'],
                                                                                             scenario['sample_step'],
                                                                                             scenario['uniform_normalization'],
                                                                                             0, #scenario['train_frac'],
                                                                                             1, #scenario['val_frac'],
                                                                                             scenario['nshots'],
                                                                                             1, #0, #scenario['verbose']
                                                                                             scenario['flattop_only'],
                                                                                             randomize=False,
                                                                                             pruning_functions=scenario['pruning_functions'],
                                                                                             excluded_shots = excluded_shots,
                                                                                             delta_sigs = [],
                                                                                             invert_q=scenario.setdefault('invert_q',False),
                                                                                             val_idx=None) #if scenario['val_idx'] then whatever random final number in shot number was used for validation during training; if 0, then the val set is the test set; if None then it does a random split of samples based on the train and val frac

            print('Data processing took {}s'.format(time.time()-prev_time))
            valdata = helpers.normalization.renormalize(
                helpers.normalization.denormalize(
                    valdata.copy(),normalization_dict, verbose=0),
                scenario['normalization_dict'],verbose=0)

            val_generator = DataGenerator(valdata,
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
                                            sample_weights = None)


            predictions_arr = model.predict_generator(val_generator, verbose=0, workers=4, use_multiprocessing=True)
            predictions = {sig: arr for sig, arr in zip(scenario['target_profile_names'],predictions_arr)}

            inp = {sig:[] for sig in scenario['target_profile_names']}
            true = {sig:[] for sig in scenario['target_profile_names']}
            baseline = {sig:[] for sig in scenario['target_profile_names']}
            for i in range(len(val_generator)):
                sample = val_generator[i]
                for sig in scenario['target_profile_names']:
                    inp[sig].append(sample[0]['input_'+sig].squeeze())
                    true[sig].append(sample[0]['input_'+sig].squeeze() + sample[1]['target_'+sig])
                    baseline[sig].append(sample[1]['target_'+sig])

            inp = {sig:np.concatenate(inp[sig],axis=0).squeeze() for sig in scenario['target_profile_names']}
            true = {sig:np.concatenate(true[sig],axis=0).squeeze() for sig in scenario['target_profile_names']}

            denormed_predictions={sig: helpers.normalization.denormalize_arr(predictions[sig]+inp[sig],scenario['normalization_dict'][sig]) for sig in scenario['target_profile_names']}
            inp={sig: helpers.normalization.denormalize_arr(inp[sig],scenario['normalization_dict'][sig]) for sig in scenario['target_profile_names']}
            true={sig: helpers.normalization.denormalize_arr(true[sig],scenario['normalization_dict'][sig]) for sig in scenario['target_profile_names']}

            baseline = {sig:np.concatenate(baseline[sig],axis=0) for sig in scenario['target_profile_names']}

            # calculate errors
            model_err = {sig: np.abs(predictions[sig] - baseline[sig]) for sig in scenario['target_profile_names']}
            baseline_err = {sig: np.abs(baseline[sig]) for sig in scenario['target_profile_names']}

            evaluation_metrics = {}
            for metric_name,metric in metrics.items():
                s = 0
                for sig in scenario['target_profile_names']:
                    key = sig + '_' + metric_name
                    val = metric(baseline[sig],predictions[sig])
                    s += val/len(scenario['target_profile_names'])
                    evaluation_metrics[key] = val
                    print(key)
                    print(val)
                evaluation_metrics[metric_name] = s

            for sig in scenario['target_profile_names']:
                evaluation_metrics['sigma_ML_'+sig]=sigma(inp[sig],true[sig],denormed_predictions[sig])
                evaluation_metrics['sigma_baseline_'+sig]=sigma(inp[sig],true[sig],inp[sig])
                evaluation_metrics['inp_'+sig]=inp[sig]
                evaluation_metrics['true_'+sig]=true[sig]
                evaluation_metrics['denormed_predictions_'+sig]=denormed_predictions[sig]

            if excluded_year_only:
                scenario['evaluation_metrics_excluded_year_only'] = evaluation_metrics
            else:
                scenario['evaluation_metrics'] = evaluation_metrics
            if 'date' not in scenario:
                scenario['date'] = datetime.datetime.strptime(scenario['runname'].split('_')[-2],'%d%b%y-%H-%M')

            prev_time=time.time()
            # with open(file_path,'wb+') as f:
            #     pickle.dump(copy.deepcopy(scenario),f)
            print('Repickling took {}s'.format(time.time()-prev_time))

            print('saved evaluation metrics')
            #print(evaluation_metrics)
        except Exception as e: 
            print(e)

    print('done')
