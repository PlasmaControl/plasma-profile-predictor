import pickle
import keras
import numpy as np
import random
import os
import sys
import itertools
import copy
import datetime
from collections import OrderedDict

import helpers
from helpers.data_generator import process_data, DataGenerator
from helpers.hyperparam_helpers import make_bash_scripts
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE, percent_correct_sign
from helpers.callbacks import CyclicLR, TensorBoardWrapper
from models.LSTMConv2D import get_model_simple_lstm, get_model_conv2d, get_model_linear_systems
from models.LSTMConv1D import build_lstmconv1d_joe, build_dumb_simple_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras import backend as K


def main(scenario_index=-2):

    ###################
    # set session
    ###################
    num_cores = 8
    req_mem = 48 # gb
    ngpu = 1
    
    
#     seed_value= 0
#     os.environ['PYTHONHASHSEED']=str(seed_value)
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     tf.set_random_seed(seed_value)
    
    config = tf.ConfigProto(intra_op_parallelism_threads=4*num_cores,
                            inter_op_parallelism_threads=4*num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': 1,
                                          'GPU': ngpu})
    session = tf.Session(config=config)
    K.set_session(session)

    ###############
    # global stuff
    ###############
    
    checkpt_dir = os.path.expanduser("~/run_results_03_04/")
    if not os.path.exists(checkpt_dir):
        os.makedirs(checkpt_dir)
        
    ###############
    # scenarios
    ###############

    efit_type='EFIT02'

    default_scenario = {'actuator_names': ['target_density','pinj','tinj','curr_target'],
                        'input_profile_names': ['dens','temp','itemp','q','rotation'],
                        'target_profile_names': ['dens','temp','itemp','q','rotation'],
                        'scalar_input_names' : ['density_estimate','li_EFIT01','volume_EFIT01','triangularity_top_EFIT01','triangularity_bot_EFIT01'],
                        'profile_downsample' : 2,
                        'model_type' : 'conv2d',
                        'model_kwargs': {'max_channels': 32},
                        'std_activation' : 'relu',
                        'sample_weighting':'std',
                        'loss_function': 'mse',
                        'loss_function_kwargs':{},                  
                        'batch_size' : 128,
                        'epochs' : 200,
                        'flattop_only': True,
                        'predict_deltas' : True,
#                         'raw_data_path':'/scratch/gpfs/jabbate/full_data/train_data_full.pkl',
                        'raw_data_path':'/scratch/gpfs/jabbate/old_stuff/new_data/final_data.pkl',
                        'process_data':True,
                        'invert_q': True,
                        'processed_filename_base': '/scratch/gpfs/jabbate/data_60_ms_randomized_',
                        'optimizer': 'adagrad',
                        'optimizer_kwargs': {},
                        'shuffle_generators': True,
                        'pruning_functions':['remove_nan',
                                             'remove_dudtrip',
                                             'remove_I_coil',
                                             'remove_non_gas_feedback',
#                                              'remove_non_beta_feedback',
                                             'remove_ECH'],
                        'normalization_method': 'RobustScaler',
                        'window_length': 1,
                        'window_overlap': 0,
                        'profile_lookback': 0,
                        'actuator_lookback': 6,
                        'lookahead': 4,
                        'sample_step': 1,
                        'uniform_normalization': True,
                        'train_frac': 0.8,
                        'val_frac': 0.2,
                        'val_idx': np.random.randint(1,10)
                        'nshots': 12000,
                        'excluded_shots': ['topology_TOP', 
                                           'topology_OUT',
                                           'topology_MAR',
                                           'topology_IN',
                                           'topology_DN',
                                           'topology_BOT',
                                           'test_set']} 



    
    scenarios_dict = OrderedDict()
    scenarios_dict['models'] = [{'model_type': 'conv2d', 'model_kwargs': {'max_channels':8}},
                                {'model_type': 'conv2d', 'model_kwargs': {'max_channels':12}},
                                {'model_type': 'conv2d', 'model_kwargs': {'max_channels':16, 'l2':1e-5}},
                                {'model_type': 'conv1d', 'model_kwargs': {'max_channels':8}},
                                {'model_type': 'conv1d', 'model_kwargs': {'max_channels':12}},
                                {'model_type': 'conv1d', 'model_kwargs': {'max_channels':16}}]
    scenarios_dict['profiles'] = [{'input_profile_names': ['thomson_dens_EFITRT1','thomson_temp_EFITRT1', 'cerquick_temp_EFITRT1',
                                                         'q_EFITRT1','cerquick_rotation_EFITRT1','press_EFITRT1'], 
                               'target_profile_names': ['thomson_dens_EFITRT1','thomson_temp_EFITRT1', 'cerquick_temp_EFITRT1',
                                                         'q_EFITRT1','cerquick_rotation_EFITRT1','press_EFITRT1']},
#                                 {'input_profile_names': ['thomson_dens_EFITRT1','thomson_temp_EFITRT1', 'cerquick_temp_EFITRT1',
#                                                          'q_EFITRT1','cerquick_rotation_EFITRT1','ffprime_EFITRT1','press_EFITRT1'], 
#                                'target_profile_names': ['thomson_dens_EFITRT1','thomson_temp_EFITRT1', 'cerquick_temp_EFITRT1',
#                                                          'q_EFITRT1','cerquick_rotation_EFITRT1','ffprime_EFITRT1','press_EFITRT1']},
                                {'input_profile_names': ['thomson_dens_EFITRT1','thomson_temp_EFITRT1', 'cerquick_temp_EFITRT1',
                                                         'q_EFITRT1','cerquick_rotation_EFITRT1'], 
                               'target_profile_names': ['thomson_dens_EFITRT1','thomson_temp_EFITRT1', 'cerquick_temp_EFITRT1',
                                                         'q_EFITRT1','cerquick_rotation_EFITRT1']}]
#     scenarios_dict['profiles'] = [{'input_profile_names': ['dens','temp', 'itemp','q_EFIT01','rotation','ffprime_EFIT01','press_EFIT01'],
#                                    'target_profile_names': ['dens','temp', 'itemp','q_EFIT01','rotation','ffprime_EFIT01','press_EFIT01']},
#                                   {'input_profile_names': ['dens','temp', 'q_EFIT01','rotation','ffprime_EFIT01','press_EFIT01'],
#                                    'target_profile_names': ['dens','temp','q_EFIT01','rotation','ffprime_EFIT01','press_EFIT01']},
#                                   {'input_profile_names': ['dens','temp', 'q_EFIT01','rotation','press_EFIT01'],
#                                    'target_profile_names': ['dens','temp', 'q_EFIT01','rotation','press_EFIT01']}]
    scenarios_dict['sample_weighting'] = [{'sample_weighting':'std'}]
    scenarios_dict['scalars'] = [{'scalar_input_names' : ['density_estimate','li_EFITRT1','volume_EFITRT1','kappa_EFITRT1',
                                                          'triangularity_top_EFITRT1','triangularity_bot_EFITRT1']},
                                 {'scalar_input_names':[]}]
#     scenarios_dict['scalars'] = [{'scalar_input_names' : ['density_estimate','li_EFIT01','volume_EFIT01','kappa_EFIT01',
#                                                           'triangularity_top_EFIT01','triangularity_bot_EFIT01']},
#                                  {'scalar_input_names':[]}]
    scenarios_dict['batch_size'] = [{'batch_size': 128}]
    scenarios_dict['process_data'] = [{'process_data':True}]
    scenarios_dict['predict_deltas'] = [{'predict_deltas': True},{'predict_deltas': False}]
    scenarios_dict['epochs'] = [{'epochs': 200} for i in range(1)]
    scenarios_dict['loss'] = [{'loss_function': 'mse'},
                              {'loss_function':'mae'}]
#                               {'loss_function':'normed_mse'},
#                               {'loss_function':'mean_diff_sum_2'},
#                               {'loss_function':'max_diff_sum_2'},
#                               {'loss_function':'mean_diff2_sum2'},
#                               {'loss_function':'max_diff2_sum2'}]
    

    scenarios = []
    runtimes = []
    for scenario in itertools.product(*list(scenarios_dict.values())):
        foo = {k: v for d in scenario for k, v in d.items()}
        scenarios.append(foo)
        if foo['model_type'] == 'conv2d':
            runtimes.append(2.5*128/foo['batch_size']*foo['epochs']+30)
        elif foo['model_type'] == 'simple_dense':
            runtimes.append(1*128/foo['batch_size']*foo['epochs']+30)
        elif foo['model_type'] == 'conv1d':
            runtimes.append(2*128/foo['batch_size']*foo['epochs']+30)
        elif foo['model_type'] == 'simple_lstm':
            runtimes.append(1*128/foo['batch_size']*foo['epochs']+30)
        elif foo['model_type'] == 'linear_systems':
            runtimes.append(1.5*128/foo['batch_size']*foo['epochs']+30)
        else:
            runtimes.append(4*60)
    num_scenarios = len(scenarios)


    ###############
    # Batch Run
    ###############
    if scenario_index == -1:
        make_bash_scripts(num_scenarios, checkpt_dir, num_cores, ngpu, req_mem, runtimes)
        print('Created Driver Scripts in ' + checkpt_dir)
        for i in range(num_scenarios):
            os.system('sbatch {}'.format(os.path.join(
                checkpt_dir, 'driver' + str(i) + '.sh')))
        print('Jobs submitted, exiting')
        return

    ###############
    # Load Scenario and Data
    ###############    
    if scenario_index >= 0:
        verbose=2
        print('Loading Scenario ' + str(scenario_index) + ':')
        scenario = scenarios[scenario_index]
    else:
        verbose=1
        print('Loading Default Scenario:')
        scenario = default_scenario
    print(scenario)

    if scenario['process_data']:
        scenario.update({k:v for k,v in default_scenario.items() if k not in scenario.keys()})
        scenario['lookbacks'] = {}
        for sig in scenario['input_profile_names'] + scenario['target_profile_names']:
            scenario['lookbacks'][sig] = scenario['profile_lookback']
        for sig in scenario['actuator_names'] + scenario['scalar_input_names']:
            scenario['lookbacks'][sig] = scenario['actuator_lookback']
        scenario['sig_names'] = scenario['input_profile_names'] + scenario['target_profile_names'] + scenario['actuator_names'] + scenario['scalar_input_names']

        if 'raw_data_path' not in scenario.keys():
            scenario['raw_data_path'] = default_scenario['raw_data_path']
        traindata, valdata, normalization_dict = process_data(scenario['raw_data_path'],
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
                                                              verbose,
                                                              scenario['flattop_only'],
                                                              pruning_functions = scenario['pruning_functions'],
                                                              excluded_shots = scenario['excluded_shots'],
                                                              invert_q = scenario['invert_q'],
                                                              val_idx = scenario['val_idx'])

        scenario['dt'] = np.mean(np.diff(traindata['time']))/1000 # in seconds
        scenario['normalization_dict'] = normalization_dict

    else:        
        if 'processed_filename_base' not in scenario.keys():
            scenario['processed_filename_base'] = default_scenario['processed_filename_base']
        
        if scenario['flattop_only']:
            scenario['processed_filename_base'] += 'flattop/'
        else:
            scenario['processed_filename_base'] += 'all/'
         
        with open(os.path.join(scenario['processed_filename_base'], 'param_dict.pkl'), 'rb') as f:
            param_dict = pickle.load(f)
        with open(os.path.join(scenario['processed_filename_base'], 'train.pkl'), 'rb') as f:
            traindata = pickle.load(f)
        with open(os.path.join(scenario['processed_filename_base'], 'val.pkl'), 'rb') as f:
            valdata = pickle.load(f)
        print('Data Loaded')
        scenario.update({k:v for k,v in param_dict.items() if k not in scenario.keys()})
        scenario.update({k:v for k,v in default_scenario.items() if k not in scenario.keys()})


  
    scenario['profile_length'] = int(np.ceil(65/scenario['profile_downsample']))
    scenario['date'] = datetime.datetime.now()
    scenario['runname'] = 'model-' + scenario['model_type'] + \
              '_profiles-' + '-'.join(scenario['input_profile_names']) + \
              '_act-' + '-'.join(scenario['actuator_names']) + \
              datetime.datetime.strftime(scenario['date'],"_%d%b%y-%H-%M", )

    if scenario_index >= 0:
        scenario['runname'] += '_Scenario-' + str(scenario_index)

    print(scenario['runname'])

    ###############
    # Make data generators
    ###############      
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
                                    scenario['shuffle_generators'],
                                    sample_weights=scenario['sample_weighting'])
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
                                  scenario['shuffle_generators'],
                                  sample_weights=scenario['sample_weighting'])
    print('Made Generators')


    ###############
    # Get model and optimizer
    ############### 

    models_dict = {'simple_lstm': get_model_simple_lstm,
              'conv2d': get_model_conv2d,
              'linear_systems': get_model_linear_systems,
              'conv1d':  build_lstmconv1d_joe,
              'simple_dense':  build_dumb_simple_model}

    optimizers_dict = {'sgd': keras.optimizers.SGD,
                  'rmsprop': keras.optimizers.RMSprop,
                  'adagrad': keras.optimizers.Adagrad,
                  'adadelta': keras.optimizers.Adadelta,
                  'adam': keras.optimizers.Adam,
                  'adamax': keras.optimizers.Adamax,
                  'nadam': keras.optimizers.Nadam}
    
    model = models_dict[scenario['model_type']](scenario['input_profile_names'],
                                               scenario['target_profile_names'],
                                               scenario['scalar_input_names'],
                                               scenario['actuator_names'],
                                               scenario['lookbacks'],
                                               scenario['lookahead'],
                                               scenario['profile_length'],
                                               scenario['std_activation'],
                                               **scenario['model_kwargs'])
    model.summary()
    if ngpu>1:
        parallel_model = keras.utils.multi_gpu_model(model, gpus=ngpu)

    optimizer = optimizers_dict[scenario['optimizer']](**scenario['optimizer_kwargs'])

    ###############
    # Get losses and metrics
    ############### 
    losses_dict = {'mse': keras.losses.mean_squared_error,
              'mae': keras.losses.mean_absolute_error,
              'normed_mse': helpers.custom_losses.normed_mse,
              'mean_diff_sum_2': helpers.custom_losses.mean_diff_sum_2,
              'max_diff_sum_2': helpers.custom_losses.max_diff_sum_2, 
              'mean_diff2_sum2': helpers.custom_losses.mean_diff2_sum2, 
              'max_diff2_sum2': helpers.custom_losses.max_diff2_sum2}
    
    if scenario['loss_function'] == 'hinge_mse':
        scenario['loss_function_kwargs']['mse_weight_vector'] = np.linspace(1, 
            scenario['loss_function_kwargs']['mse_weight_edge']**(1/scenario['loss_function_kwargs']['mse_weight_power']), 
                                                    scenario['profile_length'])**scenario['loss_function_kwargs']['mse_weight_power']
        loss = {}
        for sig in target_profile_names:
            loss.update({'target_'+sig: hinge_mse_loss(sig, model, scenario['loss_function_kwargs']['hinge_weight'],
                                                   scenario['loss_function_kwargs']['mse_weight_vector'], scenario['predict_deltas'])})
        
    else:
        loss = losses_dict[scenario['loss_function']]
    
    metrics = [keras.metrics.mean_squared_error, 
               helpers.custom_losses.normed_mse, 
               helpers.custom_losses.mean_diff_sum_2, 
               helpers.custom_losses.max_diff_sum_2, 
               helpers.custom_losses.mean_diff2_sum2, 
               helpers.custom_losses.max_diff2_sum2]
    

    callbacks = []
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                       verbose=1, mode='min', min_delta=0.001,
                                       cooldown=1, min_lr=0))
    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001, patience=8, 
                                   verbose=1, mode='min', restore_best_weights=True))
    if ngpu<=1:
        callbacks.append(ModelCheckpoint(checkpt_dir+scenario['runname']+'.h5', monitor='val_loss',
                                         verbose=0, save_best_only=True,
                                         save_weights_only=False, mode='auto', period=1))
  

    scenario['steps_per_epoch'] = len(train_generator)
    scenario['val_steps'] = len(val_generator)
    print('Train generator length: {}'.format(len(train_generator)))


    ###############
    # Save scenario
    ############### 
    with open(checkpt_dir + scenario['runname'] + '_params.pkl', 'wb+') as f:
        pickle.dump(copy.deepcopy(scenario), f)
    print('Saved Analysis params before run')


    ###############
    # Compile and Train
    ###############
    if ngpu>1:
        parallel_model.compile(optimizer, loss, metrics)
        print('Parallel model compiled, starting training')
        history = parallel_model.fit_generator(train_generator,
                                               steps_per_epoch=scenario['steps_per_epoch'],
                                               epochs=scenario['epochs'],
                                               callbacks=callbacks,
                                               validation_data=val_generator,
                                               validation_steps=scenario['val_steps'],
                                               verbose=verbose)
    else:
        model.compile(optimizer, loss, metrics)
        print('Model compiled, starting training')
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=scenario['steps_per_epoch'],
                                      epochs=scenario['epochs'],
                                      callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=scenario['val_steps'],
                                      verbose=verbose)


    ###############
    # Save Results
    ############### 
    scenario['model_path'] = checkpt_dir + scenario['runname'] + '.h5'
    scenario['history'] = history.history
    scenario['history_params'] = history.params
      
    if not any([isinstance(cb,ModelCheckpoint) for cb in callbacks]):
        model.save(scenario['model_path'])
    with open(checkpt_dir + scenario['runname'] + '_params.pkl', 'wb+') as f:
        pickle.dump(copy.deepcopy(scenario), f)
    print('Saved Analysis params after completion')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
