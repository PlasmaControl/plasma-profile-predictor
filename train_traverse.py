import pickle
import keras
import numpy as np

from helpers.data_generator import process_data, DataGenerator
from helpers.hyperparam_helpers import make_bash_scripts
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
from helpers.custom_losses import percent_correct_sign, baseline_MAE
from models.LSTMConv2D import get_model_lstm_conv2d, get_model_simple_lstm
from models.LSTMConv2D import get_model_linear_systems, get_model_conv2d
from models.LSTMConv1D import build_lstmconv1d_joe, build_dumb_simple_model
from utils.callbacks import CyclicLR, TensorBoardWrapper
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from time import strftime, localtime
import tensorflow as tf
from keras import backend as K
from collections import OrderedDict
import os
import sys
import itertools
import copy

def main(scenario_index=-2):

    ###################
    # set session
    ###################
    num_cores = 16
    ngpu = 0
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
    
    checkpt_dir = os.path.expanduser("~/run_results/")

    ###############
    # scenarios
    ###############


    default_scenario = {'actuator_names': ['pinj', 'curr', 'tinj', 'gasA'],
                        'input_profile_names': ['temp', 'dens'],
                        'target_profile_names': ['temp', 'dens'],
                        'scalar_input_names' : [],
                        'profile_downsample' : 2,
                        'model_type' : 'simple_dense',
                        'model_kwargs': {},
                        'std_activation' : 'relu',
                        'hinge_weight' : 50,
                        'mse_weight_power' : 2,
                        'mse_weight_edge' : np.sqrt(10),
                        'mse_power':2.5,
                        'batch_size' : 128,
                        'epochs' : 50,
                        'verbose' : 1,
                        'flattop_only': True,
                        'predict_deltas' : True,
                        'processed_filename_base': '/scratch/gpfs/jabbate/data_60_ms_randomized_',
                        'optimizer': 'adagrad',
                        'optimizer_kwargs': {},
                        'shuffle_generators': True}
   
    scenarios_dict = OrderedDict()
    scenarios_dict['models'] = [{'model_type': 'simple_dense', 'epochs': 50},
                                {'model_type': 'conv2d', 'epochs': 100}]
    scenarios_dict['actuators_scalars'] = [{'actuator_names': ['pinj', 'curr', 'tinj', 'gasA'],
                                            'scalar_input_names':[]},
                                           {'actuator_names': ['pinj', 'curr', 'tinj', 'gasA',
                                             'gasB', 'gasC', 'gasD'],
                                            'scalar_input_names':[]},
                                           {'actuator_names': ['pinj', 'curr', 'tinj',
                                             'target_density', 'gas_feedback'],
                                            'scalar_input_names':['density_estimate']}]
    scenarios_dict['flattop'] = [{'flattop_only': True},
                                {'flattop_only': False}]
    scenarios_dict['inputs'] =  [{'input_profile_names': ['temp','dens']},
                                {'input_profile_names': ['thomson_dens_EFITRT1',
                                                         'thomson_temp_EFITRT1']}]
    scenarios_dict['targets'] = [{'target_profile_names': ['temp','dens']}]
    scenarios_dict['batch_size'] = [{'batch_size': 128}]
    scenarios_dict['predict_deltas'] = [{'predict_deltas': True},
                                        {'predict_deltas': False}]


    scenarios = []
    runtimes = []
    for scenario in itertools.product(*list(scenarios_dict.values())):
        foo = {k: v for d in scenario for k, v in d.items()}
        scenarios.append(foo)
        if foo['model_type'] == 'conv2d':
            runtimes.append(5*128/foo['batch_size']*foo['epochs'])
        elif foo['model_type'] == 'simple_dense':
            runtimes.append(1*128/foo['batch_size']*foo['epochs'])
        elif foo['model_type'] == 'conv1d':
            runtimes.append(3.5*128/foo['batch_size']*foo['epochs'])
        else:
            runtimes.append(4*60)
    num_scenarios = len(scenarios)


    ###############
    # Batch Run
    ###############
    if scenario_index == -1:
        make_bash_scripts(num_scenarios, checkpt_dir, num_cores, ngpu, runtimes)
        print('Created Driver Scripts in ' + checkpt_dir)
        for i in range(4):
            os.system('sbatch {}'.format(os.path.join(
                checkpt_dir, 'driver' + str(i) + '.sh')))
        print('Jobs submitted, exiting')
        return


    ###############
    # Load Scenario and Data
    ###############    
    if scenario_index >= 0:
        print('Loading Scenario ' + str(scenario_index) + ':')
        scenario = scenarios[scenario_index]
    else:
        print('Loading Default Scenario:')
        scenario = default_scenario
    print(scenario)
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

    # update scenario with standard values for things that werent specified
    scenario.update({k:v for k,v in param_dict.items() if k not in scenario.keys()})
    scenario.update({k:v for k,v in default_scenario.items() if k not in scenario.keys()})

    scenario['profile_length'] = int(np.ceil(65/scenario['profile_downsample']))
    scenario['mse_weight_vector'] = np.linspace(
        1, scenario['mse_weight_edge'], scenario['profile_length'])**scenario['mse_weight_power']
          
    scenario['runname'] = 'model-' + scenario['model_type'] + \
              '_profiles-' + '-'.join(scenario['input_profile_names']) + \
              '_act-' + '-'.join(scenario['actuator_names']) + \
              '_targ-' + '-'.join(scenario['target_profile_names']) + \
              '_profLB-' + str(scenario['profile_lookback']) + \
              '_actLB-' + str(scenario['actuator_lookback']) +\
              '_norm-' + scenario['normalization_method'] + \
              '_activ-' + scenario['std_activation'] + \
              '_nshots-' + str(scenario['nshots']) + \
              '_ftop-' + str(scenario['flattop_only']) + \
              strftime("_%d%b%y-%H-%M", localtime())

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
                                    scenario['shuffle_generators'])
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
                                  scenario['shuffle_generators'])
    print('Made Generators')


    ###############
    # Get model and optimizer
    ############### 
    models = {'simple_lstm': get_model_simple_lstm,
              'lstm_conv2d': get_model_lstm_conv2d,
              'conv2d': get_model_conv2d,
              'linear_systems': get_model_linear_systems,
              'conv1d': build_lstmconv1d_joe,
              'simple_dense': build_dumb_simple_model}

    optimizers = {'sgd': keras.optimizers.SGD,
                  'rmsprop': keras.optimizers.RMSprop,
                  'adagrad': keras.optimizers.Adagrad,
                  'adadelta': keras.optimizers.Adadelta,
                  'adam': keras.optimizers.Adam,
                  'adamax': keras.optimizers.Adamax,
                  'nadam': keras.optimizers.Nadam}
    
    # with tf.device('/cpu:0'):
    model = models[scenario['model_type']](scenario['input_profile_names'],
                                           scenario['target_profile_names'],
                                           scenario['scalar_input_names'],
                                           scenario['actuator_names'],
                                           scenario['lookbacks'],
                                           scenario['lookahead'],
                                           scenario['profile_length'],
                                           scenario['std_activation'],
                                           **scenario['model_kwargs'])

    if ngpu>1:
        parallel_model = keras.utils.multi_gpu_model(model, gpus=ngpu)

    optimizer = optimizers[scenario['optimizer']](**scenario['optimizer_kwargs'])

    ###############
    # Get losses and metrics
    ############### 
    
    loss = {}
    metrics = {}
    for sig in scenario['target_profile_names']:
        loss.update({'target_'+sig: hinge_mse_loss(sig,
                                                   model,
                                                   scenario['hinge_weight'],
                                                   scenario['mse_weight_vector'],
                                                   scenario['mse_power'],
                                                   scenario['predict_deltas'])})
        metrics.update({'target_'+sig: []})
        metrics['target_'+sig].append(denorm_loss(sig,
                                                  model,
                                                  scenario['normalization_dict'][sig],
                                                  keras.metrics.MAE,
                                                  scenario['predict_deltas']))
        metrics['target_'+sig].append(percent_correct_sign(sig,
                                                           model,
                                                           scenario['predict_deltas']))
        metrics['target_' + sig].append(percent_baseline_error(sig,
                                                               model,
                                                               scenario['predict_deltas']))

    callbacks = []
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                       verbose=1, mode='auto', min_delta=0.001,
                                       cooldown=1, min_lr=0))
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
                                               verbose=scenario['verbose'])
    else:
        model.compile(optimizer, loss, metrics)
        print('Model compiled, starting training')
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=scenario['steps_per_epoch'],
                                      epochs=scenario['epochs'],
                                      callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=scenario['val_steps'],
                                      verbose=scenario['verbose'])


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
