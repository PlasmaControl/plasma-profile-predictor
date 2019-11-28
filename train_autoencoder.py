import pickle
import keras
import numpy as np
from helpers.data_generator import process_data, AutoEncoderDataGenerator
from helpers.hyperparam_helpers import make_bash_scripts
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
from helpers.custom_losses import percent_correct_sign, baseline_MAE
from helpers.results_processing import write_results_autoencoder
import models.autoencoder
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
    num_cores = 32
    ngpu = 1
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

    checkpt_dir = os.path.expanduser("~/run_results_11_19/")
    if not os.path.exists(checkpt_dir):
        os.makedirs(checkpt_dir)

    ###############
    # scenarios
    ###############

    efit_type = 'EFIT02'

    default_scenario = {'actuator_names': ['pinj', 'curr', 'tinj','gasA'],
                        'profile_names': ['thomson_temp_{}'.format(efit_type),
                                          'thomson_dens_{}'.format(efit_type),
                                          'ffprime_{}'.format(efit_type),
                                          'press_{}'.format(efit_type),
                                          'q_{}'.format(efit_type)],
                        'scalar_names': [],
                        'profile_downsample': 2,
                        'state_encoder_type': 'dense',
                        'state_decoder_type': 'dense',
                        'control_encoder_type': 'dense',
                        'control_decoder_type': 'dense',
                        'state_encoder_kwargs': {'num_layers': 6,
                                                 'layer_scale': 2,
                                                 'std_activation':'relu'},
                        'state_decoder_kwargs': {'num_layers': 6,
                                                 'layer_scale': 2,
                                                 'std_activation':'relu'},
                        'control_encoder_kwargs': {'num_layers': 10,
                                                   'layer_scale': 2,
                                                   'std_activation':'relu'},
                        'control_decoder_kwargs': {'num_layers': 10,
                                                   'layer_scale': 2,
                                                   'std_activation':'relu'},
                        'state_latent_dim':50,
                        'control_latent_dim':5,
                        'x_weight':1,
                        'u_weight':1,
                        'discount_factor':1,
                        'batch_size': 128,
                        'epochs': 100,
                        'flattop_only': True,
                        'raw_data_path': '/scratch/gpfs/jabbate/mixed_data/final_data.pkl',
                        'process_data': True,
                        'processed_filename_base': '/scratch/gpfs/jabbate/data_60_ms_randomized_',
                        'optimizer': 'adagrad',
                        'optimizer_kwargs': {},
                        'shuffle_generators': True,
                        'pruning_functions': ['remove_nan', 'remove_dudtrip', 'remove_I_coil'],
                        'normalization_method': 'RobustScaler',
                        'window_length': 1,
                        'window_overlap': 0,
                        'lookback': 0,
                        'lookahead': 3,
                        'sample_step': 1,
                        'uniform_normalization': True,
                        'train_frac': 0.8,
                        'val_frac': 0.2,
                        'nshots': 12000,
                        'excluded_shots': ['topology_TOP', 'topology_OUT', 'topology_MAR', 'topology_IN', 'topology_DN', 'topology_BOT']}

    scenarios_dict = OrderedDict()
    scenarios_dict['process_data'] = [{'process_data':True}]
    scenarios_dict['x_weight'] = [{'x_weight':0.5},
                                 {'x_weight':1},
                                 {'x_weight':2},
                                 {'x_weight':5}]
    scenarios_dict['u_weight'] = [{'u_weight':0.5},
                                 {'u_weight':1},
                                 {'u_weight':2},
                                 {'u_weight':5}]
    scenarios_dict['discount_factor'] = [{'discount_factor':0.5},
                                 {'discount_factor':0.9},
                                 {'discount_factor':0.8},
                                 {'discount_factor':0.7}]
    scenarios_dict['window_length'] = [{'window_length': 3}]


    scenarios = []
    runtimes = []
    for scenario in itertools.product(*list(scenarios_dict.values())):
        foo = {k: v for d in scenario for k, v in d.items()}
        scenarios.append(foo)
        runtimes.append(4*60)
    num_scenarios = len(scenarios)

    ###############
    # Batch Run
    ###############
    if scenario_index == -1:
        make_bash_scripts(num_scenarios, checkpt_dir,
                          num_cores, ngpu, runtimes, mode='autoencoder')
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
        verbose = 2
        print('Loading Scenario ' + str(scenario_index) + ':')
        scenario = scenarios[scenario_index]
    else:
        verbose = 1
        print('Loading Default Scenario:')
        scenario = default_scenario
    print(scenario)

    if scenario['process_data']:
        scenario.update(
            {k: v for k, v in default_scenario.items() if k not in scenario.keys()})
        scenario['sig_names'] = scenario['profile_names'] + \
            scenario['actuator_names'] + scenario['scalar_names']

        if 'raw_data_path' not in scenario.keys():
            scenario['raw_data_path'] = default_scenario['raw_data_path']
        traindata, valdata, normalization_dict = process_data(scenario['raw_data_path'],
                                                              scenario['sig_names'],
                                                              scenario['normalization_method'],
                                                              scenario['window_length'],
                                                              scenario['window_overlap'],
                                                              scenario['lookback'],
                                                              scenario['lookahead'],
                                                              scenario['sample_step'],
                                                              scenario['uniform_normalization'],
                                                              scenario['train_frac'],
                                                              scenario['val_frac'],
                                                              scenario['nshots'],
                                                              verbose,
                                                              scenario['flattop_only'],
                                                              pruning_functions=scenario['pruning_functions'],
                                                              excluded_shots=scenario['excluded_shots'])

        scenario['dt'] = np.mean(np.diff(traindata['time']))*scenario['window_length']/1000 # in seconds
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
        scenario.update({k: v for k, v in param_dict.items()
                         if k not in scenario.keys()})
        scenario.update(
            {k: v for k, v in default_scenario.items() if k not in scenario.keys()})

    scenario['profile_length'] = int(
        np.ceil(65/scenario['profile_downsample']))

    scenario['runname'] = 'model-autoencoder' + \
                          '_SET-' + scenario['state_encoder_type'] + \
                          '_SDT-' + scenario['state_decoder_type'] + \
                          '_CET-' + scenario['control_encoder_type'] + \
                          '_CDT-' + scenario['control_decoder_type'] + \
                          '_profiles-' + '-'.join(scenario['profile_names']) + \
                          '_act-' + '-'.join(scenario['actuator_names']) + \
                          '_LB-' + str(scenario['lookback']) + \
                          '_LA-' + str(scenario['lookahead']) +\
                          '_ftop-' + str(scenario['flattop_only']) + \
                          strftime("_%d%b%y-%H-%M", localtime())

    if scenario_index >= 0:
        scenario['runname'] += '_Scenario-' + str(scenario_index)

    print(scenario['runname'])

    ###############
    # Make data generators
    ###############
        
    train_generator = AutoEncoderDataGenerator(traindata,
                                               scenario['batch_size'],
                                               scenario['profile_names'],
                                               scenario['actuator_names'],
                                               scenario['scalar_names'],
                                               scenario['lookback'],
                                               scenario['lookahead'],
                                               scenario['profile_downsample'],
                                               scenario['state_latent_dim'],
                                               scenario['discount_factor'],
                                               scenario['x_weight'],
                                               scenario['u_weight'],                                            
                                               scenario['shuffle_generators'])
    val_generator = AutoEncoderDataGenerator(valdata,
                                             scenario['batch_size'],
                                             scenario['profile_names'],
                                             scenario['actuator_names'],
                                             scenario['scalar_names'],
                                             scenario['lookback'],
                                             scenario['lookahead'],
                                             scenario['profile_downsample'],
                                             scenario['state_latent_dim'],
                                             scenario['discount_factor'],
                                             scenario['x_weight'],
                                             scenario['u_weight'],
                                             scenario['shuffle_generators'])

    print('Made Generators')

    ###############
    # Get model and optimizer
    ###############
    optimizers = {'sgd': keras.optimizers.SGD,
                  'rmsprop': keras.optimizers.RMSprop,
                  'adagrad': keras.optimizers.Adagrad,
                  'adadelta': keras.optimizers.Adadelta,
                  'adam': keras.optimizers.Adam,
                  'adamax': keras.optimizers.Adamax,
                  'nadam': keras.optimizers.Nadam}

    model = models.autoencoder.make_autoencoder(scenario['state_encoder_type'],
                                                scenario['state_decoder_type'],
                                                scenario['control_encoder_type'],
                                                scenario['control_decoder_type'],
                                                scenario['state_encoder_kwargs'],
                                                scenario['state_decoder_kwargs'],
                                                scenario['control_encoder_kwargs'],
                                                scenario['control_decoder_kwargs'],
                                                scenario['profile_names'],
                                                scenario['scalar_names'],
                                                scenario['actuator_names'],
                                                scenario['state_latent_dim'],
                                                scenario['control_latent_dim'],
                                                scenario['profile_length'],
                                                scenario['lookback'],
                                                scenario['lookahead'])



    model.summary()
    if ngpu > 1:
        parallel_model = keras.utils.multi_gpu_model(model, gpus=ngpu)

    optimizer = optimizers[scenario['optimizer']](
        **scenario['optimizer_kwargs'])

    ###############
    # Get losses and metrics
    ###############

    loss = 'mse'
    metrics = ['mse','mae']
    callbacks = []
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                       verbose=1, mode='auto', min_delta=0.001,
                                       cooldown=1, min_lr=0))
    if ngpu <= 1:
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
    if ngpu > 1:
        parallel_model.compile(optimizer, loss, metrics,
                               sample_weight_mode="temporal")
        print('Parallel model compiled, starting training')
        history = parallel_model.fit_generator(train_generator,
                                               steps_per_epoch=scenario['steps_per_epoch'],
                                               epochs=scenario['epochs'],
                                               callbacks=callbacks,
                                               validation_data=val_generator,
                                               validation_steps=scenario['val_steps'],
                                               verbose=verbose)
    else:
        model.compile(optimizer, loss, metrics, sample_weight_mode="temporal")
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
    scenario['image_path'] = 'https://jabbate7.github.io/plasma-profile-predictor/results/' + scenario['runname']
    scenario['history'] = history.history
    scenario['history_params'] = history.params
    
    write_results_autoencoder(model,scenario)
    print('Wrote to google sheet')
    
    if not any([isinstance(cb, ModelCheckpoint) for cb in callbacks]):
        model.save(scenario['model_path'])
    with open(checkpt_dir + scenario['runname'] + '_params.pkl', 'wb+') as f:
        pickle.dump(copy.deepcopy(scenario), f)
    print('Saved Analysis params after completion')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()


