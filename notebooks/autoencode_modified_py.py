import pickle
import keras
import numpy as np
import random
import os
import sys
import itertools
import copy
from collections import OrderedDict
from time import strftime, localtime

from helpers.data_generator import process_data, AutoEncoderDataGenerator
from helpers.hyperparam_helpers import make_bash_scripts
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
from helpers.custom_losses import percent_correct_sign, baseline_MAE
from helpers.results_processing import write_autoencoder_results
import models.autoencoder
from helpers.callbacks import CyclicLR, TensorBoardWrapper, TimingCallback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras import backend as K

#------------------------------------------------------------------------------------------------------------------------


###################
# set session
###################
num_cores = 8
req_mem = 48 # gb
ngpu = 1

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

config = tf.ConfigProto(intra_op_parallelism_threads=4*num_cores,
                        inter_op_parallelism_threads=4*num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': ngpu})
session = tf.Session(config=config)
K.set_session(session)


#-----------------------------------------------------------------------------------------------------------------------


###############
# global stuff
###############

# Set Directory

checkpt_dir = os.path.expanduser("~/run_results_10_20_test/")
if not os.path.exists(checkpt_dir):
    os.makedirs(checkpt_dir)


#------------------------------------------------------------------------------------------------------------------------


###############
# scenarios
###############

efit_type = 'EFIT02'

default_scenario = {'actuator_names': ['pinj', 'curr', 'tinj','gasA'],
                    'profile_names': ['temp',
                                      'dens',
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
                    'batch_size': 1,
                    'epochs': 3,
                    'flattop_only': True,
                    'raw_data_path': '/scratch/gpfs/jabbate/small_data.pkl', #'/projects/EKOLEMEN/profile_predictor/DATA/full_data_with_error/small_data.pkl',#'/scratch/gpfs/jabbate/small_data.pkl', #/scratch/gpfs/jabbate/mixed_data/final_data.pkl #'/projects/EKOLEMEN/profile_predictor/DATA/full_data_with_error/small_data.pkl',
                    'process_data': True,
                    'optimizer': 'adagrad',
                    'optimizer_kwargs': {},
                    'shuffle_generators': True,
                    'pruning_functions': ['remove_nan', 'remove_dudtrip', 'remove_I_coil'],
                    'normalization_method': 'RobustScaler',
                    'window_length': 3,
                    'window_overlap': 0,
                    'lookback': 0,
                    'lookahead': 3,
                    'sample_step': 1,
                    'uniform_normalization': True,
                    'train_frac': 0.8,
                    'val_frac': 0.2,
                    'nshots': 12000,
                    #'excluded_shots': ['topology_TOP', 'topology_OUT', 'topology_MAR', 'topology_IN', 'topology_DN', 'topology_BOT']
                   'excluded_shots': ['topology_TOP', 'topology_OUT', 'topology_MAR', 'topology_BOT']} # Including DN and IN

scenarios_dict = OrderedDict()
scenarios_dict['x_weight'] = [{'x_weight':0.1},
                              {'x_weight':1}]
scenarios_dict['u_weight'] = [{'u_weight':1},
                              {'u_weight':10}]
scenarios_dict['discount_factor'] = [{'discount_factor':1.0},
                                     {'discount_factor':0.8}]
scenarios_dict['state_latent_dim'] = [{'state_latent_dim': 20},
                                      {'state_latent_dim': 50},
                                      {'state_latent_dim': 100}]
scenarios_dict['control_latent_dim'] = [{'control_latent_dim': 5},
                                        {'control_latent_dim': 10},
                                        {'control_latent_dim': 15}]
scenarios_dict['state_encoder_kwargs'] = [{'state_encoder_kwargs': {'num_layers': 6,
                                             'layer_scale': 2,
                                             'std_activation':'elu'},
                                          'state_decoder_kwargs': {'num_layers': 6,
                                             'layer_scale': 2,
                                             'std_activation':'elu'}},
                                         {'state_encoder_kwargs': {'num_layers': 10,
                                             'layer_scale': 1,
                                             'std_activation':'elu'},
                                          'state_decoder_kwargs': {'num_layers': 10,
                                             'layer_scale': 1,
                                             'std_activation':'elu'}}]
scenarios_dict['control_encoder_kwargs'] = [{'control_encoder_kwargs': {'num_layers': 6,
                                             'layer_scale': 2,
                                             'std_activation':'elu'},
                                          'control_decoder_kwargs': {'num_layers': 6,
                                             'layer_scale': 2,
                                             'std_activation':'elu'}},
                                         {'control_encoder_kwargs': {'num_layers': 10,
                                             'layer_scale': 1,
                                             'std_activation':'elu'},
                                          'control_decoder_kwargs': {'num_layers': 10,
                                             'layer_scale': 1,
                                             'std_activation':'elu'}}]


# Compile Scenarios
scenarios = []
runtimes = []
for scenario in itertools.product(*list(scenarios_dict.values())):
    foo = {k: v for d in scenario for k, v in d.items()}
    scenarios.append(foo)
    runtimes.append(6*60)
num_scenarios = len(scenarios)

#-------------------------------------------------------------------------------------------------------

###############
# Batch Run
###############
scenario_index = 1

if scenario_index == -1:
    make_bash_scripts(num_scenarios, checkpt_dir,
                      num_cores, ngpu, req_mem, runtimes, mode='autoencoder')
    print('Created Driver Scripts in ' + checkpt_dir)
    for i in range(num_scenarios):
        os.system('sbatch {}'.format(os.path.join(
            checkpt_dir, 'driver' + str(i) + '.sh')))
    print('Jobs submitted, exiting')

###############
# Load Scenario and Data
###############
if scenario_index >= 0:
    verbose = 2
    print('Loading Scenario ' + str(scenario_index) + ':')
    scenario = scenarios[scenario_index]
    scenario.update(
        {k: v for k, v in default_scenario.items() if k not in scenario.keys()})
else:
    verbose = 1
    print('Loading Default Scenario:')
    scenario = default_scenario
    
for k,v in scenario.items():
    print('{}:{}'.format(k,v))
    
    
#---------------------------------------------------------------------------------------------------------------------------

if scenario['process_data']:
    scenario['sig_names'] = scenario['profile_names'] + \
        scenario['actuator_names'] + scenario['scalar_names']

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
    scenario['dt'] = np.mean(np.diff(traindata['time']))/1000 # in seconds
    scenario['normalization_dict'] = normalization_dict
    
    
#------------------------------------------------------------------------------------------------------------------------

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
                      strftime("_%d%b%y-%H-%M", localtime())

if scenario_index >= 0:
    scenario['runname'] += '_Scenario-' + str(scenario_index)

print(scenario['runname'])

#-------------------------------------------------------------------------------------------------------------------------


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


#---------------------------------------------------------------------------------------------------------------------------


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

#---------------------------------------------------------------------------------------------------------------

if ngpu > 1:
    parallel_model = keras.utils.multi_gpu_model(model, gpus=ngpu)

optimizer = optimizers[scenario['optimizer']](
    **scenario['optimizer_kwargs'])

#----------------------------------------------------------------------------------------------------------------

###############
# Get losses and metrics
###############

#loss = 'mse'
loss = tf.keras.losses.Huber(
    delta=0.7, reduction=losses_utils.ReductionV2.AUTO, name='huber_loss'
)
metrics = ['mse','mae']

#----------------------------------------------------------------------------------------------------------------

callbacks = []

callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                   factor=0.5, 
                                   patience=5,
                                   verbose=1, 
                                   mode='auto', 
                                   min_delta=0.001,
                                   cooldown=1, 
                                   min_lr=0))

callbacks.append(EarlyStopping(monitor='val_loss', 
                               min_delta=0, 
                               patience=10, 
                               verbose=1, 
                               mode='min'))

callbacks.append(TimingCallback(time_limit=(runtimes[scenario_index]-30)*60))

if ngpu <= 1:
    callbacks.append(ModelCheckpoint(checkpt_dir+scenario['runname']+'.h5', monitor='val_loss',
                                     verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1))

scenario['steps_per_epoch'] = len(train_generator)
scenario['val_steps'] = len(val_generator)
print('Train generator length: {}'.format(len(train_generator)))



#-----------------------------------------------------------------------------------------------------------------

###############
# Save scenario
###############
with open(checkpt_dir + scenario['runname'] + '_params.pkl', 'wb+') as f:
    pickle.dump(copy.deepcopy(scenario), f)
print('Saved Analysis params before run')


#-----------------------------------------------------------------------------------------------------------------

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
                                           verbose=1)
else:
    model.compile(optimizer, loss, metrics, sample_weight_mode="temporal")
    print('Model compiled, starting training')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=scenario['steps_per_epoch'],
                                  epochs=scenario['epochs'],
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=scenario['val_steps'],
                                  verbose=1)
    
#-----------------------------------------------------------------------------------------------------------------

 ###############
# Save Results
###############
scenario['model_path'] = checkpt_dir + scenario['runname'] + '.h5'
scenario['history'] = history.history
scenario['history_params'] = history.params

write_autoencoder_results(model, scenario)

if not any([isinstance(cb, ModelCheckpoint) for cb in callbacks]):
    model.save(scenario['model_path'])
    print('Saved model after training')
with open(checkpt_dir + scenario['runname'] + '_params.pkl', 'wb+') as f:
    pickle.dump(copy.deepcopy(scenario), f)
print('Saved Analysis params after training')


#-------------------------------------------------------------------------------------------------------------------






