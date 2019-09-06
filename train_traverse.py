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


def main(scenario_index=-2):

    num_cores = 16
    ngpu = 0 
    config = tf.ConfigProto(intra_op_parallelism_threads=4*num_cores,
                            inter_op_parallelism_threads=4*num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': 1,
                                          'GPU': ngpu})
    session = tf.Session(config=config)
    K.set_session(session)
    
    scenarios_dict = OrderedDict()
    scenarios_dict['models']= [{'model_type': 'simple_dense', 'epochs': 50},
                               {'model_type': 'conv2d', 'epochs': 100}]
    scenarios_dict['actuators_scalars'] = [{'actuator_names':
                                            ['pinj', 'curr', 'tinj', 'gasA'],
                                            'scalar_input_names':[]},
                                           {'actuator_names':
                                            ['pinj', 'curr', 'tinj', 'gasA',
                                             'gasB', 'gasC', 'gasD'],
                                            'scalar_input_names':[]},
                                           {'actuator_names':
                                            ['pinj', 'curr', 'tinj',
                                             'target_density', 'gas_feedback'],
                                            'scalar_input_names':['density_estimate']}]
    scenarios_dict['flattop']= [{'flattop_only': True,
                                 'processed_filename_base':
                                 '/scratch/gpfs/jabbate/data_60_ms_flattop_randomized/'}, 
                                {'flattop_only': False,
                                 'processed_filename_base':
                                 '/scratch/gpfs/jabbate/data_60_ms_include_rampup_randomized/'}]
    scenarios_dict['inputs']=  [{'input_profile_names': ['temp','dens']},
                                {'input_profile_names': ['thomson_dens_EFITRT1',
                                                         'thomson_temp_EFITRT1']}]
    scenarios_dict['targets'] = [{'target_profile_names': ['temp','dens']}]
    scenarios_dict['profile_downsample'] =  [{'profile_downsample': 2}]
    scenarios_dict['std_activation'] = [{'std_activation': 'relu'}]
    scenarios_dict['hinge_weight'] = [{'hinge_weight': 50}]
    scenarios_dict['mse_weight_edge'] = [{'mse_weight_edge': np.sqrt(10)}]
    scenarios_dict['mse_weight_power'] = [{'mse_weight_power': 2}]
    scenarios_dict['batch_size'] = [{'batch_size': 128}]
    scenarios_dict['predict_deltas'] = [{'predict_deltas': True},
                                        {'predict_deltas': False}]

    checkpt_dir = os.path.expanduser("~/run_results/")


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
    if scenario_index == -1:

        make_bash_scripts(num_scenarios, checkpt_dir, num_cores, ngpu, runtimes)
        print('Created Driver Scripts in ' + checkpt_dir)
        '''
        for i in range(num_scenarios):
            os.system('sbatch {}'.format(os.path.join(
                checkpt_dir, 'driver' + str(i) + '.sh')))
        '''
        print('Jobs submitted, exiting')
        return

# data_60_ms/' #full_data_include_current_ramps'
    #processed_filename_base = '/scratch/gpfs/jabbate/data_60_ms/'

# with tf.device('/cpu:0'):
    with open(os.path.join(processed_filename_base, 'train.pkl'), 'rb') as f:
        traindata = pickle.load(f)
    with open(os.path.join(processed_filename_base, 'val.pkl'), 'rb') as f:
        valdata = pickle.load(f)

    with open(os.path.join(processed_filename_base, 'param_dict.pkl'), 'rb') as f:
        param_dict = pickle.load(f)
    globals().update(param_dict)
    print('Data Loaded \n')
    
    actuator_names = ['pinj', 'curr', 'tinj', 'gasA']
    input_profile_names = ['temp', 'dens']
    target_profile_names = ['temp', 'dens']
    scalar_input_names = []
    profile_downsample = 2
    mse_weight_power = 2
    mse_weight_edge = np.sqrt(10)
    model_type = 'simple_dense'
    predict_deltas = True
    std_activation = 'relu'
    hinge_weight = 50
    batch_size = 128
    epochs = 50
    verbose = 1

    profile_length = int(np.ceil(65/profile_downsample))
    mse_weight_vector = np.linspace(
        1, mse_weight_edge, profile_length)**mse_weight_power

    if scenario_index >= 0:
        globals().update(scenarios[scenario_index])

    models = {'simple_lstm': get_model_simple_lstm,
              'lstm_conv2d': get_model_lstm_conv2d,
              'conv2d': get_model_conv2d,
              'linear_systems': get_model_linear_systems,
              'conv1d': build_lstmconv1d_joe,
              'simple_dense': build_dumb_simple_model}

    runname = 'model-' + model_type + \
              '_profiles-' + '-'.join(input_profile_names) + \
              '_act-' + '-'.join(actuator_names) + \
              '_targ-' + '-'.join(target_profile_names) + \
              '_profLB-' + str(profile_lookback) + \
              '_actLB-' + str(actuator_lookback) +\
              '_norm-' + normalization_method + \
              '_activ-' + std_activation + \
              '_nshots-' + str(nshots) + \
              '_ftop-' + str(flattop_only) + \
              strftime("_%d%b%y-%H-%M", localtime())

    if scenario_index >= 0:
        runname += '_Scenario-' + str(scenario_index)

    print(runname)
        
    train_generator = DataGenerator(traindata, batch_size, input_profile_names,
                                    actuator_names, target_profile_names, scalar_input_names,
                                    lookbacks, lookahead,
                                    predict_deltas, profile_downsample)
    val_generator = DataGenerator(valdata, batch_size, input_profile_names,
                                  actuator_names, target_profile_names, scalar_input_names,
                                  lookbacks, lookahead,
                                  predict_deltas, profile_downsample)
    print('Made Generators \n')
    model_kwargs = {}

    # with tf.device('/cpu:0'):
    model = models[model_type](input_profile_names, target_profile_names,
                               scalar_input_names, actuator_names, lookbacks,
                               lookahead, profile_length, std_activation, **model_kwargs)

    if ngpu>1:
        parallel_model = keras.utils.multi_gpu_model(model, gpus=2)

    optimizer = keras.optimizers.Adagrad()

    loss = {}
    metrics = {}
    for sig in target_profile_names:
        loss.update({'target_'+sig: hinge_mse_loss(sig, model, hinge_weight,
                                                   mse_weight_vector, predict_deltas)})
        metrics.update({'target_'+sig: []})
        metrics['target_'+sig].append(denorm_loss(sig, model, normalization_dict[sig],
                                                  keras.metrics.MAE, predict_deltas))
        metrics['target_'+sig].append(percent_correct_sign(sig, model,
                                                           predict_deltas))
        metrics['target_' +
                sig].append(percent_baseline_error(sig, model, predict_deltas))

    callbacks = []

    if ngpu<=1:
        callbacks.append(ModelCheckpoint(checkpt_dir+runname+'.h5', monitor='val_loss',
                                         verbose=0, save_best_only=True,
                                         save_weights_only=False, mode='auto', period=1))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                       verbose=1, mode='auto', min_delta=0.001,
                                       cooldown=1, min_lr=0))

    steps_per_epoch = len(train_generator)
    print('Train generator length: {}'.format(len(train_generator)))
    val_steps = len(val_generator)

    if ngpu>1:
        parallel_model.compile(optimizer, loss, metrics)
        print('Model Compiled \n')
        history = parallel_model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                               epochs=epochs, callbacks=callbacks,
                                               validation_data=val_generator, validation_steps=val_steps, verbose=1)  # ,
    else:
        print('Model Compiled \n')
        model.compile(optimizer, loss, metrics)
        history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                      epochs=epochs, callbacks=callbacks,
                                      validation_data=val_generator, validation_steps=val_steps, verbose=1)  # ,

    analysis_params = {'rawdata': rawdata_path,
                       'flattop_only': flattop_only,
                       'model_type': model_type,
                       'input_profile_names': input_profile_names,
                       'actuator_names': actuator_names,
                       'target_profile_names': target_profile_names,
                       'scalar_input_names': scalar_input_names,
                       'sig_names': sig_names,
                       'predict_deltas': predict_deltas,
                       'profile_lookback': profile_lookback,
                       'actuator_lookback': actuator_lookback,
                       'lookbacks': lookbacks,
                       'lookahead': lookahead,
                       'profile_length': profile_length,
                       'profile_downsample': profile_downsample,
                       'std_activation': std_activation,
                       'window_length': window_length,
                       'window_overlap': window_overlap,
                       'sample_step': sample_step,
                       'normalization_method': normalization_method,
                       'uniform_normalization': uniform_normalization,
                       'normalization_params': normalization_dict,
                       'train_frac': train_frac,
                       'val_frac': val_frac,
                       'nshots': nshots,
                       'mse_weight_vector': mse_weight_vector,
                       'mse_weight_edge': mse_weight_edge,
                       'mse_weight_power': mse_weight_power,
                       'hinge_weight': hinge_weight,
                       'batch_size': batch_size,
                       'epochs': epochs,
                       'runname': runname,
                       'model_path': checkpt_dir + runname + '.h5',
                       'history': history.history,
                       'history_params': history.params}

    with open(checkpt_dir + runname + '_params.pkl', 'wb+') as f:
        pickle.dump(analysis_params, f)
    print('Saved Analysis params')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
