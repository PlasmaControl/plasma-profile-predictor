import pickle
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
#from utils.multiGPU import ModelMGPU

from helpers.data_generator import process_data, DataGenerator
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
from helpers.custom_losses import percent_correct_sign, baseline_MAE
from models.LSTMConv2D import get_model_lstm_conv2d, get_model_simple_lstm
from models.LSTMConv2D import get_model_linear_systems, get_model_conv2d
from models.LSTMConv1D import build_lstmconv1d_joe
from utils.callbacks import CyclicLR, TensorBoardWrapper
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from time import strftime, localtime
import os

num_cores = 6
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': 4})
session = tf.Session(config=config)
K.set_session(session)

num_cores = 4
ngpu = 1

distributed=False

rank, n_ranks = 0,1

profile_downsample = 2
    
'''
efit_type='EFITRT1'
input_profile_names = ['thomson_dens_{}'.format(efit_type), 'thomson_temp_{}'.format(efit_type)]
target_profile_names = ['temp', 'dens']
actuator_names = ['pinj', 'curr', 'tinj', 'gasA']
profile_lookback = 1
actuator_lookback = 10
'''

if True:
    processed_filename_base='/global/cscratch1/sd/abbatej/processed_data/'

    with open(os.path.join(processed_filename_base,'train_{}.pkl'.format(75)),'rb') as f:
        traindata=pickle.load(f)
    with open(os.path.join(processed_filename_base,'val_{}.pkl'.format(75)),'rb') as f:
        valdata=pickle.load(f)
    with open(os.path.join(processed_filename_base,'param_dict.pkl'),'rb') as f:
        param_dict=pickle.load(f)
    globals().update(param_dict)
    
else:

    #for efit_type in ['EFITRT1','EFITRT2','EFIT01','EFIT02']:
    #for cer_type in ['cerreal','cerquick','cerauto']:
    avail_profiles = ['ffprime_{}'.format(efit_type), 'press_{}'.format(efit_type), 'ffprime_{}'.format(efit_type), 'pprime_{}'.format(efit_type),
                      'dens', 'temp', 'idens', 'itemp', 'rotation',
                     'thomson_dens_{}'.format(efit_type), 'thomson_temp_{}'.format(efit_type)]
    avail_actuators = ['curr', 'ech', 'gasA', 'gasB', 'gasC', 'gasD' 'gasE', 'pinj',
                       'pinj_15L', 'pinj_15R', 'pinj_21L', 'pinj_21R', 'pinj_30L',
                       'pinj_30R', 'pinj_33L', 'pinj_33R', 'tinj']
    available_sigs = avail_profiles + avail_actuators + ['time']

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
    lookahead = 8
    # rawdata_path = '/home/fouriest/SCHOOL/Princeton/PPPL/final_data.pkl'
    #rawdata_path = '/Users/alex/Desktop/ML/final_data_compressed.pkl'
    rawdata_path = '/global/cscratch1/sd/abbatej/processed_shots_big/final_data_batch_{}.pkl'.format(rank)
    flattop_only=True

    # checkpt_dir = '/home/fouriest/SCHOOL/Princeton/PPPL/'

    sig_names = input_profile_names + target_profile_names + actuator_names
    normalization_method = 'StandardScaler'
    window_length = 1
    window_overlap = 0
    sample_step = 3
    uniform_normalization = True
    train_frac = 0.8
    val_frac = 0.2
    nshots = 50 #TODO: replace with nbatches

    assert(all(elem in available_sigs for elem in sig_names))

    traindata, valdata, param_dict = process_data(rawdata_path, sig_names,
                                              normalization_method, window_length,
                                              window_overlap, lookbacks,
                                              lookahead, sample_step,
                                              uniform_normalization, train_frac,
                                              val_frac, nshots,
                                              flattop_only=flattop_only)
    normalization_dict=param_dict['normalization_dict']

profile_length = int(np.ceil(65/profile_downsample))
mse_weight_vector = np.linspace(1, np.sqrt(10), profile_length)**2

models = {'simple_lstm': get_model_simple_lstm,
          'lstm_conv2d': get_model_lstm_conv2d,
          'conv2d': get_model_conv2d,
          'linear_systems': get_model_linear_systems,
          'conv1d' : build_lstmconv1d_joe}

model_type = 'conv1d'

# config = tf.ConfigProto(intra_op_parallelism_cores,
#                         inter_op_parallelism_threads=num_cores,
#                         allow_soft_placement=True,
#                         device_count={'CPU': 1,
#                                       'GPU': ngpu})
# session = tf.Session(config=config)
# K.set_session(session)

models = {'simple_lstm': get_model_simple_lstm,
          'lstm_conv2d': get_model_lstm_conv2d,
          'conv2d': get_model_conv2d,
          'linear_systems': get_model_linear_systems,
          'conv1d' : build_lstmconv1d_joe}
predict_deltas = True

std_activation = 'relu'
checkpt_dir = "/home/jabbate/run_results/" #"/global/homes/a/abbatej/plasma-profile-predictor/"

hinge_weight = 50
# batch_size = 128*ngpu
batch_size = 128
epochs = 5

verbose = 1

runname = 'model-' + model_type + \
          '_profiles-' + '-'.join(input_profile_names) + \
          '_act-' + '-'.join(actuator_names) + \
          '_targ-' + '-'.join(target_profile_names) + \
          '_profLB-' + str(profile_lookback) + \
          '_actLB-' + str(actuator_lookback) +\
          '_norm-' + normalization_method + \
          '_activ-' + std_activation + \
          '_nshots-' + str(nshots) + \
          strftime("_%d%b%y-%H-%M", localtime())

train_generator = DataGenerator(traindata, batch_size, input_profile_names,
                                actuator_names, target_profile_names,
                                lookbacks, lookahead,
                                predict_deltas, profile_downsample)
val_generator = DataGenerator(valdata, batch_size, input_profile_names,
                              actuator_names, target_profile_names,
                              lookbacks, lookahead,
                              predict_deltas, profile_downsample)

steps_per_epoch = 5203 #10406 #len(train_generator) 
val_steps = 1300 #2601 #len(val_generator)
# model = models[model_type](input_profile_names, target_profile_names,
#                            actuator_names, profile_lookback, actuator_lookback,
#                            lookahead, profile_length, std_activation)
model = models[model_type](input_profile_names, target_profile_names,
                           actuator_names, profile_lookback, actuator_lookback,
                           lookahead, profile_length, std_activation)

model.summary()

#model = ModelMGPU(model, 4)
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

    # callbacks.append(TensorBoardWrapper(val_generator, log_dir=checkpt_dir +
    #                                     'tensorboard_logs/'+runname, histogram_freq=1,
    #                                     batch_size=batch_size, write_graph=True, write_grads=True))
    
# callbacks.append(CyclicLR(base_lr=0.0005, max_lr=0.006,
#                           step_size=4*steps_per_epoch, mode='triangular2'))
callbacks.append(ModelCheckpoint(checkpt_dir+runname+'.h5', monitor='val_loss',
                                 verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                   verbose=1, mode='auto', min_delta=0.001,
                                   cooldown=1, min_lr=0))
callbacks.append(TensorBoardWrapper(val_generator, log_dir=checkpt_dir +
                                    'tensorboard_logs/'+runname, histogram_freq=1,
                                    batch_size=batch_size, write_graph=True, write_grads=True))


model.compile(optimizer, loss, metrics)
history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                              epochs=epochs, verbose=2, callbacks=callbacks,
                              validation_data=val_generator, validation_steps=val_steps,
                              max_queue_size=10, workers=4)

analysis_params = {'rawdata': rawdata_path,
                   'model_type': model_type,
                   'input_profile_names': input_profile_names,
                   'actuator_names': actuator_names,
                   'target_profile_names': target_profile_names,
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
                   'normalization_params': param_dict,
                   'train_frac': train_frac,
                   'val_frac': val_frac,
                   'nshots': nshots,
                   'mse_weight_vector': mse_weight_vector,
                   'hinge_weight': hinge_weight,
                   'batch_size': batch_size,
                   'epochs': epochs,
                   'runname': runname,
                   'model_path': checkpt_dir + runname + '.h5',
                   'history': history.history,
                   'history_params': history.params}

with open(checkpt_dir + runname + 'params.pkl', 'wb+') as f:
    pickle.dump(analysis_params, f)

