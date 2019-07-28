import pickle
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
from helpers.data_generator import process_data, DataGenerator, TensorBoardWrapper
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_correct_sign
from models.LSTMConv2D import get_model_LSTMConv2D, get_model_simple_LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


num_cores = 4
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': 1})
session = tf.Session(config=config)
K.set_session(session)

avail_profiles = ['dens', 'ffprime', 'idens', 'itemp', 'press', 'rotation',
                  'temp', 'thomson_dens', 'thomson_temp']
avail_actuators = ['curr', 'ech', 'gasA', 'gasB', 'gasC', 'gasD' 'gasE', 'pinj',
                   'pinj_15L', 'pinj_15R', 'pinj_21L', 'pinj_21R', 'pinj_30L',
                   'pinj_30R', 'pinj_33L', 'pinj_33R', 'tinj']
available_sigs = avail_profiles + avail_actuators + ['time']

runname = 'big_model_predict_full_hinge_loss_std_norm'
input_profile_names = ['temp', 'dens', 'rotation']
target_profile_names = ['temp']
actuator_names = ['pinj', 'curr']
predict_deltas = False
profile_lookback = 8
actuator_lookback = 8
lookahead = 3
profile_length = 65
final_profile_channels = 10
rawdata_path = '/home/fouriest/SCHOOL/Princeton/PPPL/final_data.pkl'
checkpt_dir = '/home/fouriest/SCHOOL/Princeton/PPPL/'
sig_names = input_profile_names + target_profile_names + actuator_names
normalization_method = 'StandardScaler'
window_length = 1
window_overlap = 0
sample_step = 5
uniform_normalization = True
train_frac = 0.8
val_frac = 0.2
nshots = 1000
mse_weight_vector = np.linspace(1, np.sqrt(10), profile_length)**2
hinge_weight = 50
batch_size = 128
epochs = 50
verbose = 1
assert(all(elem in available_sigs for elem in sig_names))

traindata, valdata, param_dict = process_data(rawdata_path, sig_names, normalization_method,
                                              window_length, window_overlap, profile_lookback,
                                              lookahead, sample_step, uniform_normalization,
                                              train_frac, val_frac, nshots)
train_generator = DataGenerator(traindata, batch_size, input_profile_names,
                                actuator_names, target_profile_names, profile_lookback,
                                actuator_lookback, lookahead, predict_deltas)
val_generator = DataGenerator(valdata, batch_size, input_profile_names,
                              actuator_names, target_profile_names, profile_lookback,
                              actuator_lookback, lookahead, predict_deltas)
steps_per_epoch = len(train_generator)
val_steps = len(val_generator)
model = get_model_LSTMConv2D(input_profile_names, target_profile_names,
                             actuator_names, profile_lookback, lookahead, profile_length,
                             final_profile_channels)
# model = get_model_simple_LSTM(input_profile_names, target_profile_names,
#                               actuator_names, profile_lookback, lookahead, profile_length)
model.summary()
optimizer = keras.optimizers.Nadam()
loss = []
loss.append(hinge_mse_loss(
    'temp', model, hinge_weight, mse_weight_vector, predict_deltas))
metrics = []
metrics.append(denorm_loss(
    param_dict['temp'], keras.metrics.MAE, predict_deltas, 'temp', model))
metrics.append(percent_correct_sign('temp', model, predict_deltas))
# loss = ['mse']
# metrics = ['mae']

callbacks = []
callbacks.append(ModelCheckpoint(checkpt_dir + runname + '.h5', monitor='val_loss', verbose=0,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                   verbose=1, mode='auto', min_delta=0.001,
                                   cooldown=1, min_lr=0))
callbacks.append(TensorBoardWrapper(val_generator, log_dir=checkpt_dir + 'tensorboard_logs/' + runname,
                                    histogram_freq=1, batch_size=batch_size, write_graph=True))

model.compile(optimizer, loss, metrics)
history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                              epochs=epochs, verbose=verbose, callbacks=callbacks,
                              validation_data=val_generator, validation_steps=val_steps,
                              max_queue_size=10, workers=4, use_multiprocessing=False)

analysis_params = {'rawdata': rawdata_path,
                   'input_profile_names': input_profile_names,
                   'actuator_names': actuator_names,
                   'target_profile_names': target_profile_names,
                   'predict_deltas': predict_deltas,
                   'sig_names': sig_names,
                   'window_length': window_length,
                   'window_overlap': window_overlap,
                   'profile_lookback': profile_lookback,
                   'actuator_lookback': actuator_lookback,
                   'lookahead': lookahead,
                   'sample_step': sample_step,
                   'model_path': checkpt_dir + runname + '.h5',
                   'normalization_params': param_dict,
                   'history': history}
with open(checkpt_dir + runname + '.pkl', 'wb+') as f:
    pickle.dump(analysis_params, f)
