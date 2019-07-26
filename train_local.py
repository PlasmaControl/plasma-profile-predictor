import keras
import numpy as np
from helpers.data_generator import process_data, DataGenerator
from helpers.custom_losses import denorm_loss, hinge_mse_loss
from models.LSTMConv2D import get_model_LSTMConv2D

available_sigs = ['curr', 'thomson_temp', 'pinj_30L', 'pinj_30R', 'pinj_15R', 'pinj', 'ffprime', 'tinj', 'pinj_21L', 'pinj_15L', 'pinj_33L', 'ech',
                  'pinj_33R', 'press', 'rotation', 'thomson_dens', 'pinj_21R', 'idens', 'temp', 'gasA', 'gasC', 'gasB', 'gasE', 'gasD', 'dens', 'time', 'itemp']

input_profile_names = ['temp', 'dens', 'rotation']
target_profile_names = ['temp']
actuator_names = ['pinj', 'curr']
predict_deltas = True
lookback = 8
lookahead = 3
profile_length = 65
final_profile_channels = 10
rawdata_path = '/home/fouriest/SCHOOL/Princeton/PPPL/final_data.pkl'
checkpt_filepath = '/home/fouriest/SCHOOL/Princeton/PPPL/model.h5'
sig_names = input_profile_names + target_profile_names + actuator_names
normalization_method = 'StandardScaler'
window_length = 1
window_overlap = 0
sample_step = 5
uniform_normalization = True
train_frac = 0.7
val_frac = 0.2
nshots = 300
mse_weight_vector = np.linspace(1, np.sqrt(10), profile_length)**2
hinge_weight = 50
batch_size = 128
epochs = 30
verbose = 1


traindata, valdata, param_dict = process_data(rawdata_path, sig_names, normalization_method,
                                              window_length, window_overlap, lookback,
                                              lookahead, sample_step, uniform_normalization,
                                              train_frac, val_frac, nshots)
train_generator = DataGenerator(traindata, batch_size, input_profile_names,
                                actuator_names, target_profile_names, lookback, predict_deltas)
val_generator = DataGenerator(valdata, batch_size, input_profile_names,
                              actuator_names, target_profile_names, lookback, predict_deltas)
steps_per_epoch = len(train_generator)
val_steps = len(val_generator)
model = get_model_LSTMConv2D(input_profile_names, target_profile_names,
                             actuator_names, lookback, lookahead, profile_length,
                             final_profile_channels)

optimizer = keras.optimizers.Nadam()
loss = {'target_temp': hinge_mse_loss(
    'temp', model, hinge_weight, mse_weight_vector, predict_deltas)}

metrics = {'target_temp': denorm_loss(param_dict['temp'], keras.metrics.MAE)}


checkpt = keras.callbacks.ModelCheckpoint(checkpt_filepath, monitor='val_mean_absolute_error',
                                          verbose=0, save_best_only=True,
                                          save_weights_only=False, mode='auto', period=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                              verbose=1, mode='auto', min_delta=0.001,
                                              cooldown=1, min_lr=0)

callbacks = [reduce_lr]

model.compile(optimizer, loss, metrics)
history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                              epochs=epochs, verbose=verbose, callbacks=callbacks,
                              validation_data=val_generator, validation_steps=val_steps,
                              max_queue_size=10, workers=4, use_multiprocessing=True)
