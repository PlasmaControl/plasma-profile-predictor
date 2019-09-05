import pickle
import keras
import numpy as np

from helpers.data_generator import process_data, DataGenerator
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
from helpers.custom_losses import percent_correct_sign, baseline_MAE
from models.LSTMConv2D import get_model_lstm_conv2d, get_model_simple_lstm
from models.LSTMConv2D import get_model_linear_systems, get_model_conv2d
from models.LSTMConv1D import build_lstmconv1d_joe
from utils.callbacks import CyclicLR, TensorBoardWrapper
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import os

distributed=False

processed_filename_base='/scratch/gpfs/jabbate/data_60_ms/'#data_60_ms/' #full_data_include_current_ramps' 

#with tf.device('/cpu:0'):
with open(os.path.join(processed_filename_base,'train.pkl'),'rb') as f:
    traindata=pickle.load(f)
with open(os.path.join(processed_filename_base,'val.pkl'),'rb') as f:
    valdata=pickle.load(f)

with open(os.path.join(processed_filename_base,'param_dict.pkl'),'rb') as f:
    param_dict=pickle.load(f)
globals().update(param_dict)

############# CAN REDUCE THE SIGNALS HERE #####################

actuator_names=['pinj', 'curr', 'tinj', 'gasA']
#input_profile_names=['temp','dens']
input_profile_names=['thomson_temp_EFITRT1','thomson_dens_EFITRT1']
target_profile_names=['temp','dens']

###############################################################

profile_downsample = 2
profile_length = int(np.ceil(65/profile_downsample))
mse_weight_vector = np.linspace(1, np.sqrt(10), profile_length)**2

models = {'simple_lstm': get_model_simple_lstm,
          'lstm_conv2d': get_model_lstm_conv2d,
          'conv2d': get_model_conv2d,
          'linear_systems': get_model_linear_systems,
          'conv1d' : build_lstmconv1d_joe}

#model_type = 'conv1d'
model_type = 'conv2d'

predict_deltas = True

std_activation = 'relu'
checkpt_dir = "/home/jabbate/test_all_gas_vs_gasA_and_zipfit_vs_regular/" #"/global/homes/a/abbatej/plasma-profile-predictor/"

hinge_weight = 50
batch_size = 512
epochs = 50
verbose = 1

#runname = 'joe_zipfit_gasA'
#runname = 'joe_thomson_gasA'
#runname = 'rory_zipfit_gasA'
runname = 'rory_thomson_gasA'

train_generator = DataGenerator(traindata, batch_size, input_profile_names,
                                actuator_names, target_profile_names,
                                lookbacks, lookahead,
                                predict_deltas, profile_downsample)
val_generator = DataGenerator(valdata, batch_size, input_profile_names,
                              actuator_names, target_profile_names,
                              lookbacks, lookahead,
                              predict_deltas, profile_downsample)

#with tf.device('/cpu:0'):
model = models[model_type](input_profile_names, target_profile_names,
                           actuator_names, profile_lookback, actuator_lookback,
                           lookahead, profile_length, std_activation)

if distributed:
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

if not distributed:
    callbacks.append(ModelCheckpoint(checkpt_dir+runname+'.h5', monitor='val_loss',
                                     verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                   verbose=1, mode='auto', min_delta=0.001,
                                   cooldown=1, min_lr=0))

steps_per_epoch=len(train_generator)
print('Train generator length: {}'.format(len(train_generator)))
val_steps= len(val_generator)
# history = parallel_model.fit(x=np.array(list(train_generator)), 
#                              epochs=epochs, verbose=2, callbacks=callbacks,
#                              validation_data=val_generator, validation_steps=val_steps)#,

if distributed:
    parallel_model.compile(optimizer, loss, metrics)
    history = parallel_model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                           epochs=epochs, callbacks=callbacks,
                                           validation_data=val_generator, validation_steps=val_steps, verbose=1)#,
#max_queue_size=10, workers=4)
else:
    model.compile(optimizer, loss, metrics)
    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                           epochs=epochs, callbacks=callbacks,
                                           validation_data=val_generator, validation_steps=val_steps, verbose=1)#,

analysis_params = {'rawdata': rawdata_path,
                   'flattop_only': flattop_only,
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
                   'normalization_params': normalization_dict,
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
