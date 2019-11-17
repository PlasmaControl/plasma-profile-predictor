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


state_latent_dim = 25
control_latent_dim = 10
std_activation = 'relu'
state_num_layers = 6
control_num_layers = 10
profile_names = ['Te', 'Ne', 'P', 'q', 'ffprime']
scalar_names = ['kappa', 'tritop', 'tribot', 'drsep']
actuator_names = ['pinj', 'curr', 'tinj', 'gasA']
num_profiles = len(profile_names)
num_scalars = len(scalar_names)
num_actuators = len(actuator_names)
lookback = 0
lookahead = 1
profile_length = 33
layer_scale = 2

num_profiles = len(profile_names)
num_scalars = len(scalar_names)
num_actuators = len(actuator_names)
state_dim = num_profiles*profile_length + num_scalars*lookback

state_encoder = get_state_encoder_dense(profile_names, scalar_names, lookback,
                                        profile_length, state_latent_dim,
                                        std_activation, state_num_layers, layer_scale=layer_scale)
state_decoder = get_state_decoder_dense(profile_names, scalar_names, lookback,
                                        profile_length, state_latent_dim,
                                        std_activation, state_num_layers, layer_scale=layer_scale)
control_encoder = get_control_encoder_dense(actuator_names, control_latent_dim,
                                            std_activation, control_num_layers, layer_scale=layer_scale)
control_decoder = get_control_decoder_dense(actuator_names, control_latent_dim,
                                            std_activation, control_num_layers, layer_scale=layer_scale)
state_joiner, state_splitter = get_state_splitter_joiner(profile_names, scalar_names,
                                                         lookback, lookahead+1, profile_length)
control_joiner, control_splitter = get_control_splitter_joiner(
    actuator_names, lookback+lookahead)
model = make_autoencoder(state_encoder, state_decoder, control_encoder, control_decoder,
                         state_joiner, state_splitter, control_joiner, control_splitter,
                         profile_names, scalar_names, actuator_names, state_latent_dim,
                         control_latent_dim, lookback, lookahead, regularization=None)
