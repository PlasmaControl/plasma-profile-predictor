from keras.layers import Input, Dense, LSTM, Conv1D, Conv2D, ConvLSTM2D
from keras.layers import Dot, Add, Multiply, Concatenate, Reshape, Permute
from keras.models import Model
import numpy as np


def get_model_LSTMConv2D(input_profile_names, target_profile_names,
                         actuator_names, lookback, lookahead, profile_length,
                         final_profile_channels):

    profile_inshape = (lookback, profile_length)
    actuator_inshape = (lookback + lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)

    profile_inputs = []
    profiles = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input(profile_inshape, name='input_' + input_profile_names[i]))
        profiles.append(Reshape((lookback, profile_length, 1))
                        (profile_inputs[i]))
        profiles[i] = Dense(units=5, activation='relu')(profiles[i])
        profiles[i] = Conv2D(filters=5, kernel_size=(1, 3), strides=(1, 1), padding='same',
                             activation='relu')(profiles[i])
        profiles[i] = Dense(units=7, activation='relu')(profiles[i])
        profiles[i] = Conv2D(filters=10, kernel_size=(1, 5), strides=(1, 1), padding='same',
                             activation='relu')(profiles[i])
        profiles[i] = Dense(units=15, activation='relu')(profiles[i])
        profiles[i] = Conv2D(filters=20, kernel_size=(1, 7), strides=(1, 1), padding='same',
                             activation='relu')(profiles[i])
        profiles[i] = Reshape((lookback, profile_length, 1, 20))(profiles[i])
        profiles[i] = ConvLSTM2D(filters=final_profile_channels, kernel_size=(1, 5),
                                 strides=(1, 1), padding='same', activation='relu',
                                 recurrent_activation='hard_sigmoid',
                                 return_sequences=True)(profiles[i])
        profiles[i] = Reshape(
            (lookback, profile_length, final_profile_channels))(profiles[i])
        # shape = (5, 32, 10)

    merged = [[] for i in range(num_profiles)]
    for i in range(num_profiles):
        for j in range(num_profiles):
            merged[i].append(Dense(units=10, activation='relu')(profiles[i]))
        merged[i] = Add()(merged[i])
        # shape = (5, 32, 10)
    actuator_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_inputs.append(
            Input(actuator_inshape, name='input_' + actuator_names[i]))
        actuators.append(Reshape((lookback+lookahead, 1))(actuator_inputs[i]))
        actuators[i] = Dense(units=5, activation='relu')(actuators[i])
        actuators[i] = Conv1D(filters=7, kernel_size=3, strides=1, padding='causal',
                              activation='relu')(actuators[i])
        actuators[i] = LSTM(units=10, activation='relu', recurrent_activation='hard_sigmoid',
                            return_sequences=True)(actuators[i])
        actuators[i] = Reshape((1, lookback+lookahead, 10))(actuators[i])
        # shape = (1, 8, channels)

    actuators = Concatenate(axis=1)(actuators)
    # shape = (num_actuators, lookback+lookahead, 10)
    prof_act = []
    for i in range(num_profiles):
        prof_act.append(Dense(units=final_profile_channels,
                              activation='relu')(actuators))
        prof_act[i] = Conv2D(filters=final_profile_channels,
                             kernel_size=(num_actuators, lookahead+1), strides=(1, 1),
                             padding='valid', activation='relu')(prof_act[i])
        # shape = (1,5,10)
        prof_act[i] = Reshape(
            (lookback, final_profile_channels, 1))(prof_act[i])
        # shape = (5,10,1)
        prof_act[i] = Dense(units=profile_length,
                            activation='relu')(prof_act[i])
        # shape = (5,10,32)
        prof_act[i] = Permute((1, 3, 2))(prof_act[i])
        # shape = (5,32,10)
        prof_act[i] = Dense(units=final_profile_channels,
                            activation='relu')(prof_act[i])

    for i in range(num_profiles):
        profiles[i] = Multiply()([profiles[i], prof_act[i]])
        profiles[i] = Add()([profiles[i], merged[i]])
        # shape = (5,32,10)
        profiles[i] = Dense(units=15, activation='relu')(profiles[i])
        profiles[i] = Conv2D(filters=20, kernel_size=(1, 5), strides=(1, 1), padding='same',
                             activation='relu')(profiles[i])
        profiles[i] = Reshape((lookback, profile_length, 1, 20))(profiles[i])
        # shape = (5,32,1 20)
        profiles[i] = ConvLSTM2D(filters=1, kernel_size=(1, 5),
                                 strides=(1, 1), padding='same', activation='relu',
                                 recurrent_activation='hard_sigmoid')(profiles[i])
        #shape = (32,1,1)
        profiles[i] = Reshape((profile_length,))(profiles[i])
        #shape = (32,)
        profiles[i] = Dense(units=profile_length, activation=None,
                            name='target_' + input_profile_names[i])(profiles[i])

    outputs = [profiles[i] for i, sig in enumerate(
        input_profile_names) if sig in target_profile_names]

    model = Model(inputs=profile_inputs + actuator_inputs, outputs=outputs)
    return model
