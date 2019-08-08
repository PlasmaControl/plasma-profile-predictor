from keras.layers import Input, Dense, LSTM, Conv1D, Conv2D, ConvLSTM2D, Dot, Add, Multiply, Concatenate, Reshape, Permute, ZeroPadding1D, Cropping1D
from keras.models import Model
import numpy as np



def get_model_conv2d_hyperparam(input_profile_names, target_profile_names,
                     actuator_names, profile_lookback, actuator_lookback,
                     lookahead, profile_length, std_activation):

    profile_inshape = (profile_lookback, profile_length)
    past_actuator_inshape = (actuator_lookback,)
    future_actuator_inshape = (lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)
    max_channels = 16

    profile_inputs = []
    profiles = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input(profile_inshape, name='input_' + input_profile_names[i]))
        profiles.append(Reshape((profile_lookback, profile_length, 1))
                        (profile_inputs[i]))
    profiles = Concatenate(axis=-1)(profiles)
    # shape = (lookback, length, channels=num_profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/8), kernel_size=(1, int(profile_length/12)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/4), kernel_size=(1, int(profile_length/8)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/2), kernel_size=(1, int(profile_length/6)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels), kernel_size=(1, int(profile_length/4)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    # shape = (lookback, length, channels)
    if profile_lookback > 1:
        profiles = Conv2D(filters=int(num_profiles*max_channels), kernel_size=(profile_lookback, 1),
                          strides=(1, 1), padding='valid', activation=std_activation)(profiles)
    profiles = Reshape((profile_length, int(
        num_profiles*max_channels)))(profiles)
    # shape = (length, channels)

    actuator_future_inputs = []
    actuator_past_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_future_inputs.append(
            Input(future_actuator_inshape, name='input_future_' + actuator_names[i]))
        actuator_past_inputs.append(
            Input(past_actuator_inshape, name='input_past_' + actuator_names[i]))
        actuators.append(Concatenate(
            axis=-1)([actuator_past_inputs[i], actuator_future_inputs[i]]))
        actuators[i] = Reshape((actuator_lookback+lookahead, 1))(actuators[i])
    actuators = Concatenate(axis=-1)(actuators)
    # shaoe = (time, num_actuators)
    actuators = Dense(units=int(num_profiles*max_channels/8),
                      activation=std_activation)(actuators)
    # actuators = Conv1D(filters=int(num_profiles*max_channels/8), kernel_size=3, strides=1,
    #                    padding='causal', activation=std_activation)(actuators)
    actuators = Dense(units=int(num_profiles*max_channels/4),
                      activation=std_activation)(actuators)
    # actuators = Conv1D(filters=int(num_profiles*max_channels/4), kernel_size=3, strides=1,
    #                    padding='causal', activation=std_activation)(actuators)
    actuators = Dense(units=int(num_profiles*max_channels/2),
                      activation=std_activation)(actuators)
    actuators = LSTM(units=int(num_profiles*max_channels), activation=std_activation,
                     recurrent_activation='hard_sigmoid')(actuators)
    actuators = Reshape((int(num_profiles*max_channels), 1))(actuators)
    # shape = (channels, 1)
    actuators = Dense(units=int(profile_length/4),
                      activation=std_activation)(actuators)
    actuators = Dense(units=int(profile_length/2),
                      activation=std_activation)(actuators)
    actuators = Dense(units=profile_length, activation=None)(actuators)
    # shape = (channels, profile_length)
    actuators = Permute(dims=(2, 1))(actuators)
    # shape = (profile_length, channels)

    merged = Add()([profiles, actuators])
    merged = Reshape((1, profile_length, int(
        num_profiles*max_channels)))(merged)
    # shape = (1, length, channels)

    prof_act = []
    for i in range(num_targets):
        prof_act.append(Conv2D(filters=max_channels, kernel_size=(1, int(profile_length/4)), strides=(1, 1),
                               padding='same', activation=std_activation)(merged))
        # shape = (1,length,max_channels)
        prof_act[i] = Conv2D(filters=int(max_channels/2), kernel_size=(1, int(profile_length/8)),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=int(max_channels/4), kernel_size=(1, int(profile_length/6)),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=int(max_channels/8), kernel_size=(1, int(profile_length/4)),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=1, kernel_size=(1, int(profile_length/4)), strides=(1, 1),
                             padding='same', activation=None)(prof_act[i])
        # shape = (1,length,1)
        prof_act[i] = Reshape((profile_length,), name='target_' +
                              target_profile_names[i])(prof_act[i])
    model = Model(inputs=profile_inputs + actuator_past_inputs +
                  actuator_future_inputs, outputs=prof_act)
    return model

def get_model_conv2d(input_profile_names, target_profile_names,
                     actuator_names, profile_lookback, actuator_lookback,
                     lookahead, profile_length, std_activation):

    profile_inshape = (profile_lookback, profile_length)
    past_actuator_inshape = (actuator_lookback,)
    future_actuator_inshape = (lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)
    max_channels = 32

    profile_inputs = []
    profiles = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input(profile_inshape, name='input_' + input_profile_names[i]))
        profiles.append(Reshape((profile_lookback, profile_length, 1))
                        (profile_inputs[i]))
    profiles = Concatenate(axis=-1)(profiles)
    # shape = (lookback, length, channels=num_profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/8), kernel_size=(1, int(profile_length/12)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/4), kernel_size=(1, int(profile_length/8)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/2), kernel_size=(1, int(profile_length/6)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels), kernel_size=(1, int(profile_length/4)),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    # shape = (lookback, length, channels)
    if profile_lookback > 1:
        profiles = Conv2D(filters=int(num_profiles*max_channels), kernel_size=(profile_lookback, 1),
                          strides=(1, 1), padding='valid', activation=std_activation)(profiles)
    profiles = Reshape((profile_length, int(
        num_profiles*max_channels)))(profiles)
    # shape = (length, channels)

    actuator_future_inputs = []
    actuator_past_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_future_inputs.append(
            Input(future_actuator_inshape, name='input_future_' + actuator_names[i]))
        actuator_past_inputs.append(
            Input(past_actuator_inshape, name='input_past_' + actuator_names[i]))
        actuators.append(Concatenate(
            axis=-1)([actuator_past_inputs[i], actuator_future_inputs[i]]))
        actuators[i] = Reshape((actuator_lookback+lookahead, 1))(actuators[i])
    actuators = Concatenate(axis=-1)(actuators)
    # shaoe = (time, num_actuators)
    actuators = Dense(units=int(num_profiles*max_channels/8),
                      activation=std_activation)(actuators)
    # actuators = Conv1D(filters=int(num_profiles*max_channels/8), kernel_size=3, strides=1,
    #                    padding='causal', activation=std_activation)(actuators)
    actuators = Dense(units=int(num_profiles*max_channels/4),
                      activation=std_activation)(actuators)
    # actuators = Conv1D(filters=int(num_profiles*max_channels/4), kernel_size=3, strides=1,
    #                    padding='causal', activation=std_activation)(actuators)
    actuators = Dense(units=int(num_profiles*max_channels/2),
                      activation=std_activation)(actuators)
    actuators = LSTM(units=int(num_profiles*max_channels), activation=std_activation,
                     recurrent_activation='hard_sigmoid')(actuators)
    actuators = Reshape((int(num_profiles*max_channels), 1))(actuators)
    # shape = (channels, 1)
    actuators = Dense(units=int(profile_length/4),
                      activation=std_activation)(actuators)
    actuators = Dense(units=int(profile_length/2),
                      activation=std_activation)(actuators)
    actuators = Dense(units=profile_length, activation=None)(actuators)
    # shape = (channels, profile_length)
    actuators = Permute(dims=(2, 1))(actuators)
    # shape = (profile_length, channels)

    merged = Add()([profiles, actuators])
    merged = Reshape((1, profile_length, int(
        num_profiles*max_channels)))(merged)
    # shape = (1, length, channels)

    prof_act = []
    for i in range(num_targets):
        prof_act.append(Conv2D(filters=max_channels, kernel_size=(1, int(profile_length/4)), strides=(1, 1),
                               padding='same', activation=std_activation)(merged))
        # shape = (1,length,max_channels)
        prof_act[i] = Conv2D(filters=int(max_channels/2), kernel_size=(1, int(profile_length/8)),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=int(max_channels/4), kernel_size=(1, int(profile_length/6)),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=int(max_channels/8), kernel_size=(1, int(profile_length/4)),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=1, kernel_size=(1, int(profile_length/4)), strides=(1, 1),
                             padding='same', activation=None)(prof_act[i])
        # shape = (1,length,1)
        prof_act[i] = Reshape((profile_length,), name='target_' +
                              target_profile_names[i])(prof_act[i])
    model = Model(inputs=profile_inputs + actuator_past_inputs +
                  actuator_future_inputs, outputs=prof_act)
    return model


def get_model_lstm_conv2d(input_profile_names, target_profile_names,
                          actuator_names, profile_lookback, actuator_lookback,
                          lookahead, profile_length, std_activation):

    profile_inshape = (profile_lookback, profile_length)
    actuator_inshape = (actuator_lookback + lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)
    max_channels = 32

    profile_inputs = []
    profiles = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input(profile_inshape, name='input_' + input_profile_names[i]))
        profiles.append(Reshape((profile_lookback, profile_length, 1))
                        (profile_inputs[i]))
    profiles = Concatenate(axis=-1)(profiles)
    # shape = (lookback, length, channels=num_profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/8), kernel_size=(1, 5),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels/4), kernel_size=(1, 10),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    profiles = Conv2D(filters=int(num_profiles*max_channels), kernel_size=(1, 15),
                      strides=(1, 1), padding='same', activation=std_activation)(profiles)
    # shape = (lookback, length, channels)
    if profile_lookback > 1:
        profiles = Reshape((profile_lookback, 1, profile_length,
                            int(num_profiles*max_channels)))(profiles)
        # shape = (lookback, 1,  length, channels)
        profiles = ConvLSTM2D(filters=int(num_profiles*max_channels), kernel_size=(10, 1),
                              strides=(1, 1), padding='same', activation=std_activation,
                              recurrent_activation='hard_sigmoid')(profiles)
        #shape = (1, length, channels)
        profiles = Reshape((profile_length, int(
            num_profiles*max_channels)))(profiles)
        # shape = (length, channels)
    else:
        profiles = Conv2D(filters=int(num_profiles*max_channels), kernel_size=(1, 10),
                          strides=(1, 1), padding='same', activation=std_activation)(profiles)
        profiles = Reshape((profile_length, int(
            num_profiles*max_channels)))(profiles)
        # shape = (length, channels)

    actuator_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_inputs.append(
            Input(actuator_inshape, name='input_' + actuator_names[i]))
        actuators.append(
            Reshape((actuator_lookback+lookahead, 1))(actuator_inputs[i]))
    actuators = Concatenate(axis=-1)(actuators)
    # shaoe = (time, num_actuators)
    actuators = Dense(units=int(num_profiles*max_channels/8),
                      activation=std_activation)(actuators)
    actuators = Conv1D(filters=int(num_profiles*max_channels/4), kernel_size=3, strides=1,
                       padding='causal', activation=std_activation)(actuators)
    actuators = LSTM(units=int(num_profiles*max_channels), activation=std_activation,
                     recurrent_activation='hard_sigmoid')(actuators)
    actuators = Reshape((int(num_profiles*max_channels), 1))(actuators)
    # shape = (channels, 1)
    actuators = Dense(units=profile_length,
                      activation=std_activation)(actuators)
    actuators = Dense(units=profile_length, activation=None)(actuators)
    # shape = (channels, profile_length)
    actuators = Permute(dims=(2, 1))(actuators)
    # shape = (profile_length, channels)

    merged = Add()([profiles, actuators])
    merged = Reshape((1, profile_length, int(
        num_profiles*max_channels)))(merged)
    # shape = (1, length, channels)

    prof_act = []
    for i in range(num_targets):
        prof_act.append(Conv2D(filters=max_channels, kernel_size=(1, 15), strides=(1, 1),
                               padding='same', activation=std_activation)(merged))
        # shape = (1,length,max_channels)
        prof_act[i] = Conv2D(filters=int(max_channels/4), kernel_size=(1, 15),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=int(max_channels/8), kernel_size=(1, 10),
                             strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
        prof_act[i] = Conv2D(filters=1, kernel_size=(1, 5), strides=(1, 1),
                             padding='same', activation=None)(prof_act[i])
        # shape = (1,length,1)
        prof_act[i] = Reshape((profile_length,), name='target_' +
                              target_profile_names[i])(prof_act[i])
    model = Model(inputs=profile_inputs + actuator_inputs, outputs=prof_act)
    return model


def get_model_simple_lstm(input_profile_names, target_profile_names,
                          actuator_names, profile_lookback, actuator_lookback,
                          lookahead, profile_length, std_activation):
    profile_inshape = (profile_lookback, profile_length)
    actuator_inshape = (actuator_lookback + lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)

    profile_inputs = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input(profile_inshape, name='input_' + input_profile_names[i]))
    if num_profiles > 1:
        profiles = Concatenate(axis=-1)(profile_inputs)
    else:
        profiles = profile_inputs[0]
    profiles = ZeroPadding1D(
        padding=(actuator_lookback-profile_lookback, lookahead))(profiles)
    actuator_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_inputs.append(
            Input(actuator_inshape, name='input_' + actuator_names[i]))
        actuators.append(
            Reshape((actuator_lookback+lookahead, 1))(actuator_inputs[i]))
    if num_actuators > 1:
        actuators = Concatenate(axis=-1)(actuators)
    else:
        actuators = actuators[0]
    full = Concatenate(axis=-1)([actuators, profiles])
    full = Dense(units=int(num_targets*profile_length*.8),
                 activation=std_activation)(full)
    full = Dense(units=int(num_targets*profile_length*.6),
                 activation=std_activation)(full)
    full = Dense(units=int(num_targets*profile_length*.4),
                 activation=std_activation)(full)
    full = LSTM(units=int(num_targets*profile_length*.6), activation=std_activation,
                recurrent_activation='hard_sigmoid')(full)
    full = Dense(units=int(num_targets*profile_length*.8),
                 activation=std_activation)(full)
    outputs = Dense(units=num_targets*profile_length, activation=None)(full)
    outputs = Reshape((num_targets*profile_length, 1))(outputs)
    targets = []
    for i in range(num_targets):
        targets.append(Cropping1D(cropping=(i*profile_length,
                                            (num_targets-i-1)*profile_length))(outputs))
        targets[i] = Reshape((profile_length,),
                             name='target_' + target_profile_names[i])(targets[i])
    model = Model(inputs=profile_inputs+actuator_inputs, outputs=targets)
    return model


def get_model_linear_systems(input_profile_names, target_profile_names,
                             actuator_names, profile_lookback, actuator_lookback,
                             lookahead, profile_length, std_activation):

    profile_inshape = (profile_lookback, profile_length)
    actuator_inshape = (actuator_lookback + lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)

    profile_inputs = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input(profile_inshape, name='input_' + input_profile_names[i]))
    if num_profiles > 1:
        profiles = Concatenate(axis=-1)(profile_inputs)
    else:
        profiles = profile_inputs[0]
    profile_response = Dense(
        int(profile_length/2*num_profiles), activation=std_activation)(profiles)
    profile_response = Dense(
        int(profile_length/2*num_profiles), activation=std_activation)(profile_response)
    if profile_lookback > 1:
        profile_response = LSTM(int(profile_length/2*num_profiles), activation=std_activation,
                                recurrent_activation='hard_sigmoid',
                                return_sequences=True)(profile_response)
    else:
        profile_response = Dense(int(profile_length/2*num_profiles),
                                 activation=std_activation)(profile_response)
    profile_response = Dense(int(profile_length/2*num_profiles),
                             activation=std_activation)(profile_response)

    actuator_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_inputs.append(
            Input(actuator_inshape, name='input_' + actuator_names[i]))
        actuators.append(
            Reshape((actuator_lookback+lookahead, 1))(actuator_inputs[i]))
    if num_actuators > 1:
        actuators = Concatenate(axis=-1)(actuators)
    else:
        actuators = actuators[0]

    actuator_response = Dense(
        profile_lookback, activation=std_activation)(actuators)
    actuator_response = Dense(
        profile_lookback, activation=std_activation)(actuator_response)
    actuator_response = LSTM(profile_lookback, activation=std_activation,
                             recurrent_activation='hard_sigmoid',
                             return_sequences=True)(actuator_response)
    total_response = Dot(axes=(2, 1))([actuator_response, profile_response])
    total_response = LSTM(int(profile_length/2*num_targets), activation=std_activation,
                          recurrent_activation='hard_sigmoid')(total_response)
    total_response = Dense(int(profile_length*.75*num_targets),
                           activation=std_activation)(total_response)
    total_response = Dense(profile_length*num_targets)(total_response)
    total_response = Reshape((num_targets*profile_length, 1))(total_response)

    targets = []
    for i in range(num_targets):
        targets.append(Cropping1D(cropping=(i*profile_length,
                                            (num_targets-i-1)*profile_length))(total_response))
        targets[i] = Reshape((profile_length,),
                             name='target_' + target_profile_names[i])(targets[i])
    model = Model(inputs=profile_inputs+actuator_inputs, outputs=targets)
    return model
