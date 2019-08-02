
from keras import models
from keras import layers

from keras.layers import Input, Dense, LSTM, Conv1D, Conv2D, ConvLSTM2D, Dot, Add, Multiply, Concatenate, Reshape, Permute, ZeroPadding1D, Cropping1D
from keras.models import Model
import numpy as np


def build_lstmconv1d_joe(input_profile_names, target_profile_names,
                actuator_names, lookbacks,
                lookahead, profile_length, std_activation):

    rnn_layer = layers.LSTM

    profile_inshape = (lookbacks[input_profile_names[0]], profile_length)
    past_actuator_inshape = (lookbacks[actuator_names[0]],)
    future_actuator_inshape = (lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)
    #max_channels = 32

    # num_actuators = len(sigs_0d)
    # num_sigs_1d = len(sigs_1d)
    # num_sigs_predict = len(sigs_predict)

    ########################################################################
    ########################
    ###Inputs###############
    ####Profiles############
    ######Past/Future Actu##
    '''
    Deliverables from inputs section:
    1. variable name: 'current_profiles'
        Tensor that concat all profiles in shape
        (profile lookback, profile len. num profiles). 
        Note: Don't think different lookback lengths are fully
        supported yet, even though we take in a lookback dict
        with lookbacks for each sig
    2. variable name: 'previous_actuators'
        Tensor for all past actuators in shape
        (lookback, num actu)
    3. variable name: 'future_actuators'
        Tensor for all past actuators in shape
        (lookahead, num actu)

    Notes:
        a) Lookback is defined to include *current* timestep 
        b) lookahead is all timesteps *after* current timestep
        c) multiple prediction sigs not working yet, need to 
            for loop probabaly to get multiple output layers?
    '''

    # input each profile sig one by one and then concat them together
    profile_inputs = []
    profiles = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input(profile_inshape, name='input_' + input_profile_names[i]))
        profiles.append(Reshape((lookbacks[input_profile_names[i]], profile_length, 1))
                        (profile_inputs[i]))
    current_profiles = Concatenate(axis=-1)(profiles)
    current_profiles = Reshape(
        (profile_length, num_profiles))(current_profiles)

    # input previous and future actuators and concat each of them
    actuator_past_inputs = []
    actuator_future_inputs = []

    previous_actuators = []
    future_actuators = []

    for i in range(num_actuators):
        actuator_future_inputs.append(
            Input(future_actuator_inshape,
                  name="input_future_{}".format(actuator_names[i]))
        )
        actuator_past_inputs.append(
            Input(past_actuator_inshape,
                  name="input_past_{}".format(actuator_names[i]))
        )

        future_actuators.append(Reshape((lookahead, 1))
                                (actuator_future_inputs[i]))
        previous_actuators.append(
            Reshape((lookbacks[actuator_names[i]], 1))(actuator_past_inputs[i]))

    future_actuators = Concatenate(axis=-1)(future_actuators)
    previous_actuators = Concatenate(axis=-1)(previous_actuators)
    
    
    print(future_actuators.shape)
    print(previous_actuators.shape)
    print(current_profiles.shape)

    #######################################################################

    # previous_actuators = layers.Input(
    #     shape=(lookbacks[sigs_0d[0]]+1, num_actuators), name="previous_actuators")
    # future_actuators = layers.Input(
    #     shape=(delay, num_actuators), name="future_actuators")

    actuator_effect = rnn_layer(
        profile_length, activation=std_activation)(previous_actuators)
    actuator_effect = layers.Reshape(
        target_shape=(profile_length, 1))(actuator_effect)

    future_actuator_effect = rnn_layer(
        profile_length, activation=std_activation)(future_actuators)
    future_actuator_effect = layers.Reshape(
        target_shape=(profile_length, 1))(future_actuator_effect)

    # current_profiles = layers.Input(
    #     shape=(profile_length, num_sigs_1d), name="previous_profiles")
    # take out for the other version

    current_profiles_processed_0 = layers.Concatenate()(
        [current_profiles, actuator_effect, future_actuator_effect])

    prof_act = []
    for i in range(num_targets):

        current_profiles_processed_1 = layers.Conv1D(filters=8, kernel_size=2,
                                                     padding='same', activation='relu')(current_profiles_processed_0)
        current_profiles_processed_2 = layers.Conv1D(filters=8, kernel_size=4,
                                                     padding='same', activation='relu')(current_profiles_processed_1)
        current_profiles_processed_3 = layers.Conv1D(filters=8, kernel_size=8,
                                                     padding='same', activation='relu')(current_profiles_processed_2)

        final_output = layers.Concatenate()(
            [current_profiles_processed_1, current_profiles_processed_2, current_profiles_processed_3])
        final_output = layers.Conv1D(filters=10, kernel_size=4,
                                     padding='same', activation='tanh')(final_output)
        final_output = layers.Conv1D(filters=1, kernel_size=4,
                                     padding='same', activation='linear')(final_output)
        final_output = layers.Reshape(target_shape = (profile_length,), name="target_"+target_profile_names[i])(final_output)
        

        prof_act.append(final_output)
    print(len(prof_act))


#     current_profiles,current_profiles_processed={},{}
#     for sig in sigs_1d:
#         current_profiles[sig]=layers.Input(shape=(rho_length_in,1),"current_profile_{}".format(sig))
#         current_profiles_processed[sig]=layers.Conv1D(filters=3,kernel_size=4)(current_profiles[sig])
    model = Model(inputs=profile_inputs + actuator_past_inputs +
                  actuator_future_inputs, outputs=prof_act)

#    model=models.Model(inputs=[previous_actuators]+list(current_profiles.values()),
#                       outputs=list(current_profiles_processed.values()))


    return model
    #######################

