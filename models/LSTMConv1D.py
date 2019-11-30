import keras

from keras import models
from keras import layers
from keras.layers import Input, Dense, LSTM, Conv1D, Conv2D, ConvLSTM2D, Dot, Add, Multiply, Concatenate, Reshape, Permute, ZeroPadding1D, Cropping1D, GlobalAveragePooling1D

from keras.models import Model


def build_lstmconv1d_joe(input_profile_names, target_profile_names, scalar_input_names,
                         actuator_names, lookbacks, lookahead, profile_length, std_activation, **kwargs):


    max_channels = kwargs.get('max_channels',16)
    rnn_layer = layers.LSTM

    # (lookbacks[input_profile_names[0]], profile_length)
    profile_inshape = (1, profile_length)
    # (lookbacks[actuator_names[0]],)

    # find the biggest lookback we;ll need for each signal
    max_profile_lookback = 0
    for sig in input_profile_names:
        if lookbacks[sig] > max_profile_lookback:
            max_profile_lookback = lookbacks[sig]
    max_actuator_lookback = 0
    for sig in actuator_names:
        if lookbacks[sig] > max_actuator_lookback:
            max_actuator_lookback = lookbacks[sig]
    max_scalar_lookback = 0
    for sig in scalar_input_names:
        if lookbacks[sig] > max_scalar_lookback:
            max_scalar_lookback = lookbacks[sig]

    past_scalar_inshape = (max(max_scalar_lookback+1, max_actuator_lookback+1),)
    future_actuator_inshape = (lookahead,)
    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)
    num_scalars = len(scalar_input_names)
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
   a) Lookback is defined NOT to include *current* timestep 
   b) lookahead is all timesteps *after* current timestep
   c) multiple prediction sigs not working yet, need to 
   for loop probabaly to get multiple output layers?
   '''

    # input each profile sig one by one and then concat them together
    profile_inputs = []
    profiles = []
    for i in range(num_profiles):
        profile_inputs.append(keras.layers.Input(
            profile_inshape, name='input_' + input_profile_names[i]))
        # profiles.append(Reshape((lookbacks[input_profile_names[i]], profile_length, 1))
        #                 (profile_inputs[i]))
        # import pdb; pdb.set_trace()
        profiles.append(keras.layers.Reshape(
            (1, profile_length, 1))(profile_inputs[i]))

    current_profiles = layers.Concatenate(axis=-1)(profiles)
    current_profiles = layers.Reshape(
        (profile_length, num_profiles))(current_profiles)

    # input previous and future actuators and concat each of them
    past_scalar_inputs = []
    actuator_future_inputs = []

    previous_scalars = []
    future_actuators = []

    for i in range(num_actuators):
        # import pdb; pdb.set_trace()
        actuator_future_inputs.append(
            layers.Input(future_actuator_inshape,
                         name="input_future_{}".format(actuator_names[i])))
        future_actuators.append(layers.Reshape((lookahead, 1))
                                (actuator_future_inputs[i]))

        past_scalar_inputs.append(
            layers.Input(past_scalar_inshape,
                         name="input_past_{}".format(actuator_names[i]))
        )
        previous_scalars.append(
            # Reshape((lookbacks[actuator_names[i]], 1))(actuator_past_inputs[i]))
            layers.Reshape((max(max_scalar_lookback+1, max_actuator_lookback+1), 1))(past_scalar_inputs[i]))

    for i in range(num_scalars):
        past_scalar_inputs.append(
            layers.Input(past_scalar_inshape,
                         name="input_{}".format(scalar_input_names[i]))
        )
        previous_scalars.append(
            # Reshape((lookbacks[actuator_names[i]], 1))(actuator_past_inputs[i]))
            layers.Reshape((max(max_scalar_lookback+1, max_actuator_lookback+1), 1))(past_scalar_inputs[i]))

    future_actuators = layers.Concatenate(axis=-1)(future_actuators)
    previous_scalars = layers.Concatenate(axis=-1)(previous_scalars)

    print(future_actuators.shape)
    print(previous_scalars.shape)
    print(current_profiles.shape)

    #######################################################################

    # previous_scalars = layers.Input(
    #     shape=(lookbacks[sigs_0d[0]]+1, num_actuators), name="previous_scalars")
    # future_actuators = layers.Input(
    #     shape=(delay, num_actuators), name="future_actuators")

    actuator_effect = rnn_layer(
        profile_length, activation=std_activation)(previous_scalars)
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

        current_profiles_processed_1 = layers.Conv1D(filters=max_channels, kernel_size=2,
                                                     padding='same', activation='relu')(current_profiles_processed_0)
        current_profiles_processed_2 = layers.Conv1D(filters=max_channels, kernel_size=4,
                                                     padding='same', activation='relu')(current_profiles_processed_1)
        current_profiles_processed_3 = layers.Conv1D(filters=max_channels, kernel_size=8,
                                                     padding='same', activation='relu')(current_profiles_processed_2)

        final_output = layers.Concatenate()(
            [current_profiles_processed_1, current_profiles_processed_2, current_profiles_processed_3])
        final_output = layers.Conv1D(filters=max_channels, kernel_size=4,
                                     padding='same', activation='relu')(final_output)
        final_output = layers.Conv1D(filters=int(max_channels/2), kernel_size=4,
                                     padding='same', activation='relu')(final_output)
        final_output = layers.Conv1D(filters=1, kernel_size=4,
                                     padding='same', activation='linear')(final_output)
        final_output = layers.Reshape(target_shape=(
            profile_length,), name="target_"+target_profile_names[i])(final_output)

        prof_act.append(final_output)
    print(len(prof_act))

    #     current_profiles,current_profiles_processed={},{}
    #     for sig in sigs_1d:
    #         current_profiles[sig]=layers.Input(shape=(rho_length_in,1),"current_profile_{}".format(sig))
    #         current_profiles_processed[sig]=layers.Conv1D(filters=3,kernel_size=4)(current_profiles[sig])
    model = Model(inputs=profile_inputs + past_scalar_inputs +
                  actuator_future_inputs, outputs=prof_act)

    #    model=models.Model(inputs=[previous_scalars]+list(current_profiles.values()),
    #                       outputs=list(current_profiles_processed.values()))

    return model
    #######################


def build_dumb_simple_model(input_profile_names, target_profile_names, scalar_input_names,
                            actuator_names, lookbacks, lookahead, profile_length, std_activation, **kwargs):

    num_profiles = len(input_profile_names)
    num_targets = len(target_profile_names)
    num_actuators = len(actuator_names)
    num_scalars = len(scalar_input_names)

    profile_inputs = []
    profiles = []
    for i in range(num_profiles):
        profile_inputs.append(
            Input((1, profile_length,), name='input_' + input_profile_names[i]))
        # size 65,
        profiles.append(Dense(units=profile_length,
                              activation=std_activation)(profile_inputs[i]))
        profiles[i] = Reshape((profile_length,))(profiles[i])
        # size 65,

    if num_scalars > 0:
        scalar_inputs = []
        scalars = []
        for i in range(num_scalars):
            scalar_inputs.append(
                Input((1,), name='input_' + scalar_input_names[i]))
            # size 1,
            scalars.append(Dense(units=profile_length,
                                 activation=std_activation)(scalar_inputs[i]))
            # size 65,

    actuator_future_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_future_inputs.append(
            Input((lookahead,), name='input_future_' + actuator_names[i]))
        # size lookahead,
        actuators.append(Reshape((lookahead, 1))(actuator_future_inputs[i]))
        # size lookahead, 1
        actuators[i] = Dense(units=profile_length,
                             activation=std_activation)(actuators[i])
        # size lookahead, 65
        actuators[i] = GlobalAveragePooling1D()(actuators[i])
        # size 65,

    outputs = []
    for i in range(num_targets):
        if num_scalars > 0:
            outputs.append(profiles+scalars+actuators)
        else:
            outputs.append(profiles+actuators)
        for j in range(len(outputs[i])):
            outputs[i][j] = Dense(units=profile_length,
                                  activation=std_activation)(outputs[i][j])
            # size 65,
        outputs[i] = Add()(outputs[i])
        # size 65,

        if kwargs.get('predict_mean'):
            outputs[i] = Dense(units=profile_length,
                               activation=std_activation)(outputs[i])
            outputs[i] = Reshape((1, profile_length))(outputs[i])
            outputs[i] = GlobalAveragePooling1D(
                name="target_"+target_profile_names[i])(outputs[i])
        else:
            outputs[i] = Dense(units=profile_length,
                               activation=std_activation, name="target_"+target_profile_names[i])(outputs[i])
    model_inputs = profile_inputs + actuator_future_inputs
    if num_scalars > 0:
        model_inputs += scalar_inputs
    model = Model(inputs=model_inputs, outputs=outputs)
    return model
