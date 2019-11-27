import numpy as np
import keras
from keras.layers import Input, Dense, Cropping1D, Dot, SimpleRNN, TimeDistributed
from keras.layers import add, subtract, Concatenate, Reshape, Permute
from keras.models import Model


def get_state_joiner(profile_names, scalar_names, timesteps, profile_length, batch_size=None):
    """Create models to split and join inputs/outputs for state variables

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): list of names of scalars 
        timesteps (int): how many timesteps in the future to predict
        profile_length (int): number of psi pts in discretized profiles

    Returns:
        joiner (model): model that takes individual inputs and returns a combined tensor
    """

    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles*profile_length + num_scalars

    profile_inputs = [Input(batch_shape=(batch_size,timesteps, profile_length),
                            name='input_' + nm) for nm in profile_names]
    profiles = Concatenate(axis=-1,name='profile_joiner')(profile_inputs) if num_profiles>1 else profile_inputs[0]
    scalar_inputs = [Input(batch_shape=(batch_size,timesteps, 1), name='input_' + nm)
                     for nm in scalar_names]
    scalars = Concatenate(
        axis=-1)(scalar_inputs) if num_scalars > 1 else scalar_inputs[0] if num_scalars > 0 else []
    x = Concatenate(axis=-1,name='state_joiner')([profiles, scalars]
                             ) if num_scalars > 0 else profiles
    if timesteps <=1:
        x = Reshape((state_dim,))(x)
    joiner_model = Model(inputs=profile_inputs +
                         scalar_inputs, outputs=x, name='state_joiner')

  
    return joiner_model


def get_state_splitter(profile_names, scalar_names, timesteps, profile_length, batch_size=None):
    """Create models to split and join inputs/outputs for state variables

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): list of names of scalars 
        timesteps (int): how many timesteps in the future to predict
        profile_length (int): number of psi pts in discretized profiles

    Returns:
        splitter (model): model that takes combined tensor and returns individual tensors for each signal
    """

    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles*profile_length + num_scalars

    yi = Input(batch_shape=(batch_size,state_dim))
    y = Reshape((state_dim, 1))(yi)
    profile_outputs = [Reshape((profile_length,),name='output_' + nm)(
        Cropping1D((i*profile_length, state_dim-(i+1)*profile_length))(y)) 
                       for i, nm in enumerate(profile_names)]
    scalar_outputs = [Reshape((1,),name='output_' + scalar_names[i])(
        Cropping1D((num_profiles*profile_length + i, state_dim-num_profiles*profile_length - (i+1)))(y))
                      for i, nm in enumerate(scalar_names)]
    splitter_model = Model(
        inputs=yi, outputs=profile_outputs + scalar_outputs, name='state_splitter')
    return splitter_model


def get_control_joiner(actuator_names, timesteps, batch_size=None):
    """Create models to split and join inputs/outputs for control variables

    Args:
        actuator_names (str): List of names of actuators
        timesteps (int): how many timesteps in the future to predict + how many timesteps of previous actuators

    Returns:
        joiner (model): model that takes individual inputs and returns a combined tensor
    """
    num_actuators = len(actuator_names)
    actuator_inputs = [Input(batch_shape=(batch_size,timesteps, 1), name='input_' + nm)
                       for nm in actuator_names]
    actuators = Concatenate(
        axis=-1,name='control_joiner')(actuator_inputs) if num_actuators > 1 else actuator_inputs[0]
    if timesteps<= 1:
        actuators = Reshape((num_actuators,))(actuators)
    joiner = Model(inputs=actuator_inputs,
                   outputs=actuators, name='control_joiner')

    return joiner


def get_control_splitter(actuator_names, timesteps, batch_size=None):
    """Create models to split and join inputs/outputs for control variables

    Args:
        actuator_names (str): List of names of actuators
        timesteps (int): how many timesteps in the future to predict + how many timesteps of previous actuators

    Returns:
        splitter (model): model that takes combined tensor and returns individual tensors for each signal
    """
    num_actuators = len(actuator_names)

    ui = Input(batch_shape=(batch_size,num_actuators))
    u = Reshape((num_actuators, 1))(ui)
    actuator_outputs = [Reshape((1, 1),name='output_' + nm)(Cropping1D(
        (i, num_actuators-(i+1)))(u)) for i, nm in enumerate(actuator_names)]
    splitter = Model(inputs=ui, outputs=actuator_outputs,
                     name='control_splitter')
    return splitter


def get_control_encoder_dense(actuator_names, control_latent_dim,
                              std_activation, batch_size=None,**kwargs):
    """Control encoder using dense network.

    Args:
        actuator_names (str): List of names of actuators
        control_latent_dim (int): dimensionality of the encoded variables
        std_activation (str or fn): activation function to apply to hidden layers
        num_layers (int): number of hidden layers
        layer_scale (float): power law scaling for size of hidden layers
            size of layer(i) = min_size + (max_size-min_size)*(i/num_layers)**layer_scale

    Returns:
        control_encoder (model): Keras model that takes each actuator as individual inputs
            and returns a single tensor of the encoded values.
    """
    layer_scale = kwargs.get('layer_scale', 1)
    num_layers = kwargs.get('num_layers', 6)
    num_actuators = len(actuator_names)
    joiner = get_control_joiner(actuator_names, 1, batch_size)
    u = joiner(joiner.inputs)
    for i in range(num_layers):
        units = int(control_latent_dim + (num_actuators-control_latent_dim)
                    * ((num_layers-i-1)/(num_layers-1))**layer_scale)
        u = Dense(units=units, activation=std_activation, use_bias=True)(u)
    u = Reshape((control_latent_dim,))(u)
    encoder = Model(inputs=joiner.inputs, outputs=u,
                    name='dense_control_encoder')
    return encoder


def get_control_decoder_dense(actuator_names, control_latent_dim,
                              std_activation, batch_size=None, **kwargs):
    """Control decoder using dense network.

    Args:
        actuator_names (str): List of names of actuators
        control_latent_dim (int): dimensionality of the encoded variables
        std_activation (str or fn): activation function to apply to hidden layers
        num_layers (int): number of hidden layers
        layer_scale (float): power law scaling for size of hidden layers
            size of layer(i) = min_size + (max_size-min_size)*(i/num_layers)**layer_scale

    Returns:
        control_encoder (model): Keras model that takes a single tensor of the 
            encoded values and returns each actuator as individual outputs.
    """
    layer_scale = kwargs.get('layer_scale', 1)
    num_layers = kwargs.get('num_layers', 6)
    num_actuators = len(actuator_names)
    splitter = get_control_splitter(actuator_names, 1, batch_size)
    ui = Input(batch_shape=(batch_size,control_latent_dim))
    u = ui
    for i in range(num_layers-1):
        units = int(num_actuators - (num_actuators-control_latent_dim)
                    * ((num_layers-i-1)/(num_layers-1))**layer_scale)
        u = Dense(units=units, activation=std_activation, use_bias=True)(u)
    u = Dense(units=num_actuators, activation='linear')(u)
    u = Reshape((num_actuators,))(u)
    outputs = splitter(u)
    decoder = Model(inputs=ui, outputs=outputs, name='dense_control_decoder')
    return decoder


def get_state_encoder_dense(profile_names, scalar_names, profile_length,
                            state_latent_dim, std_activation, batch_size=None, **kwargs):
    """State encoder using dense network.

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): List of names of scalars
        profile_length (int): number of psi pts in discretized profiles
        state_latent_dim (int): dimensionality of the encoded variables
        std_activation (str or fn): activation function to apply to hidden layers
        num_layers (int): number of hidden layers
        layer_scale (float): power law scaling for size of hidden layers
            size of layer(i) = min_size + (max_size-min_size)*(i/num_layers)**layer_scale

    Returns:
        state_encoder (model): Keras model that takes each profile and scalar as individual inputs
            and returns a single tensor of the encoded values.
    """
    layer_scale = kwargs.get('layer_scale', 1)
    num_layers = kwargs.get('num_layers', 6)
    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles*profile_length + num_scalars

    joiner = get_state_joiner(
        profile_names, scalar_names, 1, profile_length, batch_size)
    x = joiner(joiner.inputs)
    for i in range(num_layers):
        units = int(state_latent_dim + (state_dim-state_latent_dim)
                    * ((num_layers-i-1)/(num_layers-1))**layer_scale)
        x = Dense(units=units, activation=std_activation, use_bias=True)(x)
    x = Reshape((state_latent_dim,))(x)
    encoder = Model(inputs=joiner.inputs, outputs=x,
                    name='dense_state_encoder')
    return encoder


def get_state_decoder_dense(profile_names, scalar_names, profile_length,
                            state_latent_dim, std_activation, batch_size=None, **kwargs):
    """State decoder using dense network.

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): List of names of scalars
        profile_length (int): number of psi pts in discretized profiles
        state_latent_dim (int): dimensionality of the encoded variables
        std_activation (str or fn): activation function to apply to hidden layers
        num_layers (int): number of hidden layers
        layer_scale (float): power law scaling for size of hidden layers
            size of layer(i) = min_size + (max_size-min_size)*(i/num_layers)**layer_scale

    Returns:
        state_decoder (model): Keras model that takes a single tensor of the 
            encoded values and returns each profile/scalar as individual outputs.
    """
    layer_scale = kwargs.get('layer_scale', 1)
    num_layers = kwargs.get('num_layers', 6)
    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles*profile_length + num_scalars

    xi = Input(batch_shape=(batch_size,state_latent_dim))
    x = xi
    for i in range(num_layers-1, 0, -1):
        units = int(state_latent_dim + (state_dim-state_latent_dim)
                    * ((num_layers-i-1)/(num_layers-1))**layer_scale)
        x = Dense(units=units, activation=std_activation, use_bias=True)(x)
    y = Dense(units=state_dim, activation='linear')(x)
    y = Reshape((state_dim,))(y)
    splitter = get_state_splitter(
        profile_names, scalar_names, 1, profile_length, batch_size)
    outputs = splitter(y)
    decoder = Model(inputs=xi, outputs=outputs, name='dense_state_decoder')
    return decoder


def get_latent_linear_model(state_latent_dim, control_latent_dim, lookback, lookahead, regularization=None, batch_size=None):
    """Linear model for encoded variables

    Args:
        state_latent_dim (int): dimensionality of the encoded state variables
        control_latent_dim (int): dimensionality of the encoded control variables
        lookback (int): how many timesteps of past actuators to use
        lookahead (int): how many timesteps in the future to predict
        regularization (dict): Dictionary of L1 and L2 regularization parameters 
            for A and B matrices. keys 'l1A', l2A', 'l1B', 'l2B'

    Returns:
        latent_linear_system (model): Keras model that takes state and control 
            tensors as input and returns a tensor of residual values in the 
            linear dynamic approximation
    """
    if regularization is None:
        regularization = {'l1A': 0,
                          'l2A': 0,
                          'l1B': 0,
                          'l2B': 0}
    xi = Input(batch_shape=(batch_size,lookahead+1, state_latent_dim), name='xi')
    ui = Input(batch_shape=(batch_size,lookback+lookahead, control_latent_dim), name='ui')
    x0 = Reshape((state_latent_dim,), name='x0')(
        Cropping1D((0, lookahead))(xi))
    x1 = Cropping1D((1, 0), name='x1')(xi)
    u = Cropping1D((lookback, 0), name='u')(ui)

    AB = SimpleRNN(units=state_latent_dim,
                   activation='linear',
                   use_bias=False,
                   name='AB_matrices',
                   kernel_regularizer=keras.regularizers.l1_l2(
                       l1=regularization['l1B'], l2=regularization['l2B']),
                   recurrent_regularizer=keras.regularizers.l1_l2(
                       l1=regularization['l1A'], l2=regularization['l2A']),
                   return_sequences=True)

    x1est = Reshape((lookahead, state_latent_dim),
                    name='x1est')(AB(u, initial_state=x0))
    x1_residual = subtract([x1, x1est], name='linear_system_residual')
    model = Model(inputs=[xi, ui], outputs=x1_residual,
                  name='latent_linear_system')
    return model


def make_autoencoder(state_encoder_type, state_decoder_type, control_encoder_type, control_decoder_type,
                     state_encoder_kwargs, state_decoder_kwargs, control_encoder_kwargs, control_decoder_kwargs,
                     profile_names, scalar_names, actuator_names,
                     state_latent_dim, control_latent_dim, profile_length, lookback, lookahead, batch_size=None, **kwargs):
    """Linear Recurrent autoencoder

    Args:
        state_encoder_type (str): Type of netork to use for state encoding
        state_decoder_type (str): Type of netork to use for state decoding
        control_encoder_type (str): Type of netork to use for control encoding
        control_decoder_type (str): Type of netork to use for control decoding
        state_encoder_kwargs (dict): Dictionary of keyword arguments for state encoder model
        state_decoder_kwargs (dict): Dictionary of keyword arguments for state decoder model
        control_encoder_kwargs (dict): Dictionary of keyword arguments for control encoder model
        control_decoder_kwargs (dict): Dictionary of keyword arguments for control decoder model
        profile_names (str): List of names of profiles
        scalar_names (str): list of names of scalars 
        actuator_names (str): list of names of actuators
        state_latent_dim (int): dimensionality of the encoded state variables
        control_latent_dim (int): dimensionality of the encoded control variables
        profile_length (int): number of psi pts in discretized profiles
        lookback (int): how many timesteps of past actuators to use
        lookahead (int): how many timesteps in the future to predict



    Returns:
        autoencoder (model): Keras model that takes profile, scalar, and actuator 
            tensors as input and returns a tensor of residual values in the 
            state reconstruction, control reconstruction, and linear dynamic approximation
    """

    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    num_actuators = len(actuator_names)
    state_dim = num_profiles*profile_length + num_scalars
    state_encoders = {'dense': get_state_encoder_dense}
    state_decoders = {'dense': get_state_decoder_dense}
    control_encoders = {'dense': get_control_encoder_dense}
    control_decoders = {'dense': get_control_decoder_dense}

    state_encoder = state_encoders[state_encoder_type](profile_names,
                                                       scalar_names,
                                                       profile_length,
                                                       state_latent_dim,
                                                       batch_size=batch_size,
                                                       **state_encoder_kwargs)
    state_decoder = state_decoders[state_decoder_type](profile_names,
                                                       scalar_names,
                                                       profile_length,
                                                       state_latent_dim,
                                                       batch_size=batch_size,
                                                       **state_decoder_kwargs)
    control_encoder = control_encoders[control_encoder_type](actuator_names,
                                                             control_latent_dim,
                                                             batch_size=batch_size,
                                                             **control_encoder_kwargs)
    control_decoder = control_decoders[control_decoder_type](actuator_names,
                                                             control_latent_dim,
                                                             batch_size=batch_size,
                                                             **control_decoder_kwargs)
    state_input_model = get_state_joiner(profile_names,
                                         scalar_names,
                                         lookahead+1,
                                         profile_length,
                                         batch_size=batch_size)
    control_input_model = get_control_joiner(actuator_names,
                                             lookback+lookahead,
                                             batch_size=batch_size)
    
    state_joiner = get_state_joiner(profile_names,
                                    scalar_names,
                                    1,
                                    profile_length,
                                    batch_size=batch_size)
    state_splitter = get_state_splitter(profile_names,
                                        scalar_names,
                                        1,
                                        profile_length,
                                        batch_size=batch_size)
    control_joiner = get_control_joiner(actuator_names,
                                        1,
                                        batch_size=batch_size)
    control_splitter = get_control_splitter(actuator_names,
                                            1,
                                            batch_size=batch_size)

    state_encoder = Model(state_splitter.inputs,state_encoder(state_splitter.outputs))
    control_encoder = Model(control_splitter.inputs,control_encoder(control_splitter.outputs))
    state_decoder = Model(state_decoder.inputs, state_joiner(state_decoder.outputs))
    control_decoder = Model(control_decoder.inputs,control_joiner(control_decoder.outputs))

    xi = state_input_model.outputs[0]
    x = TimeDistributed(state_encoder,batch_input_shape=(batch_size,lookahead+1,state_dim),name='state_encoder_time_dist')(xi)
    xo = TimeDistributed(state_decoder,batch_input_shape=(batch_size,lookahead+1,state_dim),name='state_decoder_time_dist')(x)
    
    ui = control_input_model.outputs[0]
    u = TimeDistributed(control_encoder,batch_input_shape=(batch_size,lookback+lookahead,num_actuators),name='ctrl_encoder_time_dist')(ui)
    uo = TimeDistributed(control_decoder,batch_input_shape=(batch_size,lookback+lookahead,num_actuators),name='ctrl_decoder_time_dist')(u)
    
    regularization = kwargs.get('regularization',{'l1A': 0,'l2A': 0,'l1B': 0,'l2B': 0})
    x0 = Reshape((state_latent_dim,), name='x0')(
        Cropping1D((0, lookahead))(x))
    x1 = Cropping1D((1, 0), name='x1')(x)
    u = Cropping1D((lookback, 0), name='u')(u)

    AB = SimpleRNN(units=state_latent_dim,
                   activation='linear',
                   use_bias=False,
                   name='AB_matrices',
                   kernel_regularizer=keras.regularizers.l1_l2(
                       l1=regularization['l1B'], l2=regularization['l2B']),
                   recurrent_regularizer=keras.regularizers.l1_l2(
                       l1=regularization['l1A'], l2=regularization['l2A']),
                   return_sequences=True)

    x1est = Reshape((lookahead, state_latent_dim),
                    name='x1est')(AB(u, initial_state=x0))
    x1_residual = subtract([x1, x1est], name='linear_system_residual')
    u_res = subtract([ui, uo], name='u_residual')
    x_res = subtract([xi, xo], name='x_residual')
    model = Model(inputs=state_input_model.inputs +
                  control_input_model.inputs, outputs=[u_res, x_res, x1_residual])
    return model
