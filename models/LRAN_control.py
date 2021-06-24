import keras
from keras import layers
from keras.models import Model    
from keras import regularizers
from helpers.custom_init import downsample

def get_state_joiner(profile_names, scalar_names, timesteps, profile_length,
                     batch_size=None):
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
    profile_inputs = [layers.Input(batch_shape=(batch_size, timesteps, profile_length),
                                  name='input_' + nm) for nm in profile_names]
    profiles = layers.Concatenate(axis=-1,name='profile_joiner')(profile_inputs) if num_profiles>1 else profile_inputs[0]
    scalar_inputs = [layers.Input(batch_shape=(batch_size,timesteps, 1), name='input_' + nm)
                     for nm in scalar_names]
    scalars = layers.Concatenate(
        axis=-1)(scalar_inputs) if num_scalars > 1 else scalar_inputs[0] if num_scalars > 0 else []
    x = layers.Concatenate(axis=-1,name='state_joiner')([profiles, scalars]
                                 ) if num_scalars > 0 else profiles
    if timesteps <=1:
        x = layers.Reshape((state_dim,))(x)
    joiner_model = Model(inputs=profile_inputs +
                         scalar_inputs, outputs=x, name='state_joiner')
    
    
    return joiner_model

def get_state_joiner_conv(profile_names, timesteps, profile_length, batch_size = None):
    num_profiles = len(profile_names)
    profile_inputs = [layers.Input(batch_shape=(batch_size, timesteps, profile_length),
                                  name='input_' + nm) for nm in profile_names]
    profiles = [layers.Reshape((timesteps,1,profile_length,1))(inputs) for inputs in profile_inputs]
    profiles = layers.Concatenate(axis = -1,
                                  name='profile_joiner_conv')(profile_inputs) if num_profiles > 1 else profile_inputs[0]
    if timesteps <=1:
        profiles = layers.Reshape((1,profile_length,num_profiles))(profiles)
    joiner_conv = Model(inputs = profile_inputs, outputs = profiles, name='state_joiner_conv')
    
    return joiner_conv
        

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

    yi = layers.Input(batch_shape=(batch_size,state_dim))
    y = layers.Reshape((state_dim, 1))(yi)
    profile_outputs = [layers.Reshape((profile_length,),name='output_' + nm)(
        layers.Cropping1D((i*profile_length, state_dim-(i+1)*profile_length))(y)) 
                       for i, nm in enumerate(profile_names)]
    scalar_outputs = [layers.Reshape((1,),name='output_' + scalar_names[i])(
        layers.Cropping1D((num_profiles*profile_length + i, state_dim-num_profiles*profile_length - (i+1)))(y))
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
    actuator_inputs = [layers.Input(batch_shape=(batch_size,timesteps, 1), name='input_' + nm)
                       for nm in actuator_names]
    actuators = layers.Concatenate(
        axis=-1,name='control_joiner')(actuator_inputs) if num_actuators > 1 else actuator_inputs[0]
    if timesteps<= 1:
        actuators = layers.Reshape((num_actuators,))(actuators)
    joiner = Model(inputs=actuator_inputs,
                   outputs=actuators, name='control_joiner')

    return joiner

def get_control_splitter(actuator_names, timesteps,batch_size=None):
    """Create models to split and join inputs/outputs for control variables

    Args:
        actuator_names (str): List of names of actuators
        timesteps (int): how many timesteps in the future to predict + how many timesteps of previous actuators

    Returns:
        splitter (model): model that takes combined tensor and returns individual tensors for each signal
    """
    num_actuators = len(actuator_names)

    ui = layers.Input(batch_shape=(batch_size,num_actuators))
    u = layers.Reshape((num_actuators, 1))(ui)
    actuator_outputs = [layers.Reshape((1, 1),name='output_' + nm)(layers.Cropping1D(
        (i, num_actuators-(i+1)))(u)) for i, nm in enumerate(actuator_names)]
    splitter = Model(inputs=ui, outputs=actuator_outputs,
                     name='control_splitter')
    return splitter

def get_state_encoder(profile_names, scalar_names,
                      profile_length, state_latent_dim, std_activation,
                      batch_size=None, **kwargs):
    """
    State encoder using dense network.

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
    #x = layers.GaussianNoise(1.5)(x)
    # initializer = downsample()
    for i in range(num_layers):
        units = int(state_dim + (state_latent_dim-state_dim)
                    * ((1+i)/(num_layers))**layer_scale)
        '''
        if i==num_layers-1:
            x = layers.Dense(units = units, activation = std_activation, 
                             use_bias = True, kernel_initializer = initializer,
                             kernel_regularizer = groupLasso(1e-4, units))(x)
        else:
            x = layers.Dense(units=units, activation=std_activation, 
                             use_bias=True, kernel_initializer = initializer)(x)
        '''
        x = layers.Dense(units = units, activation = std_activation, 
                         use_bias = True)(x)
        #x = layers.Dropout(rate = 0.2)(x)
    x = layers.Reshape((state_latent_dim,))(x)
    state_encoder = Model(inputs=joiner.inputs, outputs=x,
                             name='dense_state_encoder')
    return state_encoder

def get_state_conv_encode(profile_names, scalar_names,
                      profile_length, state_latent_dim, std_activation,
                      batch_size=None, **kwargs):
    num_layers = kwargs.get('num_layers', 1)
    #TODO: make these into specifiable parameters
    max_channels = kwargs.get('max_channels', 80)
    kernel_init = kwargs.get('kernel_initializer','lecun_normal')
    bias_init = kwargs.get('bias_initializer','zeros')
    l2 = 1e-4 
    num_profiles = len(profile_names)
    joiner = get_state_joiner_conv(profile_names, 1, profile_length, batch_size)
    profiles = joiner(joiner.inputs)
    profiles = layers.Conv2D(filters=int(num_profiles*max_channels/8), kernel_size=(1, int(profile_length/12)),
                      strides=(1, 1), padding='same', activation=std_activation,
                      kernel_regularizer= regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    profiles = layers.Conv2D(filters=int(num_profiles*max_channels/4), kernel_size=(1, int(profile_length/8)),
                      strides=(1, 1), padding='same', activation=std_activation,
                     kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                     kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    profiles = layers.Conv2D(filters=int(num_profiles*max_channels/2), kernel_size=(1, int(profile_length/6)),
                      strides=(1, 1), padding='same', activation=std_activation,
                     kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                     kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    profiles = layers.Conv2D(filters=int(num_profiles*max_channels), kernel_size=(1, int(profile_length/4)),
                      strides=(1, 1), padding='same', activation=std_activation,
                     kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                     kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    enc = layers.Flatten()(profiles)
    
    #Tensors should be of shape (batch_size, profile_length*(num_profiles * max_channels))
    for i in range(num_layers):
        units = int((profile_length*max_channels*num_profiles) + ((i+1)/(num_layers))*(state_latent_dim - (profile_length * max_channels*num_profiles)))
        enc = layers.Dense(units = units, use_bias = True, activation = std_activation)(enc)
        
    conv_model = Model(inputs = joiner.inputs, outputs = enc, name = 'state_conv')
    return conv_model

def get_state_convT_decode(profile_names, scalar_names, profile_length,
                            state_latent_dim, std_activation, batch_size=None, **kwargs):
    num_layers = kwargs.get('num_layers', 1)
    num_profiles = len(profile_names)
    max_channels = kwargs.get('max_channels',80)
    kernel_init = kwargs.get('kernel_initializer','lecun_normal')
    bias_init = kwargs.get('bias_initializer','zeros')
    l2 = 1e-4 
    
    x_input = layers.Input(batch_shape=(batch_size,state_latent_dim))
    enc = x_input
    
    for i in range(num_layers):
        units = int((state_latent_dim) + ((i+1)/(num_layers))*((profile_length*max_channels*num_profiles) - state_latent_dim))
        enc = layers.Dense(units = units, use_bias=True, activation = std_activation)(enc)
    
    # Tensor should be reshaped to (batch_size, 1, profile_length, max_channels)
    profiles = layers.Reshape((1, profile_length, num_profiles*max_channels))(enc)
    profiles = layers.Conv2DTranspose(filters=int(num_profiles*max_channels/2), kernel_size=(1, int(profile_length/6)),
                      strides=(1, 1), padding='same', activation=std_activation,
                     kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                     kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    profiles = layers.Conv2DTranspose(filters=int(num_profiles*max_channels/4), kernel_size=(1, int(profile_length/8)),
                      strides=(1, 1), padding='same', activation=std_activation,
                     kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                     kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    profiles = layers.Conv2DTranspose(filters=int(num_profiles*max_channels/8), kernel_size=(1, int(profile_length/12)),
                      strides=(1, 1), padding='same', activation=std_activation,
                      kernel_regularizer= regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    profiles = layers.Conv2DTranspose(filters=num_profiles, kernel_size =(1, int(profile_length/12)),
                             strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(l2),
                                      bias_regularizer=regularizers.l2(l2), activation = std_activation,
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(profiles)
    # Should be (batch_size, 165)
    out = layers.Flatten()(profiles)
    #out = get_state_splitter(profile_names, [], 1, profile_length, batch_size=batch_size)(out)
    # Returns list of tensors of shape (batch_size, 33)
    
    convT_model = Model(inputs= x_input, outputs = out, name = 'state_convT')
    return convT_model

def get_state_decoder(profile_names, scalar_names, profile_length,
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
    state_dim = num_profiles*profile_length
    xi = layers.Input(batch_shape=(batch_size,state_latent_dim))
    x = xi
    #x = layers.GaussianNoise(5)(x)
    #initializer = downsample()
    for i in range(num_layers-1):
        units = int(state_latent_dim + (state_dim-state_latent_dim)
                    * ((1+i)/(num_layers))**layer_scale)
        '''
        if i == 0:
            x = layers.Dense(units = units, activation = std_activation, 
                             use_bias = True, kernel_initializer = initializer,
                             kernel_regularizer = groupLasso(1e-4, units))(x)
        else:
            x = layers.Dense(units=units, activation=std_activation, 
                             use_bias=True, kernel_initializer = initializer)(x)
        '''
        x = layers.Dense(units=units, activation=std_activation, 
                             use_bias=True)(x)

        #x = layers.Dropout(rate = 0.2)(x)
    # y = layers.Dense(units=state_dim, activation=std_activation)(x)
    x = layers.Dense(units = state_dim, activation = 'linear', use_bias = True)(x)
    outputs = layers.Reshape((state_dim,))(x)
    decoder = Model(inputs=xi, outputs=outputs, name='dense_state_decoder')
    return decoder



def get_control_encoder(actuator_names, control_latent_dim,
                              std_activation, batch_size=None,**kwargs):
    """
    Control encoder using dense network.

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
    assert num_layers > 0 
    if num_layers > 1:
        for i in range(num_layers):
            units = int(num_actuators + (control_latent_dim-num_actuators)
                        * ((i+1)/(num_layers))**layer_scale)
            u = layers.Dense(units=units, activation=std_activation, use_bias=False)(u)
        u = layers.Reshape((control_latent_dim,))(u)
        encoder = Model(inputs=joiner.inputs, outputs=u,
                        name='dense_control_encoder')
    else:
        u = layers.Dense(units=control_latent_dim, activation=std_activation, 
                         kernel_initializer='identity', use_bias = False)(u)
        u = layers.Reshape((control_latent_dim,))(u)
        encoder = Model(inputs=joiner.inputs, outputs=u,
                        name='dense_control_encoder')
    return encoder



def get_control_decoder(actuator_names, control_latent_dim,
                              std_activation, batch_size=None, **kwargs):
    """
    Control decoder using dense network.
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
    ui = layers.Input(batch_shape=(batch_size,control_latent_dim))
    u = ui
    assert num_layers > 0
    if num_layers>1:
        for i in range(num_layers-1):
            units = int(control_latent_dim + (num_actuators-control_latent_dim)
                        * ((i+1)/(num_layers))**layer_scale)
            u = layers.Dense(units=units, activation=std_activation, use_bias=False)(u)
        u = layers.Dense(units=num_actuators, activation='linear')(u)
        outputs = layers.Reshape((num_actuators,))(u)
        decoder = Model(inputs=ui, outputs=outputs, name='dense_control_decoder')
    else:
        u = layers.Dense(units=num_actuators, activation=std_activation, 
                         kernel_initializer='identity', use_bias=False)(u)
        outputs = layers.Reshape((num_actuators,))(u)
        decoder = Model(inputs=ui, outputs=outputs, name='dense_control_decoder')
    return decoder


def make_LRAN(state_encoder_type, state_decoder_type, control_encoder_type, control_decoder_type, 
              state_encoder_kwargs, state_decoder_kwargs, control_encoder_kwargs, control_decoder_kwargs, 
              profile_names, scalar_names, actuator_names, state_latent_dim, control_latent_dim, 
              profile_length, lookback, lookahead, batch_size=None, **kwargs):
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
    max_channels = state_encoder_kwargs['max_channels']
    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    num_actuators = len(actuator_names)
    state_dim = num_profiles*profile_length + num_scalars
    state_encoders = {'dense': get_state_encoder, 'conv': get_state_conv_encode}
    state_decoders = {'dense': get_state_decoder, 'conv': get_state_convT_decode}
    control_encoders = {'dense': get_control_encoder}
    control_decoders = {'dense': get_control_decoder}
    
    # Get models, joiners, splitters
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
    
    state_splitter = get_state_splitter(profile_names,
                                        scalar_names,
                                        1,
                                        profile_length,
                                        batch_size=batch_size)
  
    control_input_model = get_control_joiner(actuator_names,
                                             lookback+lookahead,
                                             batch_size=batch_size)
    
    control_splitter = get_control_splitter(actuator_names,
                                            1,
                                            batch_size=batch_size)
  
    # Get data for each window
    x_input = state_input_model.outputs[0]
    u_input = control_input_model.outputs[0]
    x_input_future = layers.Cropping1D((1,0), name='x_input_future')(x_input)
    x_input_future = layers.Reshape((lookahead, 1, state_dim))(x_input_future)
    x_input_future = layers.Cropping2D(((0,0),(0,num_scalars)), data_format='channels_first')(x_input_future)
    x_input_future = layers.Reshape((lookahead, (state_dim-num_scalars)))(x_input_future)

    # Create Encoders that take combined tensor as input (TimeDistributed only allows 1 input)
    state_encoder = Model(state_splitter.inputs, state_encoder(state_splitter.outputs))
    control_encoder = Model(control_splitter.inputs, control_encoder(control_splitter.outputs))
    
    # State and Control latent space representation
    x = layers.TimeDistributed(state_encoder,
                               batch_input_shape=(batch_size, lookahead+1,state_dim),
                               name='state_encoder_time_dist')(x_input)
    u = layers.TimeDistributed(control_encoder,
                               batch_input_shape=(batch_size,
                                                  lookahead+lookback,num_actuators),
                               name='ctrl_encoder_time_dist')(u_input)

    u_out = layers.TimeDistributed(control_decoder,
                                   batch_input_shape=(batch_size,
                                                      lookahead+lookback,control_latent_dim),
                                   name='ctrl_decoder_time_dist')(u)

    
    # Initial latent state and future latent state
    latent_initial = layers.Reshape((state_latent_dim,),name='x_initial')(
        layers.Cropping1D((0,lookahead))(x))
    latent_future = layers.Cropping1D((1,0), name='x_future')(x)
    
    # Control "proposals" in latent space and linear system evolution
    u = layers.Cropping1D((lookback,0), name='u')(u)
    regularization = kwargs.get('regularization',{'l1A': 0,'l2A': 0,'l1B': 0,'l2B': 0})
    Koopman = layers.SimpleRNN(units = state_latent_dim,
                               activation = 'linear',
                               use_bias = False,
                               name = 'AB_matrices',
                               kernel_regularizer=keras.regularizers.l1_l2(
                                   l1=regularization['l1B'], l2=regularization['l2B']),
                               recurrent_regularizer=keras.regularizers.l1_l2(
                                   l1=regularization['l1A'], l2=regularization['l2A']),
                               return_sequences=True)
    latent_future_predict = layers.Reshape((lookahead, state_latent_dim),name='x1est')(
        Koopman(u, initial_state=latent_initial))

    # Decode evolution 
    x_out_future = layers.TimeDistributed(state_decoder,
                                          name='state_decoder_time_dist')(latent_future_predict)
    
    # Residuals
    latent_residual = layers.subtract([latent_future, latent_future_predict],
                                      name='linear_system_residual')
    x_res = layers.subtract([x_input_future, x_out_future], name='x_residual')
    u_res = layers.subtract([u_out, u_input], name='u_residual')
    
    
    model = Model(inputs=state_input_model.inputs+control_input_model.inputs,
                        outputs = [x_res, u_res, latent_residual])
    
    return model
