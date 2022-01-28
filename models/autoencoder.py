import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Cropping1D,
    Dot,
    SimpleRNN,
    add,
    subtract,
    Concatenate,
    Reshape,
    Permute,
    BatchNormalization,
    ELU,
    ReLU,
    LeakyReLU,
)
from tensorflow.keras.models import Model
from helpers.custom_layers import (
    MultiTimeDistributed,
    InverseBatchNormalization,
    InverseDense,
)
from helpers.custom_activations import InverseLeakyReLU
from helpers.custom_constraints import SoftOrthonormal, Orthonormal, Invertible

activations = {
    "relu": ReLU(),
    "elu": ELU(),
    "leaky_relu": LeakyReLU(),
    "inv_leaky_relu": InverseLeakyReLU(),
}


def inverse_layer(layer):
    if isinstance(layer, Dense):
        return InverseDense(layer)
    elif isinstance(layer, BatchNormalization):
        return InverseBatchNormalization(layer)
    else:
        raise ValueError("No inverse defined for layer {}".format(layer))


def get_state_input_model(
    profile_names, scalar_names, timesteps, profile_length, batch_size
):
    """Create models to split and join inputs/outputs for state variables

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): list of names of scalars
        timesteps (int): how many timesteps in the future to predict
        profile_length (int): number of psi pts in discretized profiles

    Returns:
        input (model): model that takes individual inputs and returns normlized values
    """

    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles * profile_length + num_scalars

    profile_inputs = [
        Input(batch_shape=(batch_size, timesteps, profile_length), name="input_" + nm)
        for nm in profile_names
    ]
    scalar_inputs = [
        Input(batch_shape=(batch_size, timesteps, 1), name="input_" + nm)
        for nm in scalar_names
    ]

    norms = [
        BatchNormalization(center=False, scale=False, name="norm_" + nm)(x)
        for x, nm in zip(profile_inputs + scalar_inputs, profile_names + scalar_names)
    ]
    input_model = Model(
        inputs=profile_inputs + scalar_inputs, outputs=norms, name="state_input"
    )

    return input_model


def get_control_input_model(actuator_names, timesteps, batch_size):
    """Create models to split and join inputs/outputs for control variables

    Args:
        actuator_names (str): list of names of actuators
        timesteps (int): how many timesteps in the future to predict

    Returns:
        input (model): model that takes individual inputs and returns normlized values
    """

    actuator_inputs = [
        Input(batch_shape=(batch_size, timesteps, 1), name="input_" + nm)
        for nm in actuator_names
    ]
    norms = [
        BatchNormalization(center=False, scale=False, name="norm_" + nm)(x)
        for x, nm in zip(actuator_inputs, actuator_names)
    ]
    input_model = Model(inputs=actuator_inputs, outputs=norms, name="control_input")

    return input_model


def get_state_joiner(profile_names, scalar_names, profile_length, batch_size):
    """Create models to split and join inputs/outputs for state variables

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): list of names of scalars
        profile_length (int): number of psi pts in discretized profiles

    Returns:
        joiner (model): model that takes individual inputs and returns a combined tensor
    """

    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles * profile_length + num_scalars

    profile_inputs = [
        Input(batch_shape=(batch_size, profile_length), name="input_" + nm)
        for nm in profile_names
    ]
    profiles = (
        Concatenate(axis=-1, name="profile_joiner")(profile_inputs)
        if num_profiles > 1
        else profile_inputs[0]
    )
    scalar_inputs = [
        Input(batch_shape=(batch_size, 1), name="input_" + nm) for nm in scalar_names
    ]
    scalars = (
        Concatenate(axis=-1)(scalar_inputs)
        if num_scalars > 1
        else scalar_inputs[0]
        if num_scalars > 0
        else []
    )
    x = (
        Concatenate(axis=-1, name="state_joiner")([profiles, scalars])
        if num_scalars > 0
        else profiles
    )
    joiner_model = Model(
        inputs=profile_inputs + scalar_inputs, outputs=x, name="state_joiner"
    )

    return joiner_model


def get_state_splitter(profile_names, scalar_names, profile_length, batch_size):
    """Create models to split and join inputs/outputs for state variables

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): list of names of scalars
        profile_length (int): number of psi pts in discretized profiles

    Returns:
        splitter (model): model that takes combined tensor and returns individual tensors for each signal
    """

    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles * profile_length + num_scalars

    yi = Input(batch_shape=(batch_size, state_dim))
    y = Reshape((state_dim, 1))(yi)
    profile_outputs = [
        Reshape((profile_length,), name="output_" + nm)(
            Cropping1D((i * profile_length, state_dim - (i + 1) * profile_length))(y)
        )
        for i, nm in enumerate(profile_names)
    ]
    scalar_outputs = [
        Reshape((1,), name="output_" + scalar_names[i])(
            Cropping1D(
                (
                    num_profiles * profile_length + i,
                    state_dim - num_profiles * profile_length - (i + 1),
                )
            )(y)
        )
        for i, nm in enumerate(scalar_names)
    ]
    splitter_model = Model(
        inputs=yi, outputs=profile_outputs + scalar_outputs, name="state_splitter"
    )
    return splitter_model


def get_control_joiner(actuator_names, batch_size):
    """Create models to split and join inputs/outputs for control variables

    Args:
        actuator_names (str): List of names of actuators

    Returns:
        joiner (model): model that takes individual inputs and returns a combined tensor
    """
    num_actuators = len(actuator_names)
    actuator_inputs = [
        Input(batch_shape=(batch_size, 1), name="input_" + nm) for nm in actuator_names
    ]
    actuators = (
        Concatenate(axis=-1, name="control_joiner")(actuator_inputs)
        if num_actuators > 1
        else actuator_inputs[0]
    )
    joiner = Model(inputs=actuator_inputs, outputs=actuators, name="control_joiner")

    return joiner


def get_control_splitter(actuator_names, batch_size):
    """Create models to split and join inputs/outputs for control variables

    Args:
        actuator_names (str): List of names of actuators

    Returns:
        splitter (model): model that takes combined tensor and returns individual tensors for each signal
    """
    num_actuators = len(actuator_names)

    ui = Input(batch_shape=(batch_size, num_actuators))
    u = Reshape((num_actuators, 1))(ui)
    actuator_outputs = [
        Reshape((1,), name="output_" + nm)(Cropping1D((i, num_actuators - (i + 1)))(u))
        for i, nm in enumerate(actuator_names)
    ]
    splitter = Model(inputs=ui, outputs=actuator_outputs, name="control_splitter")
    return splitter


def get_control_encoder_dense(actuator_names, control_latent_dim, batch_size, **kwargs):
    """Control encoder/decoder using dense network.

    Args:
        actuator_names (str): List of names of actuators
        control_latent_dim (int): dimensionality of the encoded variables
        batch_size (int) : number of samples per minibatch
        kwargs:
        std_activation (str or callable or tuple): activation function to apply to
            hidden layers. If a list or tuple, uses first element for encoder, 2nd for decoder
        num_layers (int or tuple of int): number of hidden layers, for encoder and decoder
        layer_scale (float or tuple of float): power law scaling for size of hidden layers
            size of layer(i) = min_size + (max_size-min_size)*(i/num_layers)**layer_scale

    Returns:
        control_encoder (model): Keras model that takes each actuator as individual inputs
            and returns a single tensor of the encoded values.
        control_decoder (model): Keras model that takes a single tensor of the
            encoded values and returns each actuator as individual outputs.

    """
    layer_scale = kwargs.pop("layer_scale", [1, 1])
    if not isinstance(layer_scale, (list, tuple)):
        layer_scale = [layer_scale]
    num_layers = kwargs.pop("num_layers", [6, 6])
    if not isinstance(num_layers, (list, tuple)):
        num_layers = [num_layers]
    std_activation = kwargs.pop("std_activation", ["elu", "elu"])
    if not isinstance(std_activation, (list, tuple)):
        std_activation = [std_activation]
    num_actuators = len(actuator_names)

    joiner = get_control_joiner(actuator_names, batch_size)
    u = joiner(joiner.inputs)
    for i in range(num_layers[0]):
        units = int(
            control_latent_dim
            + (num_actuators - control_latent_dim)
            * ((num_layers[0] - i - 1) / (num_layers[0] - 1)) ** layer_scale[0]
        )
        u = Dense(units=units, activation=std_activation[0], use_bias=True, **kwargs)(u)
    encoder = Model(inputs=joiner.inputs, outputs=u, name="dense_control_encoder")

    ui = Input(batch_shape=(batch_size, control_latent_dim))
    u = ui
    for i in range(num_layers[-1] - 1):
        units = int(
            num_actuators
            - (num_actuators - control_latent_dim)
            * ((num_layers[-1] - i - 1) / (num_layers[-1] - 1)) ** layer_scale[-1]
        )
        u = Dense(units=units, activation=std_activation[-1], use_bias=True, **kwargs)(
            u
        )
    u = Dense(units=num_actuators, activation="linear", **kwargs)(u)
    splitter = get_control_splitter(actuator_names, batch_size)
    outputs = splitter(u)
    decoder = Model(inputs=ui, outputs=outputs, name="dense_control_decoder")

    return encoder, decoder


def get_control_encoder_none(actuator_names, control_latent_dim, batch_size, **kwargs):

    """Control encoder/decoder using identity transformation.

    Args:
        actuator_names (str): List of names of actuators
        control_latent_dim (int): dimensionality of the encoded variables
        batch_size (int) : number of samples per minibatch
        kwargs:

    Returns:
        control_encoder (model): Keras model that takes each actuator as individual inputs
            and returns a single tensor of the encoded values.
        control_decoder (model): Keras model that takes a single tensor of the
            encoded values and returns each actuator as individual outputs.

    """

    assert len(actuator_names) == control_latent_dim
    num_actuators = len(actuator_names)
    joiner = get_control_joiner(actuator_names, batch_size)
    u = joiner(joiner.inputs)
    encoder = Model(inputs=joiner.inputs, outputs=u, name="none_control_encoder")

    splitter = get_control_splitter(actuator_names, batch_size)
    ui = Input(batch_shape=(batch_size, control_latent_dim))
    u = ui
    u = Reshape((num_actuators,))(u)
    outputs = splitter(u)
    decoder = Model(inputs=ui, outputs=outputs, name="none_control_decoder")

    return encoder, decoder


def get_state_encoder_dense(
    profile_names, scalar_names, profile_length, state_latent_dim, batch_size, **kwargs
):
    """State encoder/decoder using dense network.

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): List of names of scalars
        profile_length (int): number of psi pts in discretized profiles
        state_latent_dim (int): dimensionality of the encoded variables
        batch_size (int) : number of samples per minibatch
        kwargs:
        activation (str or fn or tuple): activation function to apply to hidden layers
        num_layers (int or tuple): number of hidden layers, for encoder and decoder
        layer_scale (float or tuple): power law scaling for size of hidden layers
            size of layer(i) = min_size + (max_size-min_size)*(i/num_layers)**layer_scale

    Returns:
        state_encoder (model): Keras model that takes each profile and scalar as individual inputs
            and returns a single tensor of the encoded values.
        state_decoder (model): Keras model that takes a single tensor of the
            encoded values and returns each profile/scalar as individual outputs.
    """
    layer_scale = kwargs.pop("layer_scale", [1, 1])
    if not isinstance(layer_scale, (list, tuple)):
        layer_scale = [layer_scale]
    num_layers = kwargs.pop("num_layers", [6, 6])
    if not isinstance(num_layers, (list, tuple)):
        num_layers = [num_layers]
    activation = kwargs.pop("std_activation", ["elu", "elu"])
    if not isinstance(activation, (list, tuple)):
        activation = [activation]
    norm = kwargs.pop("norm", False)
    for i, act in enumerate(activation):
        if act in activations:
            activation[i] = activations[act]

    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles * profile_length + num_scalars

    joiner = get_state_joiner(profile_names, scalar_names, profile_length, batch_size)
    x = joiner(joiner.inputs)
    for i in range(num_layers[0]):
        units = int(
            state_latent_dim
            + (state_dim - state_latent_dim)
            * ((num_layers[0] - i - 1) / (num_layers[0] - 1)) ** layer_scale[0]
        )
        x = Dense(
            units=units,
            activation=activation[0],
            **kwargs,
        )(x)
    if norm:
        x = BatchNormalization(center=False, scale=False, name="latent_state_norm")(x)
    encoder = Model(inputs=joiner.inputs, outputs=x, name="dense_state_encoder")

    xi = Input(batch_shape=(batch_size, state_latent_dim))
    x = xi
    for i in range(num_layers[-1] - 1, 0, -1):
        units = int(
            state_latent_dim
            + (state_dim - state_latent_dim)
            * ((num_layers[-1] - i - 1) / (num_layers[-1] - 1)) ** layer_scale[-1]
        )
        x = Dense(
            units=units,
            activation=activation[-1],
            **kwargs,
        )(x)
    y = Dense(units=state_dim, activation=None, **kwargs)(x)
    splitter = get_state_splitter(
        profile_names, scalar_names, profile_length, batch_size
    )
    outputs = splitter(y)
    decoder = Model(inputs=xi, outputs=outputs, name="dense_state_decoder")
    return encoder, decoder


def get_state_encoder_invertible(
    profile_names, scalar_names, profile_length, state_latent_dim, batch_size, **kwargs
):
    """State encoder/decoder using invertible transformations.

    Args:
        profile_names (str): List of names of profiles
        scalar_names (str): List of names of scalars
        profile_length (int): number of psi pts in discretized profiles
        state_latent_dim (int): dimensionality of the encoded variables
        batch_size (int) : number of samples per minibatch
        kwargs:
        activation (str or fn or tuple): activation function to apply to hidden layers
        num_layers (int or tuple): number of hidden layers, for encoder and decoder
        layer_scale (float or tuple): power law scaling for size of hidden layers
            size of layer(i) = min_size + (max_size-min_size)*(i/num_layers)**layer_scale

    Returns:
        state_encoder (model): Keras model that takes each profile and scalar as individual inputs
            and returns a single tensor of the encoded values.
        state_decoder (model): Keras model that takes a single tensor of the
            encoded values and returns each profile/scalar as individual outputs.
    """
    num_profiles = len(profile_names)
    num_scalars = len(scalar_names)
    state_dim = num_profiles * profile_length + num_scalars
    assert state_dim == state_latent_dim

    _ = kwargs.pop("layer_scale", None)
    num_layers = kwargs.pop("num_layers", 6)
    if isinstance(num_layers, (list, tuple)):
        num_layers = num_layers[0]
    activation = kwargs.pop("std_activation", "leaky_relu")
    if isinstance(activation, (list, tuple)):
        activation = activation[0]
    if activation in activations:
        activation = activations[activation]
    norm = kwargs.pop("norm", True)

    kwargs.setdefault("kernel_initializer", "orthogonal")
    kwargs.setdefault("kernel_constraint", Orthonormal())
    kwargs.setdefault("bias_regularizer", "l2")

    joiner = get_state_joiner(profile_names, scalar_names, profile_length, batch_size)
    x = joiner(joiner.inputs)

    layers = []
    for i in range(num_layers):
        layer = Dense(
            units=state_latent_dim,
            activation=activation,
            **kwargs,
        )
        layers.append(layer)
    if norm:
        layers.append(
            BatchNormalization(center=False, scale=False, name="latent_state_norm")
        )
    for layer in layers:
        x = layer(x)

    z = Input(state_latent_dim)
    zi = z
    for layer in layers[::-1]:
        zi = inverse_layer(layer)(zi)

    splitter = get_state_splitter(
        profile_names, scalar_names, profile_length, batch_size
    )
    outputs = splitter(zi)
    encoder = Model(inputs=joiner.inputs, outputs=x, name="invertible_state_encoder")
    decoder = Model(inputs=z, outputs=outputs, name="invertible_state_decoder")
    return encoder, decoder


def get_latent_linear_model(
    state_latent_dim, control_latent_dim, lookahead, batch_size, regularization=None
):
    """Linear model for encoded variables

    Args:
        state_latent_dim (int): dimensionality of the encoded state variables
        control_latent_dim (int): dimensionality of the encoded control variables
        lookahead (int): how many timesteps in the future to predict
        batch_size (int) : number of samples per minibatch
        regularization (dict): Dictionary of L1 and L2 regularization parameters
            for A and B matrices. keys 'l1A', l2A', 'l1B', 'l2B'

    Returns:
        latent_linear_system (model): Keras model that takes state and control
            tensors as input and returns a tensor of residual values in the
            linear dynamic approximation
    """
    if regularization is None:
        regularization = {"l1A": 0, "l2A": 0, "l1B": 0, "l2B": 0}
    xi = Input(batch_shape=(batch_size, lookahead + 1, state_latent_dim), name="xi")
    ui = Input(batch_shape=(batch_size, lookahead, control_latent_dim), name="ui")
    x0 = Reshape((state_latent_dim,), name="x0")(Cropping1D((0, lookahead))(xi))
    x1 = Cropping1D((1, 0), name="x1")(xi)
    u = Cropping1D((0, 1), name="u")(ui)

    AB = SimpleRNN(
        units=state_latent_dim,
        activation="linear",
        use_bias=False,
        name="AB_matrices",
        kernel_regularizer=keras.regularizers.l1_l2(
            l1=regularization["l1B"], l2=regularization["l2B"]
        ),
        recurrent_regularizer=keras.regularizers.l1_l2(
            l1=regularization["l1A"], l2=regularization["l2A"]
        ),
        return_sequences=True,
    )

    x1est = Reshape((lookahead, state_latent_dim), name="x1est")(
        AB(u, initial_state=x0)
    )
    x1_residual = subtract([x1, x1est], name="linear_system_residual")
    model = Model(inputs=[xi, ui], outputs=x1_residual, name="latent_linear_system")
    return model


def make_autoencoder(
    state_encoder_type,
    control_encoder_type,
    state_encoder_kwargs,
    control_encoder_kwargs,
    recurrent_kwargs,
    profile_names,
    scalar_names,
    actuator_names,
    state_latent_dim,
    control_latent_dim,
    profile_length,
    lookahead,
    batch_size=None,
):
    """Linear Recurrent autoencoder

    Args:
        state_encoder_type (str): Type of netork to use for state encoding
        control_encoder_type (str): Type of netork to use for control encoding
        state_encoder_kwargs (dict): Dictionary of keyword arguments for state encoder model
        control_encoder_kwargs (dict): Dictionary of keyword arguments for control encoder model
        recurrent_kwargs (dict): Dictionary of keyword arguments for latent linear model
        profile_names (str): List of names of profiles
        scalar_names (str): list of names of scalars
        actuator_names (str): list of names of actuators
        state_latent_dim (int): dimensionality of the encoded state variables
        control_latent_dim (int): dimensionality of the encoded control variables
        profile_length (int): number of psi pts in discretized profiles
        lookahead (int): how many timesteps in the future to predict
        batch_size (int) : how many samples per minibatch


    Returns:
        autoencoder (model): Keras model that takes profile, scalar, and actuator
            tensors as input and returns a tensor of residual values in the
            state reconstruction, control reconstruction, and linear dynamic approximation
    """

    state_encoders = {
        "dense": get_state_encoder_dense,
        "invertible": get_state_encoder_invertible,
    }
    control_encoders = {
        "dense": get_control_encoder_dense,
        "none": get_control_encoder_none,
    }

    profile_inputs = [
        Input(
            batch_shape=(batch_size, lookahead + 1, profile_length), name="input_" + nm
        )
        for nm in profile_names
    ]
    scalar_inputs = [
        Input(batch_shape=(batch_size, lookahead + 1, 1), name="input_" + nm)
        for nm in scalar_names
    ]
    actuator_inputs = [
        Input(batch_shape=(batch_size, lookahead + 1, 1), name="input_" + nm)
        for nm in actuator_names
    ]

    state_encoder, state_decoder = state_encoders[state_encoder_type](
        profile_names,
        scalar_names,
        profile_length,
        state_latent_dim,
        batch_size=batch_size,
        **state_encoder_kwargs,
    )
    control_encoder, control_decoder = control_encoders[control_encoder_type](
        actuator_names,
        control_latent_dim,
        batch_size=batch_size,
        **control_encoder_kwargs,
    )
    state_input = get_state_input_model(
        profile_names,
        scalar_names,
        lookahead + 1,
        profile_length,
        batch_size=batch_size,
    )
    control_input = get_control_input_model(
        actuator_names, lookahead + 1, batch_size=batch_size
    )

    xi = state_input(profile_inputs + scalar_inputs)
    ui = control_input(actuator_inputs)

    zi = MultiTimeDistributed(state_encoder, name="state_encoder_time_dist")(xi)
    xo = MultiTimeDistributed(state_decoder, name="state_decoder_time_dist")(zi)

    vi = MultiTimeDistributed(control_encoder, name="ctrl_encoder_time_dist")(ui)
    uo = MultiTimeDistributed(control_decoder, name="ctrl_decoder_time_dist")(vi)

    z0 = Reshape((state_latent_dim,), name="x0")(Cropping1D((0, lookahead))(zi))
    z1 = Cropping1D((1, 0), name="x1")(zi)
    vi = Cropping1D((0, 1), name="u")(vi)

    AB = SimpleRNN(
        units=state_latent_dim,
        activation="linear",
        use_bias=False,
        name="AB_matrices",
        return_sequences=True,
        **recurrent_kwargs,
    )

    z1est = Reshape((lookahead, state_latent_dim), name="x1est")(
        AB(vi, initial_state=z0)
    )
    x1_residual = subtract([z1, z1est], name="linear_system_residual")
    xicat = Concatenate()(xi)
    xocat = Concatenate()(xo)
    uicat = Concatenate()(ui)
    uocat = Concatenate()(uo)
    x_res = subtract([xicat, xocat], name="x_residual")
    u_res = subtract([uicat, uocat], name="u_residual")
    model = Model(
        inputs=profile_inputs + scalar_inputs + actuator_inputs,
        outputs=[u_res, x_res, x1_residual],
    )
    return model
