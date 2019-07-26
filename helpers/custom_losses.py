import keras.backend as K
from helpers.normalization import denormalize_arr


def denorm_mse_loss(param_dict):
    """Denormed MSE loss

    Denormalizes the signal before evaluating loss function, so that different 
    normalizations may be compared.

    Args:
       param_dict (dict): Dictionary of normalization parameters for the signal.

    Returns:
        lossfn: A loss function that takes in y_true and y_pred and returns 
            a scalar loss value
    """
    def lossfn(y_true, y_pred):
        denorm_y_true = K.variable(denormalize_arr(
            y_true, param_dict), dtype='float32')
        denorm_y_pred = K.variable(denormalize_arr(
            y_pred, param_dict), dtype='float32')
        return K.mean(K.square(denorm_y_pred - denorm_y_true), axis=-1)
    return lossfn


def denorm_mae_loss(param_dict):
    """Denormed MAE loss

    Denormalizes the signal before evaluating loss function, so that different 
    normalizations may be compared.

    Args:
       param_dict (dict): Dictionary of normalization parameters for the signal.

    Returns:
        lossfn: A loss function that takes in y_true and y_pred and returns 
            a scalar loss value
    """
    def lossfn(y_true, y_pred):
        denorm_y_true = K.variable(denormalize_arr(
            y_true, param_dict), dtype='float32')
        denorm_y_pred = K.variable(denormalize_arr(
            y_pred, param_dict), dtype='float32')
        return K.mean(K.abs(denorm_y_pred - denorm_y_true), axis=-1)
    return lossfn


def hinge_mse_loss(sig, model, hinge_weight, mse_weight_vector, predict_deltas):
    """Weighted MSE + hinge loss

    Weighted MSE takes care of large deviations, while hinge loss tries to make
    sure the predicted value is on the right side of the baseline.

    Args:
        sig (str): Name of the signal that the loss is applied to.
        model: Model that is being trained
        hinge_weight (float): Relative weight applied to hinge vs weighted MSE.
        mse_weight_vector (float): Array of weights, same length as profile.
        predict_deltas (bool): Whether the model is predicting deltas or full profiles.

    Returns:
        lossfn: A loss function that takes in y_true and y_pred and returns 
            a scalar loss value
    """
    mse_weight_vector = K.variable(mse_weight_vector, dtype='float32')
    hinge_weight = K.variable(hinge_weight, dtype='float32')
    # get the current input to the model for baseline comparison
    baseline = model.get_layer('input_' + sig).input[:, -1]
    if predict_deltas:
        # if predicting deltas, baseline is zero
        baseline = K.zeros_like(baseline)

    def lossfn(y_true, y_pred):
        delta_true = y_true-baseline
        delta_pred = y_pred-baseline
        mse_loss = K.mean(K.square(y_pred-y_true)*mse_weight_vector, axis=-1)
        hinge_loss = K.mean(K.maximum(-(delta_true * delta_pred), 0.), axis=-1)
        return mse_loss + hinge_weight*hinge_loss
    return lossfn
