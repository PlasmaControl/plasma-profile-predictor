import keras.backend as K
from helpers.normalization import denormalize_arr
import numpy as np


def percent_correct_sign(sig, model, predict_deltas):
    """Wrapper for metric to measure how often the prediction has the correct sign

    Args:
        sig (str): Name of the signal that the loss is applied to.
        model: Model that is being trained
        predict_deltas (bool): Whether the model is predicting deltas or full profiles.

    Returns:
        sign_accuracy: A loss function that takes in y_true and y_pred and returns 
            the percentage of the time the prediction has the correct sign.
    """
    # baseline = predict current value
    if predict_deltas:
        # if predicting deltas, baseline is zero
        baseline = K.cast_to_floatx(0)
    else:
        # get the current input to the model for baseline comparison
        baseline = model.get_layer('input_' + sig).input[:, -1]

    def sgn_acc(y_true, y_pred):
        delta_true = y_true-baseline
        delta_pred = y_pred-baseline
        return K.mean(K.maximum(K.sign(delta_pred*delta_true), 0), axis=-1)
    return sgn_acc

def percent_baseline_error(sig, model, predict_deltas):
    """Wrapper for metric to compare model MAE to baseline MAE

    Args:
        sig (str): Name of the signal that the loss is applied to.
        model: Model that is being trained
        predict_deltas (bool): Whether the model is predicting deltas or full profiles.

    Returns:
        sign_accuracy: A loss function that takes in y_true and y_pred and returns 
            the percentage of the time the prediction has the correct sign.
    """
    if predict_deltas:
        # if predicting deltas, baseline is zero
        baseline = K.cast_to_floatx(0)
    else:
        # get the current input to the model for baseline comparison
        baseline = model.get_layer('input_' + sig).input[:, -1]
    def perBLMAE(y_true, y_pred):
        BLerr = K.maximum(K.mean(K.abs(y_true-baseline),axis=-1), K.cast_to_floatx(K.epsilon()))
        MAE = K.mean(K.abs(y_true-y_pred),axis=-1)
        return K.minimum(MAE/BLerr, 1)
    return perBLMAE
    
def baseline_MAE(sig, model, predict_deltas):
    """Wrapper for metric to measure the accuracy of predicting baseline

    Args:
        sig (str): Name of the signal that the loss is applied to.
        model: Model that is being trained
        predict_deltas (bool): Whether the model is predicting deltas or full profiles.

    Returns:
        BL_MAE: A loss function that takes in y_true and y_pred and returns 
            the percentage error wrt to predicting baseline
    """
    # baseline = predict current value
    if predict_deltas:
        # if predicting deltas, baseline is zero
        baseline = K.cast_to_floatx(0)
    else:
        # get the current input to the model for baseline comparison
        baseline = model.get_layer('input_' + sig).input[:, -1]

    def BL_MAE(y_true, y_pred):
        return K.mean(K.abs(y_true-baseline), axis=-1)
    return BL_MAE


def denorm_loss(sig, model, param_dict, loss, predict_deltas):
    """Wrapper for denormed loss functions

    Denormalizes the signal before evaluating loss function, so that different 
    normalizations may be compared.

    Args:
        sig (str): Name of the signal that the loss is applied to.
        model: Model that is being trained
        param_dict: Dictionary of normalization parameters for the signal.
        loss: Instance of keras loss function to be applied to denormed data.
        predict_deltas (bool): Whether the model is predicting deltas or full profiles.

    Returns:
        denorm_loss: A loss function that takes in y_true and y_pred and returns 
            a scalar loss value
    """
    if predict_deltas:
        baseline = K.cast_to_floatx(0) 
    else:
        baseline = model.get_layer('input_' + sig).input[:, -1]
    method = param_dict['method']
    eps = K.cast_to_floatx(K.epsilon())
    for key, val in param_dict.items():
        if type(val) in [int, float] or type(val).__module__ == 'numpy':
            param_dict[key] = K.cast_to_floatx(val)
    if method == 'StandardScaler':
        def denorm_loss(y_true, y_pred):
            denorm_pred = (y_pred+baseline) * \
                K.maximum(param_dict['std'], eps) + param_dict['mean']
            denorm_true = (y_true+baseline) * \
                K.maximum(param_dict['std'], eps) + param_dict['mean']
            return loss(denorm_true, denorm_pred)
    elif method == 'MinMax':
        def denorm_loss(y_true, y_pred):
            denorm_pred = (y_pred+baseline) * \
                K.maximum(
                    (param_dict['armax']-param_dict['armin']), eps) + param_dict['armin']
            denorm_true = (y_true+baseline) * \
                K.maximum(
                    (param_dict['armax']-param_dict['armin']), eps) + param_dict['armin']
            return loss(denorm_true, denorm_pred)
    elif method == 'MaxAbs':
        def denorm_loss(y_true, y_pred):
            denorm_pred = (y_pred+baseline) * \
                K.maximum(param_dict['maxabs'], eps)
            denorm_true = (y_true+baseline) * \
                K.maximum(param_dict['maxabs'], eps)
            return loss(denorm_true, denorm_pred)
    elif method == 'RobustScaler':
        def denorm_loss(y_true, y_pred):
            denorm_pred = (y_pred+baseline) * \
                K.maximum(param_dict['iqr'], eps) + param_dict['median']
            denorm_true = (y_true+baseline) * \
                K.maximum(param_dict['iqr'], eps) + param_dict['median']
            return loss(denorm_true, denorm_pred)
    elif method == 'PowerTransform':
        lmbda = param_dict['lambda']
        if lmbda.size > 1:
            raise NotImplementedError(
                'still working power transform thats not uniform over a profile')
        elif np.abs(lmbda) < eps:
            def denorm_loss(y_true, y_pred):
                denorm_true = (y_true+baseline) * \
                    K.maximum(param_dict['std'], eps) + param_dict['mean']
                denorm_pred = (y_pred+baseline) * \
                    K.maximum(param_dict['std'], eps) + param_dict['mean']
                true_posmask = K.cast(denorm_true >= 0, K.floatx())
                true_negmask = K.cast(denorm_true < 0, K.floatx())
                pred_posmask = K.cast(denorm_pred >= 0, K.floatx())
                pred_negmask = K.cast(denorm_pred < 0, K.floatx())
                denorm_true_pos = K.exp(denorm_true)-1
                denorm_true_neg = 1 - K.pow(K.abs(-(2 - lmbda) *
                                                  denorm_true + 1), 1 / (2 - lmbda))
                denorm_pred_pos = K.exp(denorm_pred)-1
                denorm_pred_neg = 1 - K.pow(L.abs(-(2 - lmbda) *
                                                  denorm_pred + 1), 1 / (2 - lmbda))
                denorm_true = true_posmask*denorm_true_pos + true_negmask*denorm_true_neg
                denorm_pred = pred_posmask*denorm_pred_pos + pred_negmask*denorm_pred_neg
                return loss(denorm_true, denorm_pred)
        elif np.abs(lmbda-2) < eps:
            def denorm_loss(y_true, y_pred):
                denorm_true = (y_true+baseline) * \
                    K.maximum(param_dict['std'], eps) + param_dict['mean']
                denorm_pred = (y_pred+baseline) * \
                    K.maximum(param_dict['std'], eps) + param_dict['mean']
                true_posmask = K.cast(denorm_true >= 0, K.floatx())
                true_negmask = K.cast(denorm_true < 0, K.floatx())
                pred_posmask = K.cast(denorm_pred >= 0, K.floatx())
                pred_negmask = K.cast(denorm_pred < 0, K.floatx())
                denorm_true_pos = K.pow(
                    K.abs(denorm_true * lmbda + 1), 1 / lmbda) - 1
                denorm_true_neg = 1 - K.exp(-denorm_true)
                denorm_pred_pos = K.pow(
                    K.abs(denorm_pred * lmbda + 1), 1 / lmbda) - 1
                denorm_pred_neg = 1 - K.exp(-denorm_pred)
                denorm_true = true_posmask*denorm_true_pos + true_negmask*denorm_true_neg
                denorm_pred = pred_posmask*denorm_pred_pos + pred_negmask*denorm_pred_neg
                return loss(denorm_true, denorm_pred)
        else:
            def denorm_loss(y_true, y_pred):
                denorm_true = (y_true+baseline) * \
                    K.maximum(param_dict['std'], eps) + param_dict['mean']
                denorm_pred = (y_pred+baseline) * \
                    K.maximum(param_dict['std'], eps) + param_dict['mean']
                true_posmask = K.cast(denorm_true >= 0, K.floatx())
                true_negmask = K.cast(denorm_true < 0, K.floatx())
                pred_posmask = K.cast(denorm_pred >= 0, K.floatx())
                pred_negmask = K.cast(denorm_pred < 0, K.floatx())
                denorm_true_pos = K.pow(
                    K.abs(denorm_true * lmbda + 1), 1 / lmbda) - 1
                denorm_true_neg = 1 - K.pow(K.abs(-(2 - lmbda) *
                                                  denorm_true + 1), 1 / (2 - lmbda))
                denorm_pred_pos = K.pow(
                    K.abs(denorm_pred * lmbda + 1), 1 / lmbda) - 1
                denorm_pred_neg = 1 - K.pow(K.abs(-(2 - lmbda) *
                                                  denorm_pred + 1), 1 / (2 - lmbda))
                denorm_true = true_posmask*denorm_true_pos + true_negmask*denorm_true_neg
                denorm_pred = pred_posmask*denorm_pred_pos + pred_negmask*denorm_pred_neg
                return loss(denorm_true, denorm_pred)
    elif method is None or method == 'None':
        def denorm_loss(y_true, y_pred):
            return loss(y_true+baseline, y_pred+baseline)
    else:
        raise ValueError("Unknown normalization method")
    if loss.__name__ == 'mean_absolute_error':
        denorm_loss.__name__ = 'denorm_MAE'
    elif loss.__name__ == 'mean_squared_error':
        denorm_loss.__name__ = 'denorm_MSE'
    else:
        denorm_loss.__name__ = 'denorm_' + loss.__name__
    return denorm_loss


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
        hinge_mse: A loss function that takes in y_true and y_pred and returns 
            a scalar loss value
    """
    mse_weight_vector = K.constant(mse_weight_vector, dtype=K.floatx())
    hinge_weight = K.constant(hinge_weight, dtype=K.floatx())
    # need baseline for hinge loss, want to make sure prediction and
    # true are on the same side of baseline
    # baseline = predict current value
    if predict_deltas:
        # if predicting deltas, baseline is zero
        baseline = K.cast_to_floatx(0)
    else:
        # get the current input to the model for baseline comparison
        baseline = model.get_layer('input_' + sig).input[:, -1]

    def hinge_mse(y_true, y_pred):
        delta_true = y_true-baseline
        delta_pred = y_pred-baseline
        mse_loss = K.mean(K.square(y_pred-y_true)*mse_weight_vector, axis=-1)
        hinge_loss = K.mean(K.maximum(-(delta_true * delta_pred), 0.), axis=-1)
        return mse_loss + hinge_weight*hinge_loss
    return hinge_mse
