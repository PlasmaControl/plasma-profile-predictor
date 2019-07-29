import numpy as np
from scipy import optimize
from tqdm import tqdm
import keras.backend as K


def normalize_arr(data, method, uniform_over_profile=True):
    """Normalizes data before training

    Args:
        data: Numpy array. Array.shape[0] = samples
        method (str): One of `StandardScaler`, `MinMax`, `MaxAbs`,
            `RobustScaler`, `PowerTransform`.
        uniform_over_profile (bool): 'True' uses the same normalization
            parameters over a whole profile, 'False' normalizes each spatial
            point separately.

    Returns:
        data: Numpy array of normalized data.
        param_dict (dict): Dictionary of parameters used during normalization,
            to be used for denormalizing later. Eg, mean, stddev, method, etc.
    """
    param_dict = {}
    # first replace all infs and nans with mean value
    data[np.isinf(data)] = np.nan
    nanmean = np.nanmean(data, axis=(0, 1))
    param_dict['nanmean'] = nanmean
    if data.ndim > 2:
        for i in range(data.shape[2]):
            data[np.isnan(data[:, :, i]), i] = nanmean[i]
    else:
        data[np.isnan(data)] = nanmean
    # then normalize
    if method == 'StandardScaler':
        if uniform_over_profile or data.ndim < 3:
            mean = np.mean(data)
            std = np.std(data)
        else:
            mean = np.mean(data, axis=(0, 1), keepdims=True)
            std = np.std(data, axis=(0, 1), keepdims=True)
        param_dict.update({'method': method,
                           'mean': mean,
                           'std': std})
        return (data-mean)/np.maximum(std, np.finfo(np.float32).eps), param_dict

    elif method == 'MinMax':
        if uniform_over_profile or data.ndim < 3:
            armin = np.amin(data)
            armax = np.amax(data)
        else:
            armin = np.amin(data, axis=(0, 1), keepdims=True)
            armax = np.amax(data, axis=(0, 1), keepdims=True)
        param_dict.update({'method': method,
                           'armin': armin,
                           'armax': armax})
        return (data-armin)/np.maximum((armax-armin), np.finfo(np.float32).eps), param_dict

    elif method == 'MaxAbs':
        if uniform_over_profile or data.ndim < 3:
            maxabs = np.amax(np.abs(data))
        else:
            maxabs = np.amax(np.abs(data), axis=(0, 1), keepdims=True)
        param_dict.update({'method': method,
                           'maxabs': maxabs})
        return data/np.maximum(maxabs, np.finfo(np.float32).eps), param_dict

    elif method == 'RobustScaler':
        if uniform_over_profile or data.ndim < 3:
            median = np.median(data)
            iqr = np.subtract(*np.percentile(data, [75, 25]))
        else:
            median = np.median(data, axis=0)
            iqr = np.subtract(*np.percentile(data, [75, 25], axis=(0, 1)))
        param_dict.update({'method': method,
                           'median': median,
                           'iqr': iqr})
        return (data-median)/np.maximum(iqr, np.finfo(np.float32).eps), param_dict

    elif method == 'PowerTransform':
        def yeo_johnson_transform(x, lmbda):
            """Return transformed input x following Yeo-Johnson transform with
            parameter lambda.
            """
            out = np.zeros_like(x)
            pos = x >= 0  # binary mask
            # when x >= 0
            if abs(lmbda) < np.finfo(np.float32).eps:
                out[pos] = np.log1p(x[pos])
            else:  # lmbda != 0
                out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda
            # when x < 0
            if abs(lmbda - 2) > np.finfo(np.float32).eps:
                out[~pos] = - \
                    (np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
            else:  # lmbda == 2
                out[~pos] = -np.log1p(-x[~pos])
            return out

        def yeo_johnson_optimize(x):
            """Find and return optimal lambda parameter of the Yeo-Johnson
            transform by MLE, for observed data x.
            Like for Box-Cox, MLE is done via the brent optimizer. From Scipy
            """
            def _neg_log_likelihood(lmbda):
                """Return the negative log likelihood of the observed data x as a
                function of lambda. From Scipy"""
                x_trans = yeo_johnson_transform(x, lmbda)
                n_samples = x.shape[0]
                loglike = -n_samples / 2 * np.log(x_trans.var())
                loglike += (lmbda - 1) * (np.sign(x) *
                                          np.log1p(np.abs(x))).sum()
                return -loglike
            # choosing bracket -2, 2 like for boxcox
            return optimize.brent(_neg_log_likelihood, brack=(-2, 2))
        if uniform_over_profile or data.ndim < 3:
            lmbda = yeo_johnson_optimize(data.flatten())
            y = yeo_johnson_transform(
                data.flatten(), lmbda).reshape(data.shape)
            mean = np.mean(y)
            std = np.std(y)
        else:
            y = np.zeros_like(data)
            lmbda = np.array([yeo_johnson_optimize(data[:, :, i])
                              for i in range(data.shape[2])])
            for i, l in enumerate(lmbda):
                y[:, :, i] = yeo_johnson_transform(data[:, :, i], l)
            mean = np.mean(y, axis=(0, 1))
            std = np.std(y, axis=(0, 1))
        param_dict.update({'method': method,
                           'lambda': lmbda,
                           'mean': mean,
                           'std': std})
        return (y-mean)/np.maximum(std, np.finfo(np.float32).eps), param_dict
    elif method is None or method == 'None':
        param_dict.update({'method': method})
        return data, param_dict
    else:
        raise ValueError("Unknown normalization method")


def normalize(data, method, uniform_over_profile=True, verbose=1):
    """Normalizes data before training

    Args:
        data: Numpy array or dictionary of numpy arrays. If a dictionary, all
            arrays are normalized using the same method, but each array with
            respect to itself. Array.shape[0] = batches
        method (str): One of `StandardScaler`, `MinMax`, `MaxAbs`,
            `RobustScaler`, `PowerTransform`.
        uniform_over_profile (bool): 'True' uses the same normalization
            parameters over a whole profile, 'False' normalizes each spatial
            point separately.
        verbose (int): verbosity level. 0 is no CL output, 1 shows progress.

    Returns:
        data: Numpy array or dictionary of numpy arrays. Normalized data.
        param_dict (dict): Dictionary of parameters used during normalization,
            to be used for denormalizing later. Eg, mean, stddev, method, etc.
    """
    verbose = bool(verbose)
    if type(data) is dict:
        param_dict = {}
        for key in tqdm(data.keys(), desc='Normalizing', ascii=True, dynamic_ncols=True,
                        disable=not verbose):
            if key not in ['time', 'shotnum']:
                data[key], p = normalize_arr(
                    data[key], method, uniform_over_profile)
                param_dict[key] = p
        return data, param_dict
    else:
        return normalize_arr(data, method, uniform_over_profile)


def denormalize_arr(data, param_dict):
    """Denormalizes data after training

    Args:
        data: Numpy array of data to denorm.
        param_dict (dict): Dictionary of parameters used during normalization,
            to be used for denormalizing. Eg, mean, stddev, method, etc.

    Returns:
        data: Numpy array of denormalized data.
    """
    eps = np.finfo('float32').eps
    for key, val in param_dict.items():
        if K.is_tensor(val):
            val = np.array(K.eval(val))
    if param_dict['method'] == 'StandardScaler':
        return data*np.maximum(param_dict['std'], eps) + param_dict['mean']
    elif param_dict['method'] == 'MinMax':
        return data*np.maximum((param_dict['armax']-param_dict['armin']), eps)
        + param_dict['armin']
    elif param_dict['method'] == 'MaxAbs':
        return data*np.maximum(param_dict['maxabs'], eps)
    elif param_dict['method'] == 'RobustScaler':
        return data*np.maximum(param_dict['iqr'], eps) + param_dict['median']
    elif param_dict['method'] == 'PowerTransform':
        y = data*np.maximum(param_dict['std'], eps) + param_dict['mean']

        def np_yeo_johnson_inverse_transform(x, lmbda):
            """Return inverse-transformed input x following Yeo-Johnson inverse
            transform with parameter lambda. From Scipy
            """
            x_inv = np.zeros_like(x)
            pos = x >= 0
            # when x >= 0
            if np.abs(lmbda) < np.finfo(np.float32).eps:
                x_inv[pos] = np.exp(x[pos]) - 1
            else:  # lmbda != 0
                x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1
            # when x < 0
            if np.abs(lmbda - 2) > np.finfo(np.float32).eps:
                x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1,
                                           1 / (2 - lmbda))
            else:  # lmbda == 2
                x_inv[~pos] = 1 - np.exp(-x[~pos])
            return x_inv
        if param_dict['lambda'].size > 1:
            for i, l in enumerate(param_dict['lambda']):
                y[:, i] = np_yeo_johnson_inverse_transform(y[:, i], l)
        else:
            y = np_yeo_johnson_inverse_transform(
                y.flatten(), param_dict['lambda']).reshape(y.shape)
        return y
    elif param_dict['method'] is None or param_dict['method'] == 'None':
        return data
    else:
        raise ValueError("Unknown normalization method")


def denormalize(data, param_dict, verbose=1):
    """Denormalizes data after training

    Args:
        data: Numpy array or dictionary of numpy arrays.
        param_dict (dict): Dictionary of parameters used during normalization,
            to be used for denormalizing. Eg, mean, stddev, method, etc.
        verbose (int): verbosity level. 0 is no CL output, 1 shows progress.

    Returns:
        data: Numpy array or dictionary of numpy arrays. Denormalized data.
    """
    verbose = bool(verbose)
    if type(data) is dict:
        for key in tqdm(data.keys(), desc='Denormalizing', ascii=True, dynamic_ncols=True,
                        disable=not verbose):
            if key not in ['time', 'shotnum']:
                data[key] = denormalize_arr(data[key], param_dict[key])
        return data
    else:
        return denormalize_arr(data, param_dict)


def renormalize(data, param_dict, verbose=1):
    """Normalizes data using already determined parameters

    Args:
        data: Numpy array or dictionary of numpy arrays of raw data.
        param_dict (dict): Dictionary of parameters used during normalization,
            Eg, mean, stddev, method, etc.
        verbose (int): verbosity level. 0 is no CL output, 1 shows progress.

    Returns:
        data: Numpy array or dictionary of numpy arrays. Normalized data.
    """
    verbose = bool(verbose)
    if type(data) is dict:
        for key in tqdm(data.keys(), desc='Normalizing', ascii=True, dynamic_ncols=True,
                        disable=not verbose):
            if key not in ['time', 'shotnum']:
                data[key] = renormalize(data[key], param_dict[key])
        return data
    else:
        # first remove all inf/nan
        data[np.isinf(data)] = np.nan
        if data.ndim > 2:
            for i in range(data.shape[2]):
                data[np.isnan(data[:, :, i]), i] = param_dict['nanmean'][i]
        else:
            data[np.isnan(data)] = param_dict['nanmean']
        # then normalize
        if param_dict['method'] == 'StandardScaler':
            return (data - param_dict['mean'])/np.maximum(
                param_dict['std'], np.finfo(np.float32).eps)
        elif param_dict['method'] == 'MinMax':
            return (data - param_dict['armin'])/np.maximum(
                (param_dict['armax']-param_dict['armin']), np.finfo(np.float32).eps)
        elif param_dict['method'] == 'MaxAbs':
            return data/np.maximum(param_dict['maxabs'], np.finfo(np.float32).eps)
        elif param_dict['method'] == 'RobustScaler':
            return (data - param_dict['median'])/np.maximum(
                param_dict['iqr'], np.finfo(np.float32).eps)
        elif param_dict['method'] == 'PowerTransform':
            def yeo_johnson_transform(x, lmbda):
                """Return transformed input x following Yeo-Johnson transform with
                parameter lambda.
                """
                out = np.zeros_like(x)
                pos = x >= 0  # binary mask
                # when x >= 0
                if abs(lmbda) < np.finfo(np.float32).eps:
                    out[pos] = np.log1p(x[pos])
                else:  # lmbda != 0
                    out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda
                # when x < 0
                if abs(lmbda - 2) > np.finfo(np.float32).eps:
                    out[~pos] = - \
                        (np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
                else:  # lmbda == 2
                    out[~pos] = -np.log1p(-x[~pos])
                return out
            y = data
            if param_dict['lambda'].size > 1:
                for i, l in enumerate(param_dict['lambda']):
                    y[:, i] = yeo_johnson_transform(y[:, i], l)
            else:
                y = yeo_johnson_transform(
                    y.flatten(), param_dict['lambda']).reshape(y.shape)
            y = (y - param_dict['mean'])/np.maximum(
                param_dict['std'], np.finfo(np.float32).eps)
            return y
        elif param_dict['method'] is None or param_dict['method'] == 'None':
            return data
        else:
            raise ValueError("Unknown normalization method")
