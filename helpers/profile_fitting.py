import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# from what i can see in the PCS, the constraints they use are a > 0.1, b > 0.001, alpha > 0.0001, .85 < xsym < 1.15, and hwid > 0.01
# /* This is the model equation
# * y = a*MTANH(alpha, z) + b
# *
# * z = (xsym-x)/hwid
# *
# * MTANH = ((1 + alpha*z)*EXP(z) - EXP(-z))
# * / (EXP(z) + EXP(-z))
# *
# * the parameters are
# * p[0] = a
# * p[1] = b
# * p[2] = alpha
# * p[3] = xsym
# * p[4] = hwid
# with initial guess:
# // The guess for fit parameters
# double p[5] = {1.0, 3.0, 0.01, 1.0, 0.01}


def mtanh(x, p):
    a = p[0]
    b = p[1]
    alpha = p[2]
    xsym = p[3]
    hwid = p[4]

    z = (xsym - x) / hwid

    y = a * ((1 + alpha * z) * np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) + b
    return y


def mtanh_res(params, x_data, y_data, y_err):
    y_model = mtanh(x_data, params)
    res = (y_data - y_model) / y_err
    return res


def fit_mtanh(x_data, y_data, y_err=None):

    if y_err is None:
        y_err = np.ones(y_data.shape)
        y_err[np.abs(y_data) < 1e-10] = 1e30

    p0 = np.array([1.0, 3.0, 0.01, 1.0, 0.01])
    pmin = np.array([0.1, 0.001, 0.0001, 0.85, 0.01])
    pmax = np.array([np.inf, np.inf, np.inf, 1.15, np.inf])

    kwargs = {"x_data": x_data, "y_data": y_data, "y_err": y_err}
    fit_out = least_squares(
        mtanh_res,
        p0,
        jac="2-point",
        bounds=(pmin, pmax),
        method="trf",
        ftol=1e-08,
        xtol=1e-08,
        gtol=1e-08,
        x_scale=1.0,
        loss="linear",
        f_scale=1.0,
        diff_step=None,
        tr_solver=None,
        tr_options={},
        jac_sparsity=None,
        max_nfev=200,
        verbose=0,
        args=(),
        kwargs=kwargs,
    )
    return fit_out.x


def fit_profile(x_data, y_data, y_err=None):
    if y_err is None:
        y_err = np.ones(y_data.shape)
        y_err[np.abs(y_data) < 1e-10] = 1e30

    p = fit_mtanh(x_data, y_data, y_err)
    x121 = np.linspace(0, 1.2, 121)
    y121 = mtanh(x121, p)
    x65 = np.linspace(0, 1, 65)
    y65 = interp1d(x121, y121, "linear", fill_value="extrapolate", assume_sorted=True)(
        x65
    )
    return y65
