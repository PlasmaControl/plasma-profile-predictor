import numpy as np
import scipy
import copy
from helpers.normalization import normalize, denormalize, renormalize
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import helpers.plot_settings
import multiprocessing
from helpers.data_generator import process_data
from helpers.qpmpc import mpc_setup, mpc_action


class LRANMPC:
    """Helper class for doing MPC with LRAN model

    Parameters
    ----------
    model : keras model
        should be LRAN type with encoders/decoders etc
    params : dict
        parameters used for training, ie "scenario"
    Q, R : ndarray
        weight matrices
    Nlook : int
        mpc lookahead
    """

    def __init__(
        self,
        model,
        params,
        xtarget=None,
        Q=None,
        R=None,
        Nlook=None,
        use_osqp=False,
    ):

        self._setup = False
        self._model = model
        self._params = params
        self._parse_model()
        self._nu = self.B.shape[1]
        self._nz = self.A.shape[0]
        self._use_osqp = use_osqp

        self.state_names = self._params["profile_names"] + self._params["scalar_names"]
        self.actuator_names = self._params["actuator_names"]
        self.R = R
        self.Q = Q
        if Nlook is None:
            Nlook = 10
        self._Nlook = Nlook
        self.mpc_setup()

        # set default bounds/targets
        self._umin_default = {name: -np.inf for name in self.actuator_names}
        self._umax_default = {name: np.inf for name in self.actuator_names}
        self._dumin_default = {name: -np.inf for name in self.actuator_names}
        self._dumax_default = {name: np.inf for name in self.actuator_names}
        self._uref_default = {name: 0 for name in self.actuator_names}
        self._zmin_default = -np.inf * np.ones(self._nz)
        self._zmax_default = np.inf * np.ones(self._nz)
        self._xtarget_default = self.denormalize(self.decode(np.zeros(self._nz)))

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, newQ):
        if newQ is None:
            newQ = np.eye(self._nz)
        elif np.isscalar(newQ):
            newQ = newQ * np.eye(self._nz)
        elif np.atleast_1d(newQ).ndim == 1:
            newQ = np.diag(newQ)
        self._Q = newQ
        if self._setup:
            self.mpc_setup()

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, newR):
        if newR is None:
            newR = np.eye(self._nu)
        elif np.isscalar(newR):
            newR = newR * np.eye(self._nu)
        elif np.atleast_1d(newR).ndim == 1:
            newR = np.diag(newR)
        self._R = newR
        if self._setup:
            self.mpc_setup()

    def _parse_model(self):
        self._A, self._B = get_AB(self._model)

        (
            self._state_input,
            self._control_input,
            self._state_encoder,
            self._state_decoder,
            self._control_encoder,
            self._control_decoder,
        ) = get_submodels(self._model)
        norm_layers = [
            layer
            for layer in self._state_input.layers + self._control_input.layers
            if "norm" in layer.name
        ]
        self._norm_layers = {layer.name[5:]: layer for layer in norm_layers}

    def predict_latent(self, z0, u):
        """Predict future values of latent state

        Parameters
        ----------
        z0 : ndarray, shape([sample], nz,)
            initial value for latent state
        u : ndarray, shape(time, [sample], nu)
            actuator values vs time, in normalized units

        Returns
        -------
        zt : ndarray, shape(time+1, [sample], nz)
            predicted future values of latent state,
            starting at z0
        """
        zt = [z0]
        for i, ut in enumerate(u):
            zt.append(zt[i] @ self.A.T + ut @ self.B.T)
        return np.array(zt)

    def predict(self, x0, u):
        """Predict future values of physical state

        Parameters
        ----------
        x0 : dict of ndarray, shape([sample], feature)
            initial value for state in physical units
            keys should be strings from lran.state_names
        u : dict of ndarray, shape([sample], time,)
            actuator values vs time, in physical units
            keys should be strings from lran.actuator_names
            values should be 2d arrays with time as the first axis
            or 3d arrays with sample as the first axis, time as the 2nd

        Returns
        -------
        xt : dict of ndarray
            predicted future values of physical state
        """
        batch = False
        x0 = self.normalize(x0)
        u = self.normalize(u)
        z0 = self.encode(x0).squeeze()
        v = np.stack([u[key] for key in self.actuator_names], axis=-1)
        if v.ndim > 2:
            batch = True
            v = np.moveaxis(v, 0, 1)
        zt = self.predict_latent(z0, v)
        if batch:
            zt = np.moveaxis(zt, 0, 1)
        xt = self.decode(zt)
        xt = self.denormalize(xt)
        return xt

    def normalize(self, data):
        # TODO: flag to ignore infs
        out = renormalize(
            copy.copy(data), self._params["normalization_dict"], verbose=0
        )
        for key in out:
            if key in self._norm_layers:
                norm = self._norm_layers[key]
                out[key] = (out[key] - norm.moving_mean.numpy()) / np.sqrt(
                    norm.moving_variance.numpy() + norm.epsilon
                )
        return out

    def denormalize(self, data):
        out = copy.copy(data)
        for key in out:
            if key in self._norm_layers:
                norm = self._norm_layers[key]
                out[key] = (
                    out[key] * np.sqrt(norm.moving_variance.numpy() + norm.epsilon)
                    + norm.moving_mean.numpy()
                )
        out = denormalize(out, self._params["normalization_dict"], verbose=0)

        return out

    def encode(self, state):
        """Encodes physical state to latent linear state

        Parameters
        ----------
        state: dict of ndarray
            dictionary of arrays, keys should be names of states

        Returns
        -------
        latent_state : ndarray
            latent linear state

        """
        inp = {"input_" + key: np.atleast_2d(val) for key, val in state.items()}
        if inp["input_" + self._params["profile_names"][0]].ndim > 2:
            batch = True
            batch_size = inp["input_" + self._params["profile_names"][0]].shape[0]
            timesteps = inp["input_" + self._params["profile_names"][0]].shape[1]
            inp = {
                key: val.reshape((batch_size * timesteps, -1))
                for key, val in inp.items()
            }
        else:
            batch = False
        outp = self._state_encoder.predict(inp)
        if batch:
            outp = outp.reshape((batch_size, timesteps, -1))
        return outp

    def decode(self, latent_state):
        """Decodes physical state to latent linear state

        Parameters
        ----------
        latent_state : ndarray
            latent linear state

        Returns
        -------
        state: dict of ndarray
            dictionary of arrays, keys are names of states

        """
        inp = np.atleast_2d(latent_state)
        if inp.ndim > 2:
            batch = True
            batch_size = inp.shape[0]
            timesteps = inp.shape[1]
            inp = inp.reshape((batch_size * timesteps, -1))
        else:
            batch = False
        out = self._state_decoder.predict(inp)
        if batch:
            out = {
                key: val.reshape((batch_size, timesteps, -1)).squeeze()
                for key, val in zip(self.state_names, out)
            }
        else:
            out = {key: val for key, val in zip(self.state_names, out)}
        return out

    def mpc_setup(
        self, Q=None, R=None, Nlook=None, rho=0.1, sigma=1e-4, dt=None, use_osqp=None
    ):
        """setup mpc problem

        Parameters
        ----------
        Q, R: ndarray
            state and control weight matrices
        Nlook : int
            number of steps in the mpc lookahead

        """

        A = self.A
        B = self.B
        if Q is None:
            Q = self.Q
        else:
            self.Q = Q
        if R is None:
            R = self.R
        else:
            self.R = R
        if Nlook is None:
            Nlook = self._Nlook
        else:
            self._Nlook = Nlook
        if use_osqp is None:
            use_osqp = self._use_osqp
        if dt is None:
            dt = self._params["dt"]

        out = mpc_setup(A, B, Q, R, Nlook, rho, sigma, dt, use_osqp)

        if use_osqp:
            self._qp = out[0]
            self._E = out[1]
            self._F = out[2]
            self._P = None
            self._G = None
            self._Ac = None
            self._Qhat = out[3]
            self._Rhat = out[4]

        else:
            self._qp = None
            self._E = out[0]
            self._F = out[1]
            self._P = out[2]
            self._G = out[3]
            self._Ac = out[4]
            self._Qhat = out[5]
            self._Rhat = out[6]
        self._setup = True

    def mpc_action(
        self,
        xk,
        xtarget=None,
        umin={},
        dumin={},
        uref={},
        dumax={},
        umax={},
        return_all=False,
        maxiter=100,
    ):
        """find control action via MPC

        Parameters
        ----------
        xk : dictionary of ndarray
            current value for the physical state
        xtarget : dictionary of ndarray
            target value for physical state
        umin, umax : dict of float
            upper and lower bounds on control input
            defaults to +/- inf
        dumin, dumax : dict of float
            upper/lower bounds for actuator derivatives
        uref : dict of float
            nominal feedforward values for control
            defaults to 0
        return_all : bool
            whether to also return optimal control sequence for entire
            prediction horizon

        Returns
        -------
        u : dict of float
            action to take at current timestep
        uhat : dict of ndarray
            actions to take over horizon, only returned if return_all is True
        """
        # set values
        if xtarget is None:
            xtarget = self._xtarget_default
        umind = self._umin_default.copy()
        umaxd = self._umax_default.copy()
        urefd = self._uref_default.copy()
        dumind = self._dumin_default.copy()
        dumaxd = self._dumax_default.copy()
        # TODO: target/bounds in future
        umind.update(umin)
        umaxd.update(umax)
        urefd.update(uref)
        dumind.update(dumin)
        dumaxd.update(dumax)
        # normalize actuator bounds, keeping infs
        umina = []
        umaxa = []
        urefa = []
        dumina = []
        dumaxa = []
        for name in self.actuator_names:
            if np.isinf(umind[name]):
                umina.append(umind[name])
            else:
                umina.append(
                    self.normalize({name: np.atleast_1d(umind[name])})[name].squeeze()
                )
            if np.isinf(dumind[name]):
                dumina.append(dumind[name])
            else:
                dumina.append(
                    self.normalize({name: np.atleast_1d(dumind[name])})[name].squeeze()
                )
            if np.isinf(dumaxd[name]):
                dumaxa.append(dumaxd[name])
            else:
                dumaxa.append(
                    self.normalize({name: np.atleast_1d(dumaxd[name])})[name].squeeze()
                )
            if np.isinf(umaxd[name]):
                umaxa.append(umaxd[name])
            else:
                umaxa.append(
                    self.normalize({name: np.atleast_1d(umaxd[name])})[name].squeeze()
                )
            urefa.append(
                self.normalize({name: np.atleast_1d(urefd[name])})[name].squeeze()
            )
        umin = np.asarray(umina)
        umax = np.asarray(umaxa)
        uref = np.asarray(urefa)
        dumin = np.asarray(dumina)
        dumax = np.asarray(dumaxa)

        # encode state and target
        xk = self.normalize(xk)
        zk = self.encode(xk).squeeze()
        xtarget = self.normalize(xtarget)
        ztarget = self.encode(xtarget).squeeze()

        # set up inequality constraints
        uminhat = np.tile(umin, (self._Nlook,))
        umaxhat = np.tile(umax, (self._Nlook,))
        duminhat = np.tile(dumin, (self._Nlook - 1,))
        dumaxhat = np.tile(dumax, (self._Nlook - 1,))
        urefhat = np.tile(uref, (self._Nlook,))
        ztarget = np.tile(ztarget, (self._Nlook,))

        # TODO: warm starting based on prev soln?
        uhat = np.zeros(self._nu * self._Nlook)
        lagrange = np.zeros(self._nu * (2 * self._Nlook - 1))

        uhat, lagrange = mpc_action(
            zk,
            ztarget,
            uhat,
            lagrange,
            uminhat,
            duminhat,
            urefhat,
            dumaxhat,
            umaxhat,
            self._E,
            self._F,
            self._P,
            self._G,
            self._Ac,
            self._Qhat,
            self._Rhat,
            rho=0.1,
            sigma=1e-4,
            alpha=1.6,
            maxiter=maxiter,
            qp=self._qp,
        )
        uhat = uhat.reshape((self._Nlook, -1))

        # parse into output dict in physical units
        control = {sig: uhat[:, i] for i, sig in enumerate(self.actuator_names)}
        uhat = self.denormalize(control)
        u = {key: val[0] for key, val in uhat.items()}
        if return_all:
            return u, uhat
        return u


def get_AB(model):
    """Get linear system A, B matrices from trained LRAN

    Parameters
    ----------
    model : keras model
        trained LRAN model

    Returns
    -------
    A : ndarray, shape(N,N)
        system dynamics matrix
    B : ndarray, shape(N,M)
        system control matrix
    """
    A = model.get_layer("AB_matrices").get_weights()[1].T
    B = model.get_layer("AB_matrices").get_weights()[0].T
    return A, B


def get_submodels(model):
    """Get encoder/decoder submodels from trained LRAN

    Parameters
    ----------
    model : keras model
        trained LRAN model

    Returns
    -------
    state_encoder, state_decoder, control_encoder, control_decoder : keras models
        encoder/decoder submodels for state and control
    """
    state_input = model.get_layer("state_input")
    control_input = model.get_layer("control_input")
    state_encoder = model.get_layer("state_encoder_time_dist").layer
    control_encoder = model.get_layer("ctrl_encoder_time_dist").layer
    state_decoder = model.get_layer("state_decoder_time_dist").layer
    control_decoder = model.get_layer("ctrl_decoder_time_dist").layer

    return (
        state_input,
        control_input,
        state_encoder,
        state_decoder,
        control_encoder,
        control_decoder,
    )


def plot_autoencoder_AB(A, B, filename=None, **kwargs):
    """Plot heatmap of A, B matrices

    Parameters
    ----------
    A : ndarray, shape(N,N)
        system dynamics matrix
    B : ndarray, shape(N,M)
        system control matrix
    filename : str, optional
        filename to save figure

    Returns
    -------
    fig, ax : matplotlib figure / axes

    """

    f, axes = plt.subplots(
        1,
        2,
        figsize=kwargs.get("figsize", (28, 14)),
        gridspec_kw={"width_ratios": [A.shape[0], B.shape[1]]},
    )
    sns.heatmap(
        A,
        cmap=kwargs.get("cmap", "Spectral"),
        annot=kwargs.get("annot", False),
        square=kwargs.get("square", True),
        robust=kwargs.get("robust", False),
        ax=axes[0],
    ).set_title("A")
    sns.heatmap(
        B,
        cmap=kwargs.get("cmap", "Spectral"),
        annot=kwargs.get("annot", False),
        square=kwargs.get("square", True),
        robust=kwargs.get("robust", False),
        ax=axes[1],
    ).set_title("B")

    if filename:
        f.savefig(filename, bbox_inches="tight")
    return f, axes


def plot_autoencoder_spectrum(A, dt=0.05, filename=None, **kwargs):
    """Plot eigenvalue spectrum of discrete and continuous time system

    Parameters
    ----------
    A : ndarray, shape(N,N)
        system dynamics matrix
    dt : float, optional
        timestep
    filename : str, optional
        filename to save figure

    Returns
    -------
    fig, ax : matplotlib figure / axes

    """
    eigvals, eigvecs = np.linalg.eig(A)
    logeigvals = np.log(eigvals)
    for i, elem in enumerate(logeigvals):
        if abs(np.imag(elem) - np.pi) < np.finfo(np.float32).resolution:
            logeigvals[i] = np.real(elem) + 0j
    logeigvals = logeigvals / dt

    f, axes = plt.subplots(1, 2, figsize=kwargs.get("figsize", (28, 14)))
    axes[0].scatter(np.real(eigvals), np.imag(eigvals))
    t = np.linspace(0, 2 * np.pi, 1000)
    axes[0].plot(np.cos(t), np.sin(t))

    axes[0].set_title("Eigenvalues of A")
    axes[0].grid(color="gray")
    axes[0].set_xlabel("Re($\lambda$)")
    axes[0].set_ylabel("Im($\lambda$)")

    axes[1].scatter(np.real(logeigvals), np.imag(logeigvals))
    axes[1].set_title("Eigenvalues of A")
    axes[1].grid(color="gray")
    axes[1].set_xlabel("Growth Rate (1/s)")
    axes[1].set_ylabel("$\omega$ (rad/s)")
    axes[1].set_xlim(
        (
            1.1 * np.min(np.real(logeigvals)),
            np.maximum(1.1 * np.max(np.real(logeigvals)), 0),
        )
    )

    if filename:
        f.savefig(filename, bbox_inches="tight")

    return f, axes


def _calc_delta_norm(i, data):
    return np.linalg.norm(data - data[i], axis=1)


def compute_lipschitz_constant(x, z, workers=1, verbose=True):
    """Compute norms of differences between states and induced Lipschitz constant


    Parameters
    ----------
    x : ndarray, shape(nsamples, nstates)
        physical state
    z : ndarray, shape(nsamples, nlatentstates)
        latent state
    workers : int, optional
        how many parallel processes to work
    verbose : bool
        whether to display progress bar

    Returns
    -------
    lipschitz_constant : ndarray
        ratio of norm of differences ||z - z'||/||x - x'||
    delta_x_norms : ndarray
        norm of differences in physical state
    delta_z_norms : ndarray
        norm of differences in latent state
    """
    assert x.shape[0] == z.shape[0]

    with multiprocessing.Pool(workers) as pool:
        delta_x_norms = pool.starmap(
            _calc_delta_norm,
            tqdm(
                [(i, x) for i in range(x.shape[0])],
                desc="computing delta x norms",
                ascii=True,
                dynamic_ncols=True,
                disable=not verbose,
            ),
        )
        delta_z_norms = pool.starmap(
            _calc_delta_norm,
            tqdm(
                [(i, z) for i in range(z.shape[0])],
                desc="computing delta z norms",
                ascii=True,
                dynamic_ncols=True,
                disable=not verbose,
            ),
        )

    delta_x_norms = np.concatenate(delta_x_norms).flatten()
    delta_z_norms = np.concatenate(delta_z_norms).flatten()
    mask = delta_x_norms != 0
    delta_x_norms = delta_x_norms[mask]
    delta_z_norms = delta_z_norms[mask]
    lipschitz_constant = delta_z_norms / delta_x_norms
    return lipschitz_constant, delta_x_norms, delta_z_norms


def compute_operator_norm(x, z):
    """Compute norms of inputs, outputs, and induced operator

    Parameters
    ----------
    x : ndarray, shape(nsamples, nstates)
        physical state
    z : ndarray, shape(nsamples, nlatentstates)
        latent state

    Returns
    -------
    operator_norm : ndarray
        induced norm of the encoding operator
    x_norm : ndarray
        norm of physical state x
    z_norm : ndarray
        norm of latent state z
    """

    z_norm = np.linalg.norm(z, axis=1)
    x_norm = np.linalg.norm(x, axis=1)
    mask = x_norm != 0
    z_norm = z_norm[mask]
    x_norm = x_norm[mask]
    operator_norm = z_norm / x_norm
    return operator_norm, x_norm, z_norm


def compute_encoder_data(model, scenario, rawdata, verbose=2):
    """Gets input and output data from encoder and residuals

    Parameters
    ----------
    model : keras model
        full LRAN autoencoder model
    scenario : dict
        training hyperparams and settings
    rawdata : dict or str
        raw unprocessed data or path to pkl
    verbose : int
        level of verbosity

    Returns
    -------
    encoder_data : dict
        valdata : dict of ndarray
            validation data used
        normalization_dict : dict
            normalization parameters used
        x0, x1 : ndarray
            physical state at time 0, 1
        z0, z1 : ndarray
            latent state at time 0, 1
        u0 : ndarray
            physical input
        v0 : ndarray
            encoded input
        dx : ndarray
            residual in physical state
        dz : ndarray
            residual in latent state
    """
    if isinstance(rawdata, str):
        traindata, valdata, normalization_dict = process_data(
            rawdata,
            scenario["sig_names"],
            None,  # scenario["normalization_method"],
            scenario["window_length"],
            scenario["window_overlap"],
            0,  # scenario["lookback"],
            scenario["lookahead"],
            scenario["sample_step"],
            scenario["uniform_normalization"],
            1,  # scenario['train_frac'],
            0,  # scenario['val_frac'],
            scenario["nshots"],
            verbose,
            scenario["flattop_only"],
            randomize=False,
            pruning_functions=scenario["pruning_functions"],
            invert_q=scenario["invert_q"],
            excluded_shots=scenario["excluded_shots"],
            val_idx=0,
        )
        del traindata
    else:
        valdata = rawdata
        normalization_dict = {}

    nsamples = len(valdata["time"])
    nsamples -= nsamples % scenario["batch_size"]
    # parse data into timesteps / arrays
    xk_dict = {
        key: (
            valdata[key][
                :nsamples,
                : scenario["lookahead"] + 1,
                :: scenario["profile_downsample"],
            ]
            if valdata[key].ndim == 3
            else valdata[key][:nsamples, : scenario["lookahead"] + 1]
        )
        for key in (scenario["profile_names"] + scenario["scalar_names"])
    }

    uk_dict = {
        key: (
            valdata[key][:nsamples, : scenario["lookahead"] + 1, ::2]
            if valdata[key].ndim == 3
            else valdata[key][:nsamples, : scenario["lookahead"] + 1]
        )
        for key in (scenario["actuator_names"])
    }

    lran = LRANMPC(model, scenario)

    if verbose:
        print("computing sigma")

    sigma = compute_sigma_metric(lran, xk_dict, uk_dict)

    if verbose:
        print("Encoding")

    # encode data
    xk_dict_norm = lran.normalize(xk_dict)
    uk_dict_norm = lran.normalize(uk_dict)
    xk = np.concatenate(
        [
            xk_dict_norm[sig].reshape((nsamples, scenario["lookahead"] + 1, -1))
            for sig in lran.state_names
        ],
        axis=-1,
    )
    uk = np.concatenate(
        [
            uk_dict_norm[sig].reshape((nsamples, scenario["lookahead"] + 1, -1))
            for sig in lran.actuator_names
        ],
        axis=-1,
    )

    zk = lran.encode(xk_dict_norm)
    vk = np.stack([uk_dict_norm[sig] for sig in scenario["actuator_names"]], axis=-1)
    vk = np.moveaxis(vk, 0, 1)
    # compute residuals
    if verbose:
        print("Predicting")
    zkp = lran.predict_latent(zk[:, 0, :], vk)
    vk = np.moveaxis(vk, 0, 1)
    zkp = np.moveaxis(zkp, 0, 1)
    dz = zkp[:, :-1, :] - zk

    if verbose:
        print("Decoding")
    xkp = lran.decode(zkp)
    xkp = np.concatenate(
        [xkp[sig].reshape(zkp.shape[:2] + (-1,)) for sig in lran.state_names],
        axis=-1,
    )
    dx = xkp[:, :-1, :] - xk

    encoder_data = {
        "valdata": valdata,
        "normalization_dict": normalization_dict,
        "x0": xk[:, 0, :],
        "x1": xk[:, 1, :],
        "z0": zk[:, 0, :],
        "z1": zk[:, 1, :],
        "u0": uk[:, 0, :],
        "v0": vk[:, 0, :],
        "xp0": xkp[:, 0, :],
        "xp1": xkp[:, 1, :],
        "zp0": zkp[:, 0, :],
        "zp1": zkp[:, 1, :],
        "dx": dx,
        "dz": dz,
        "sigma": sigma,
    }
    return encoder_data


def compute_norm_data(x, z, nsamples=None, workers=1, verbose=True):
    """Computes data, operator, and lipschitz norms

    Parameters
    ----------
    x : ndarray, shape(nsamples, nstates)
        physical state
    z : ndarray, shape(nsamples, nlatentstates)
        latent state
    workers : int, optional
        how many parallel processes to work
    verbose : bool
        whether to display progress bar

    Returns
    -------
    norm_data : dict of ndarray
        operator_norm : ndarray
            induced norm of the encoding operator
        x_norm : ndarray
            norm of physical state x
        z_norm : ndarray
            norm of latent state z
        lipschitz_constant : ndarray
            ratio of norm of differences ||z - z'||/||x - x'||
        delta_x_norms : ndarray
            norm of differences in physical state
        delta_z_norms : ndarray
            norm of differences in latent state
    """
    if nsamples is None:
        nsamples = len(x)

    operator_norm, x0_norm, z0_norm = compute_operator_norm(x[:nsamples], z[:nsamples])
    lipschitz_constant, delta_x0_norms, delta_z0_norms = compute_lipschitz_constant(
        x[: int(np.sqrt(nsamples))], z[: int(np.sqrt(nsamples))], workers, verbose
    )

    norm_data = {
        "lipschitz_constant": lipschitz_constant,
        "delta_x0_norms": delta_x0_norms,
        "delta_z0_norms": delta_z0_norms,
        "operator_norm": operator_norm,
        "x0_norm": x0_norm,
        "z0_norm": z0_norm,
    }
    return norm_data


def compute_sigma_metric(lran, x, u):
    """Computes sigma metric for comparison to transport codes

    sigma = norm(error)/norm(profile)

    Parameters
    ----------
    lran : LRANMPC objet
        model/predictor class
    x : dict of ndarray, shape (samples, times, space)
        input data for profiles and scalars
    u : dict of ndarray, shape (samples, times, 1)
        actuator inputs to model

    Returns
    -------
    sigma : dict of ndarray, shape (samples, times)
        sigma metric for each profile/scalar over time
    """
    # make sure scalars are treated correctly
    for key in x.keys():
        if x[key].ndim == 2:
            x[key] = x[key][:, :, None]

    x0 = {key: val[:, 0] for key, val in x.items()}
    xp = lran.predict(x0, u)
    for key in xp.keys():
        if xp[key].ndim == 2:
            xp[key] = xp[key][:, :, None]

    num = {key: np.linalg.norm(x[key] - xp[key][:, :-1], axis=-1) for key in x.keys()}
    den = {key: np.linalg.norm(x[key], axis=-1) for key in x.keys()}

    # avoid dividing by zero
    mask = ~np.any([np.isclose(den[key], 0) for key in x.keys()], axis=(0, -1))
    sigma = {key: num[key][mask] / den[key][mask] for key in x.keys()}

    return sigma
