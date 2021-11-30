import numpy as np
import scipy
import osqp
import matplotlib
import matplotlib.pyplot as plt


class LRANMPC:
    """Helper class for doing MPC with LRAN model

    Parameters
    ----------
    model : keras model
        should be LRAN type with encoders/decoders etc
    params : dict
        parameters used for training, ie "scenario"
    xtarget : dict of ndarray
        target state
    Q, R : ndarray
        weight matrices
    Nlook : int
        mpc lookahead
    umin, umax : ndarray
        bounds on control
    terminal : bool
        whether to include terminal cost in objective
    """

    def __init__(
        self,
        model,
        params,
        xtarget=None,
        Q=None,
        R=None,
        Nlook=None,
        umin=None,
        umax=None,
        terminal=True,
    ):

        self._model = model
        self._params = params
        self._xtarget = xtarget
        self._parse_model()
        self._nu = self.B.shape[1]
        self._nz = self.A.shape[0]
        self._terminal = terminal

        self.R = R
        self.Q = Q
        if Nlook is None:
            Nlook = 10
        self._Nlook = Nlook
        if umin is None:
            umin = -np.inf * np.ones(self._nu)
        self._umin = umin
        if umax is None:
            umax = np.inf * np.ones(self._nu)
        self._umax = umax

        self._zmin = -np.inf * np.ones(self._nz)
        self._zmax = np.inf * np.ones(self._nz)

        self.mpc_setup()

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

    def _parse_model(self):
        self._A, self._B = get_AB(self._model)

        (
            self._state_encoder,
            self._control_encoder,
            self._state_decoder,
            self._control_decoder,
        ) = get_submodels(self._model)

    def _normalize(self, data):
        out = {}
        for key, val in data.items():
            out[key] = (
                val - self._params["normalization_dict"][key]["mean"]
            ) / self._params["normalization_dict"][key]["std"]
        return out

    def _denormalize(self, data):
        out = {}
        for key, val in data.items():
            out[key] = (
                val * self._params["normalization_dict"][key]["std"]
                + self._params["normalization_dict"][key]["mean"]
            )
        return out

    def mpc_setup(self, Q=None, R=None, Nlook=None):
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

        # E = [A, A**2, A**3,...]
        # F = [B 0 0 ...]
        #     [AB B 0 0 ...]
        #     [A^2B AB B 0 0 ...]
        E = np.vstack([np.linalg.matrix_power(A, i + 1) for i in range(Nlook)])
        F = []
        for i in range(Nlook):
            Frow = np.hstack(
                [np.linalg.matrix_power(A, j) @ B for j in range(i + 1)][::-1]
            )
            F.append(
                np.hstack([Frow, np.zeros((A.shape[0], (Nlook - i - 1) * B.shape[1]))])
            )

        F = np.vstack(F)
        nx = A.shape[0]
        nu = B.shape[1]
        Ix = np.vstack([np.eye(Nlook * nx), -np.eye(Nlook * nx)])
        Iu = np.vstack([np.eye(Nlook * nu), -np.eye(Nlook * nu)])
        Aineq_x = Ix @ F
        Aineq_u = Iu
        Aineq = np.vstack([Aineq_x, Aineq_u])

        # expand cost matrices
        Qhat = [Q] * Nlook
        Rhat = [R] * Nlook

        if self._terminal:
            P = scipy.linalg.solve_discrete_are(A, B, Q, R)
            Qhat[-1] = P
        Qhat = scipy.linalg.block_diag(*Qhat)
        Rhat = scipy.linalg.block_diag(*Rhat)

        H = F.T @ Qhat @ F + Rhat
        # symmetrize for accuracy
        H = (H + H.T) / 2
        qp = osqp.OSQP()
        lu = np.ones(Aineq.shape[0])
        qp.setup(
            P=scipy.sparse.csc_matrix(H),
            q=np.ones(H.shape[1]),
            A=scipy.sparse.csc_matrix(Aineq),
            l=-lu,
            u=lu,
            verbose=False,
        )

        self._qp = qp
        self._Qhat = Qhat
        self._E = E
        self._F = F
        self._H = H

    def encode(self, state):
        """Encodes physical state to latent linear state

        Parameters
        ----------
        state: dict of ndarray
            dictionary of arrays, keys should be names of profiles

        Returns
        -------
        latent_state : ndarray
            latent linear state

        """
        return self._state_encoder.predict(
            {"input_" + key: val for key, val in state.items()}
        )

    def decode(self, latent_state):
        """Decodes physical state to latent linear state

        Parameters
        ----------
        latent_state : ndarray
            latent linear state

        Returns
        -------
        state: dict of ndarray
            dictionary of arrays, keys should be names of profiles

        """
        return self._state_decoder.predict(latent_state)

    def mpc_action(
        self, t, xk, xtarget=None, umin=None, umax=None, zmin=None, zmax=None
    ):
        """find control action via MPC

        Parameters
        ----------
        xk : dictionary of ndarray
            current value for the physical state
        xtarget : dictionary of ndarray
            target value for physical state
        umin, umax : ndarray
            upper and lower bounds on control input
        zmin, zmax : ndarray
            upper and lower bounds on latent state

        Returns
        -------
        u : ndarray
            action to take at current timestep
        uhat : ndarray
            actions to take over prediction horizon
        """

        # TODO: allow bounds and target to be time varying

        # set values
        if xtarget is None:
            xtarget = self._xtarget
        if umin is None:
            umin = self._umin
        if umax is None:
            umax = self._umax
        if zmin is None:
            zmin = self._zmin
        if zmax is None:
            zmax = self._zmax

        # normalize actuator bounds
        umin = self._normalize(
            {key: umin[i] for i, key in enumerate(self._params["actuator_names"])}
        )
        umin = np.array([umin[key] for key in self._params["actuator_names"]])
        umax = self._normalize(
            {key: umax[i] for i, key in enumerate(self._params["actuator_names"])}
        )
        umax = np.array([umax[key] for key in self._params["actuator_names"]])

        # encode state and target
        xk = self._normalize(xk)
        zk = self.encode(xk).squeeze()

        if xtarget is not None:
            xtarget = self._normalize(xtarget)
            ztarget = self.encode(xtarget).squeeze()
        else:  # default target : encoded state of 0, roughly mean value for physical state
            ztarget = np.zeros(self._nz)

        # set up inequality constraints
        Ix = np.concatenate(
            [np.eye(self._Nlook * self._nz), -np.eye(self._Nlook * self._nz)]
        )
        bineq_z = (
            np.concatenate(
                [np.tile(zmax, (self._Nlook,)), -np.tile(zmin, (self._Nlook,))]
            )
            - Ix @ self._E @ zk
        )
        bineq_u = np.concatenate(
            [np.tile(umax, (self._Nlook,)), -np.tile(umin, (self._Nlook,))]
        )
        bineq = np.concatenate([bineq_z, bineq_u])

        # form qp
        rhat = np.tile(ztarget, (self._Nlook,))
        ft = (zk.T @ self._E.T - rhat.T) @ self._Qhat @ self._F
        f = ft.T

        # solve qp
        self._qp.update(q=f, l=-np.inf * np.ones_like(bineq), u=bineq)
        results = self._qp.solve()
        uhat = results.x
        u = uhat[: self._nu].reshape((-1, 1))

        # parse into output dict in physical units
        control = {}
        for i, sig in enumerate(self._params["actuator_names"]):
            control[sig] = u[i]

        control = self._denormalize(control)
        return control


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
    state_encoder = model.get_layer("state_encoder_time_dist").layer.layers[-1]
    control_encoder = model.get_layer("ctrl_encoder_time_dist").layer.layers[-1]
    state_decoder = model.get_layer("state_decoder_time_dist").layer.layers[-1]
    control_decoder = model.get_layer("ctrl_decoder_time_dist").layer.layers[-1]

    return state_encoder, state_decoder, control_encoder, control_decoder


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


def compute_ctrb(A, B):
    """Compute controllability matrix

    Parameters
    ----------
    A : ndarray, shape(N,N)
        system dynamics matrix
    B : ndarray, shape(N,M)
        system control matrix

    Returns
    -------
    C : ndarray, shape(N, N*M)
        controllability matrix
    k : int
        rank of controllability matrix
    cols : int
        number of columns needed for rank(C) = k
    """
    C = np.hstack(
        [B] + [(np.linalg.matrix_power(A, i) @ B) for i in range(1, A.shape[0])]
    )
    k = np.linalg.matrix_rank(C)

    # find out how many cols are needed for full rank
    cols = k - 1
    ki = 0
    while ki < k:
        cols += 1
        ki = np.linalg.matrix_rank(C[:, :cols])

    return C, k, cols


def compute_grammian(A, B, k=None):
    """Compute discrete time controllability grammian

    Parameters
    ----------
    A : ndarray, shape(N,N)
        system dynamics matrix
    B : ndarray, shape(N,M)
        system control matrix
    k : int, optional
        number of timesteps for finite horizon. If None or np.inf, defaults to infinite horizon

    Returns
    -------
    Wc : ndarray, shape(N,N)
        discrete time controllability grammian
    """

    if k == np.inf or k is None:
        Wc = scipy.linalg.solve_discrete_lyapunov(A, B @ B.T)

    else:  # finite horizon
        Wc = np.zeros_like(A)
        Ai = np.eye(A.shape[0])
        for i in range(k):
            AiB = Ai @ B
            Wc += AiB @ AiB.T
            Ai = Ai @ A
    return Wc
