import numpy as np
import scipy.linalg
import scipy.sparse


def qp_setup(P, A, rho, sigma):
    """Create normal equation matrix for solving QP

    Parameters
    ----------
    P : ndarray, shape(N,N)
        quadratic cost matrix
    A : ndarray, shape(M,N)
        constraint matrix
    rho : float
        step size
    sigma : float
        regularization parameters

    Returns
    -------
    G : ndarray, shape(N,N)
        inverse normal equation matrix
    """

    G = P + sigma * np.eye(P.shape[0]) + rho * A.T @ A

    return np.linalg.inv(G)


def qp_solve(G, P, q, A, l, u, rho, sigma, alpha, x0, y0, maxiter):
    """Solve quadratic problem of the form

    min 1/2 x.T P x + q.T x
    s.t   l <= Ax <= u

    Parameters
    ----------
    G : ndarray, shape(N,N)
        inverse normal equation matrix, from qp_setup
    P : ndarray, shape(N,N)
        quadratic cost matrix
    q : ndarray, shape(N,)
        linear cost vector
    A : ndarray, shape(M,N)
        constraint matrix
    l, u : ndarray, shape(M,)
        lower and upper bounds for constraints
    rho : float
        step size
    sigma : float
        regularization parameter
    alpha : float
        over-relaxation parameter
    x0 : ndarray, shape(N,)
        initial guess for decision variables
    y0 : ndarray, shape(M,)
        initial guess for lagrange multipliers
    maxiter : int
        number of iterations to run

    Returns
    -------
    x : ndarray, shape(N,)
        solution of QP
    y : ndarray, shape(M,)
        lagrange multipliers corresponding to x
    r_prim : float
        primal residual
    r_dual : float
        dual residual
    """
    xk = x0
    zk = A @ x0
    yk = y0

    r_prim = np.inf
    r_dual = np.inf
    k = 0
    while k < maxiter:
        k += 1
        w = sigma * xk - q + A.T @ (rho * zk - yk)
        xhk1 = G @ w
        zhk1 = A @ xhk1
        xk1 = alpha * xhk1 + (1 - alpha) * xk
        zk1 = np.clip(alpha * zhk1 + (1 - alpha) * zk + yk / rho, l, u)
        yk1 = yk + rho * (alpha * zhk1 + (1 - alpha) * zk - zk1)

        xk = xk1
        zk = zk1
        yk = yk1

    r_prim = np.max(np.abs(A @ xk - zk))
    r_dual = np.max(np.abs(P @ xk + q + A.T @ yk))

    return xk, yk, r_prim, r_dual


def mpc_setup(A, B, Q, R, Nlook, rho, sigma, dt, use_osqp=False):
    """setup mpc problem

    Parameters
    ----------
    A, B : ndarray
        system matrices, such that x(t+1) = A x(t) + B u
    Q, R: ndarray
        state and control weight matrices
    Nlook : int
        number of steps in the mpc lookahead
    rho : float
        step size
    sigma : float
        regularization parameter
    dt : float
        discretization time of model, in s

    Returns
    -------
    E : ndarray, shape(nz*Nlook, nz)
        initial condition prediction matrix
    F : ndarray, shape(nz*Nlook, nu*Nlook)
        control prediction matrix
    P : ndarray, shape(nu*Nlook, nu*Nlook)
        quadratic cost matrix
    G : ndarray, shape(N,N)
        inverse normal equation matrix, from qp_setup
    Ac : ndarray, shape(nu*(2Nlook-1), nu*Nlook)
        constraint matrix for bounds on variables and derivatives
    Qhat : ndarray, shape(nz*Nlook)
        diagonal of state weighting matrix
    Rhat : ndarray, shape(nu*Nlook)
        diagonal of control weighting matrix
    """
    # A.1.2
    # E = [A, A**2, A**3,...].T
    E = np.vstack([np.linalg.matrix_power(A, i + 1) for i in range(Nlook)])

    # A.1.3
    # F = [B 0 0 ...]
    #     [AB B 0 0 ...]
    #     [A^2B AB B 0 0 ...]
    F = []
    for i in range(Nlook):
        Frow = np.hstack([np.linalg.matrix_power(A, j) @ B for j in range(i + 1)][::-1])
        F.append(
            np.hstack([Frow, np.zeros((A.shape[0], (Nlook - i - 1) * B.shape[1]))])
        )
    F = np.vstack(F)

    # A.1.3
    Qhat = [Q] * Nlook
    Qhat = scipy.linalg.block_diag(*Qhat)

    # A.1.4
    Rhat = [R] * Nlook
    Rhat = scipy.linalg.block_diag(*Rhat)

    # A.1.5
    P = F.T @ Qhat @ F + Rhat
    # symmetrize for accuracy
    # P = (P + P.T) / 2

    # A.1.6
    nu = B.shape[1]
    Ac_bounds = np.eye(nu * Nlook)
    c = [-1] + [0] * (nu * (Nlook - 1) - 1)
    c = np.array(c)
    r = [-1] + [0] * (nu - 1) + [1] + [0] * (Nlook * nu - nu - 1)
    r = np.array(r)
    Ac_rate = 1 / dt * scipy.linalg.toeplitz(c, r)
    Ac = np.vstack([Ac_bounds, Ac_rate])

    if use_osqp:
        import osqp

        qp = osqp.OSQP()
        lu = np.ones(Ac.shape[0])
        qp.setup(
            P=scipy.sparse.csc_matrix(P),
            q=np.ones(P.shape[1]),
            A=scipy.sparse.csc_matrix(Ac),
            l=-lu,
            u=lu,
            verbose=False,
            adaptive_rho=False,
            scaling=10,
            rho=0.1,
            alpha=1.6,
            sigma=1e-4,
            check_termination=0,
            warm_start=True,
        )
        return qp, E, F, np.diag(Qhat), np.diag(Rhat)

    # A.1.7
    G = qp_setup(P, Ac, rho, sigma)
    return E, F, P, G, Ac, np.diag(Qhat), np.diag(Rhat)


def mpc_action(
    zk,
    ztarget,
    uhat,
    lagrange,
    uminhat,
    duminhat,
    urefhat,
    dumaxhat,
    umaxhat,
    E,
    F,
    P,
    G,
    Ac,
    Qhat,
    Rhat,
    rho,
    sigma,
    alpha,
    maxiter,
    qp=None,
):
    """find control action via MPC

    Parameters
    ----------
    zk : ndarray, shape(nz)
        current value for the state
    ztarget : ndarray, shape(nz*Nlook)
        target value now and in future
    uminhat : ndarray, shape(nu*Nlook)
        lower bounds on control input now and in future
    duminhat : ndarray, shape(nu*(Nlook-1))
        lower bounds for actuator derivatives
    urefhat : ndarray, shape(nu*Nlook)
        reference values for actuators
    dumaxhat : ndarray, shape(nu*(Nlook-1))
        upper bounds for actuator derivatives
    umaxhat : ndarray, shape(nu*Nlook)
        upper bounds on control input now and in future
    E : ndarray, shape(nz*Nlook, nz)
        initial condition prediction matrix
    F : ndarray, shape(nz*Nlook, nu*Nlook)
        control prediction matrix
    P : ndarray, shape(nu*Nlook, nu*Nlook)
        quadratic cost matrix
    G : ndarray, shape(N,N)
        inverse normal equation matrix, from qp_setup
    Ac : ndarray, shape(nu*(2Nlook-1), nu*Nlook)
        constraint matrix for bounds on variables and derivatives
    Qhat : ndarray, shape(nz*Nlook)
        diagonal of state weighting matrix
    Rhat : ndarray, shape(nu*Nlook)
        diagonal of control weighting matrix
    rho : float
        step size
    sigma : float
        regularization parameter
    alpha : float
        over-relaxation parameter
    maxiter : int
        number of iterations to run

    Returns
    -------
    u : ndarray
        action to take at current timestep
    uhat : ndarray
        actions to take over prediction horizon
    """

    # A.2.4
    lower_bound = np.concatenate([uminhat, duminhat])
    upper_bound = np.concatenate([umaxhat, dumaxhat])

    # A.2.5
    f = (E @ zk - ztarget) @ (Qhat[:, None] * F) - (Rhat * urefhat)
    #     f = 2 * f

    if qp is not None:
        qp.update(q=f, l=lower_bound, u=upper_bound)
        qp.update_settings(max_iter=maxiter)
        qp.warm_start(x=uhat, y=lagrange)
        results = qp.solve()
        uhat = results.x
        lagrange = results.y

    # A.2.6
    else:
        uhat, lagrange, r_p, r_d = qp_solve(
            P=P,
            G=G,
            q=f,
            A=Ac,
            l=lower_bound,
            u=upper_bound,
            rho=rho,
            sigma=sigma,
            alpha=alpha,
            x0=uhat,
            y0=lagrange,
            maxiter=maxiter,
        )

    return uhat, lagrange


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
