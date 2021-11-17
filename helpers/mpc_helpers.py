import numpy as np
import scipy
import osqp





class LRANMPC():
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
    """
    
    def __init__(self, model, params, xtarget=None, Q=None, R=None, Nlook=None, umin=None, umax=None):
        
        self._model = model
        self._params = params
        self._xtarget = xtarget
        self._parse_model()
        self._nu = self._B.shape[1]
        self._nz = self._A.shape[0]
        
        
        if Q is None:
            Q = np.eye(self._nz)
        self._Q = Q
        if R is None:
            R = np.eye(self._nu)
        self._R = R
        if Nlook is None:
            Nlook = 10
        self._Nlook = Nlook
        if umin is None:
            umin = -np.inf*np.ones(self._nu)
        self._umin = umin
        if umax is None:
            umax = np.inf*np.ones(self._nu)
        self._umax = umax

            
        self._zmin = -np.inf*np.ones(self._nz)
        self._zmax = np.inf*np.ones(self._nz)
        
        self.mpc_setup()
        
        
    def _parse_model(self):
        self._A = self._model.get_layer('AB_matrices').get_weights()[1].T
        self._B = self._model.get_layer('AB_matrices').get_weights()[0].T
        
        from keras.models import Model
        self._state_encoder = self._model.get_layer('state_encoder_time_dist').layer.layers[-1]
        self._control_encoder = self._model.get_layer('ctrl_encoder_time_dist').layer.layers[-1]
        self._state_decoder = self._model.get_layer('state_decoder_time_dist').layer.layers[-1]
        self._control_decoder = self._model.get_layer('ctrl_decoder_time_dist').layer.layers[-1]
        # self._state_decoder = Model(self._model.get_layer('state_decoder_time_dist').layer.layers[0].input,
        #                       self._model.get_layer('state_decoder_time_dist').layer.layers[-2].get_output_at(1),
        #                      name='state_decoder')    
        # self._control_decoder = Model(self._model.get_layer('ctrl_decoder_time_dist').layer.layers[0].input,
        #                         self._model.get_layer('ctrl_decoder_time_dist').layer.layers[-2].get_output_at(1),
        #                        name='control_decoder')

        
    def _normalize(self, data):
        out = {}
        for key, val in data.items():
            out[key] = (val - self._params['normalization_dict'][key]['mean'])/self._params['normalization_dict'][key]['std']
        return out
        
    def _denormalize(self, data):
        out = {}
        for key, val in data.items():
            out[key] = val * self._params['normalization_dict'][key]['std'] + self._params['normalization_dict'][key]['mean']
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
        
        A = self._A
        B = self._B
        if Q is None:
            Q = self._Q
        else:
            self._Q = Q
        if R is None:
            R = self._R
        else:
            self._R = R
        if Nlook is None:
            Nlook = self._Nlook
        else:
            self._Nlook = Nlook
            
        # E = [A, A**2, A**3,...]
        # F = [B 0 0 ...]
        #     [AB B 0 0 ...]
        #     [A^2B AB B 0 0 ...]
        E = np.vstack([np.linalg.matrix_power(A, i+1) for i in range(Nlook)])    
        F = []
        for i in range(Nlook):
            Frow = np.hstack([np.linalg.matrix_power(A, j) @ B for j in range(i+1)][::-1])
            F.append(np.hstack([Frow, np.zeros((A.shape[0],(Nlook-i-1)*B.shape[1]))]))

        F = np.vstack(F)
        nx = A.shape[0]
        nu = B.shape[1]
        Ix = np.vstack([np.eye(Nlook*nx), -np.eye(Nlook*nx)])
        Iu = np.vstack([np.eye(Nlook*nu), -np.eye(Nlook*nu)])
        Aineq_x = Ix @ F
        Aineq_u = Iu
        Aineq = np.vstack([Aineq_x, Aineq_u])

        P = scipy.linalg.solve_discrete_are(A,B,Q,R)
        # expand cost matrices
        Qhat = [Q]*Nlook;
        Rhat = [R]*Nlook;
        Qhat[-1] = P
        Qhat = scipy.linalg.block_diag(*Qhat)
        Rhat = scipy.linalg.block_diag(*Rhat)

        H = F.T @ Qhat @ F + Rhat
        # symmetrize for accuracy
        H = (H+H.T)/2;
        qp = osqp.OSQP()
        lu = np.ones(Aineq.shape[0])
        qp.setup(P=scipy.sparse.csc_matrix(H), q=np.ones(H.shape[1]), A=scipy.sparse.csc_matrix(Aineq), l=-lu, u=lu, verbose=False)

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
        return self._state_encoder.predict({"input_" + key : val for key, val in state.items()})
    
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
        
    def mpc_action(self, t, xk, xtarget=None, umin=None, umax=None, zmin=None, zmax=None):
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
        
        # encode state and target
        xk = self._normalize(xk)
        zk = self.encode(xk).squeeze()

        if xtarget is not None:
            xtarget = self._normalize(xtarget)
            ztarget = self.encode(xtarget).squeeze()
        else: # default target : encoded state of 0, roughly mean value for physical state
            ztarget = np.zeros(self._nz)
                
        # set up inequality constraints
        Ix = np.concatenate([np.eye(self._Nlook*self._nz), -np.eye(self._Nlook*self._nz)])
        bineq_z = np.concatenate([np.tile(zmax, (self._Nlook,)), -np.tile(zmin, (self._Nlook,))]) - Ix @ self._E @ zk 
        bineq_u = np.concatenate([np.tile(umax, (self._Nlook,)) , -np.tile(umin, (self._Nlook,))])
        bineq = np.concatenate([bineq_z, bineq_u])

        # form qp
        rhat = np.tile(ztarget,(self._Nlook,));  
        ft = (zk.T @ self._E.T - rhat.T) @ self._Qhat @ self._F
        f = ft.T

        # solve qp
        self._qp.update(q=f, l=-np.inf*np.ones_like(bineq), u=bineq)
        results = self._qp.solve()
        uhat = results.x
        u = uhat[:self._nu].reshape((-1,1))

        # parse into output dict in physical units
        control = {}
        for i, sig in enumerate(self._params["actuator_names"]):
            control[sig] = u[i]
            
        control = self._denormalize(control)
        return control
