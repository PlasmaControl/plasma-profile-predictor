import time
print(time.clock())
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.models import Model
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle
import keras
import tensorflow as tf
from keras import backend as K
import sys
sys.path.append(os.path.abspath('../'))
import helpers
from helpers.data_generator import process_data, AutoEncoderDataGenerator, DataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from time import strftime, localtime
import matplotlib
import matplotlib.gridspec as gridspec
import copy
from tqdm import tqdm_notebook
from helpers.normalization import normalize, denormalize, renormalize
import scipy
from keras.utils.vis_utils import model_to_dot
from IPython.display import Image, display
from helpers.custom_init import downsample
from helpers.custom_reg import groupLasso
import helpers
from tqdm import tqdm
from cvxopt import matrix
from cvxopt import solvers
from scipy.optimize import minimize


# Extract data from model

def get_AB(model):
    A = model.get_layer('AB_matrices').get_weights()[1].T
    B = model.get_layer('AB_matrices').get_weights()[0].T
    return A,B 
def get_submodels(model):
    from keras.models import Model
    state_encoder = model.get_layer('state_encoder_time_dist').layer.layers[-1]
    control_encoder = model.get_layer('ctrl_encoder_time_dist').layer.layers[-1]
    state_decoder = model.get_layer('state_decoder_time_dist').layer
#     control_decoder = model.get_layer('ctrl_decoder_time_dist').layer
    return state_encoder, state_decoder, control_encoder
        
def get_state_and_inputs(scenario,inputs,**kwargs):
    state_inputs = {}
    x0 = {}
    for sig in scenario['profile_names']+scenario['scalar_names']:
        state_inputs[sig] = np.squeeze(inputs[0]['input_'+sig])
        if sig in scenario['profile_names']:
            x0['input_'+sig] = inputs[0]['input_'+sig][0][0].reshape((1,1,scenario['profile_length']))
        else:
            x0['input_'+sig] = inputs[0]['input_'+sig][0][0].reshape((1,1,1))
    
    control_inputs = {}
    for sig in scenario['actuator_names']:
        control_inputs['input_'+sig] = inputs[0]['input_'+sig]
    return x0, control_inputs, state_inputs

def encode_state_and_inputs(state_encoder,control_encoder,scenario,x0,control_inputs,**kwargs):
    # encode control
    T = scenario['lookback'] + scenario['lookahead']
    u = []
    for i in range(T):
        temp_input = {k:v[:,i].reshape((1,1,1)) for k,v in control_inputs.items()}
        u.append(np.squeeze(control_encoder.predict(temp_input)))
        
    # encode state and propogate
    x0 = np.squeeze(state_encoder.predict(x0))
    return x0, u
    
def decode_state(state_decoder,x):
    return state_decoder.predict(x[np.newaxis,:])


def decode_inputs(control_decoder, inputs):
    return control_decoder.predict(inputs)

def get_final_state(state_encoder,scenario,inputs,**kwargs):
    state_inputs = {}
    xf = {}
    for sig in scenario['profile_names']+scenario['scalar_names']:
        state_inputs[sig] = np.squeeze(inputs[0]['input_'+sig])
        if sig in scenario['profile_names']:
            xf['input_'+sig] = inputs[0]['input_'+sig][0][-1].reshape((1,1,scenario['profile_length']))
        else:
            xf['input_'+sig] = inputs[0]['input_'+sig][0][-1].reshape((1,1,1))
    
    xf_enc = np.squeeze(state_encoder.predict(xf))
    return xf, xf_enc

def get_state_predictions(scenario,x_dec):
    state_predictions = {}
    for i, sig in enumerate(scenario['profile_names']):
        lenvar = len(x_dec[0])-len(scenario['scalar_names'])
        state_predictions[sig] = np.squeeze(x_dec[0][i*33:(i+1)*33])
    return state_predictions


def get_model_prediction(x0, control_inputs, u_mpc, A, B, n):
    # generates mpc and autoencoder predictions
    # both actual control_inputs and u_mpc are designed to be optional
    
    state_pred_arr = []
    mpc_pred_arr = []
    mpc_enc_pred_arr = []
    state_enc_pred_arr = []
    x_mpc = x0
    # Propagate through model
    for i in range(0, n):
        if (control_inputs is not None):
#         x0 = (A @ x0  + B @ control_inputs[i])
            x0 = (x0 @ A.T  + control_inputs[i] @ B.T)
            state_enc_pred_arr.append(x0)
            state_pred_arr.append(get_state_predictions(scenario,decode_state(state_decoder,x0)))
        if (u_mpc is not None):
            x_mpc = (x_mpc @ A.T  + u_mpc[i] @ B.T)
            mpc_enc_pred_arr.append(x_mpc)
            mpc_pred_arr.append(get_state_predictions(scenario,decode_state(state_decoder,x_mpc)))

    # print("State_predictions: {}".format(state_predictions))
    return state_pred_arr, mpc_pred_arr, state_enc_pred_arr, mpc_enc_pred_arr

def get_pred(index, n, plot = False, filtered = False, mpc_filtered = False):
    # Generates predictions for autoencoder and mpc (with autoencoder model) for data at index index
    # n is the forecast horizon
    # Plotting the data is optional; enabling the butterworth filter is also optional
    
    profiles = scenario['profile_names']
    temp_x = enc_x[index]
    temp_u = enc_inputs[index]

    N = temp_x.shape[0]
    M = enc_inputs.shape[2]

    # Generate Q,R
    Q = np.eye(N)*1e10
    R = np.eye(M)*1e-15

    # Run MPC
    #sol_mpc = solve_Neo_MPC_system(Q,R,A,B,temp_x,enc_XF[index],n)
    sol_mpc = brute_force_opt_system(A, B, temp_x, enc_XF[index], n)
    sol_mpc = np.array(sol_mpc['x'])
    temp_u_mpc = sol_mpc.reshape(n, M)
    state_pred, mpc_pred, state_enc_pred_arr, mpc_enc_pred_arr = get_model_prediction(temp_x, temp_u, temp_u_mpc, A, B, n)
    
    return state_pred, mpc_pred, state_enc_pred_arr, mpc_enc_pred_arr, temp_u_mpc

def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_discrete_are
    return solve_discrete_are(A, B, Q, R) 

def solve_Neo_MPC_system(Q,R,A,B,x0,xf,n):
    
    
    # Define parameters
    N = A.shape[0]
    M = B.shape[1]
#     print("N: {}".format(N))
#     print("M: {}".format(M))
    
    # Reshape to avoid complications
    x0 = x0.reshape((N,1))
    xf = xf.reshape((N,1))

    ############################### Generate Matrices ####################################

    # Generate Matrix M
    M_bar = np.zeros((N * n, N))
    rsl = slice(0, N)
    M_bar[rsl, :N] = A

    for i in range(1, n):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        M_bar[rsl, :N] = A @ M_bar[rsl_p, :N]
    
#     print("M_bar: {}".format(M_bar))
    
    # Generate Q_bar
    Q_bar = np.zeros((N * n, N * n))
    rsl = slice(0, N)
    Q_bar[rsl, :N] = Q

    for i in range(1, n-1):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        Q_bar[rsl, N : (i + 1) * N] = Q_bar[rsl_p, : i * N]
    
    #Q_N = lqr(A,B,Q,R)
    rsl = slice((n-1) * N, n * N)
    #Q_bar[rsl, rsl] = Q_N
    Q_bar[rsl, rsl] = Q
    
#     print("Q_bar: {}".format(Q_bar))

    # Generate R_bar
    R_bar = np.kron(np.eye(n),R)
    
#     print("R_bar: {}".format(R_bar))

    # Generate V
    V = np.zeros((N * n, n * M))
    rsl = slice(0, N)
    V[rsl, :M] = B #Make first line

    for i in range(1, n):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        V[rsl, :M] = A @ V[rsl_p, :M] # A^(N-1)*B
        V[rsl, M : (i + 1) * M] = V[rsl_p, : i * M]
    
#     print("V: {}".format(V))
        
#     # Generate L
#     L = np.zeros((N * n, N * n))
#     rsl = slice(0, N)
#     L[rsl, :N] = A #Make first line

#     for i in range(1, n):
#         rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
#         L[rsl, :N] = L[rsl_p, :N]
#         L[rsl, N : (i + 1) * N] = A @ L[rsl_p, : i * N]
    
#     print("L: {}".format(np.matrix.view(L)))
    
    #Generate X_F from x_f
    X_F = np.zeros((N*n,1))
    
    for i in range(0,n):
        rsl = slice(i * N, (i + 1)*N)
        X_F[rsl] = xf
    
#     print("XF: {}".format(X_F))
    
    # Generate D_bar, d
    D_bar = np.zeros((2*M*n,M*n))
    rsl = slice(0,M*n)
    D_bar[rsl, rsl] = np.eye(M*n)
    D_bar[slice(M*n,2*M*n), rsl] = -np.eye(M*n)

    d = np.zeros((2*M*n,1))
    
    # Limits on U
    lim = 1e15
    
    d[rsl,:] = np.ones((M*n,1))*lim
    d[slice(M*n,2*M*n),:] = np.ones((M*n,1))*lim

#     print("D_bar: {}".format(D_bar))
#     print("D: {}".format(d))
    
    # Generate F and H matrices   
    temp = np.transpose(V) @ (Q_bar)
    F = temp @ (M_bar.dot(x0) - X_F) # + L @ (X_F)
    
    temp = np.transpose(V) @ (Q_bar)
    H = temp @ (V) + R_bar
    
#     print("H: {}".format(H))
#     print("F: {}".format(F))

    ########################################### Do computations #############################

    # Define QP parameters (with NumPy)

    P = matrix(H, tc='d')
    q = matrix(F, tc='d')
    G = matrix(D_bar, tc='d')
    h = matrix(d, tc='d')

    ######################################### Print Solution ###############################
    # Suppress Output
    solvers.options['show_progress'] = False
    
    
    # Construct the QP, invoke solver
    sol = solvers.qp(P,q, G, h)

    return sol


  
def objFun(proposals, M_bar, V, X_F, x0, N) -> float:
    
    #compute future states
    X = (M_bar @ x0) + (V @ proposals.reshape(n*M, 1))
    
    #Over each timestep, compute mean absolute error over transformed spatial points, then add
#     errs = np.abs(X-X_F)
#     totalErr = 0
#     for i in range(0, n):
#         totalErr += np.mean(errs[i*N:(i+1)*N])

    errs = np.abs(X-X_F)
    totalErr = np.mean(errs[(n-1)*N:]) * 1e5
    return totalErr


def brute_force_opt_system(A, B, x0, xf, n):
    N = A.shape[0]
    M = B.shape[1]
    
    
    x0 = x0.reshape((N,1))
    xf = xf.reshape((N,1))
    # Generate Matrix M
    M_bar = np.zeros((N * n, N))

    for i in range(0, n):
        rsl = slice(i * N, (i + 1) * N)
        M_bar[rsl, :N] = np.linalg.matrix_power(A, i+1)
    
    #Generate V
    V = np.zeros((N * n, n * M))
    rsl = slice(0, N)
    V[rsl, :M] = B #Make first line

    for i in range(1, n):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        V[rsl, :M] = A @ V[rsl_p, :M] # A^(N-1)*B
        V[rsl, M : (i + 1) * M] = V[rsl_p, : i * M]
        
    #Generate X_F from x_f
    X_F = np.zeros((N*n,1))
    
    for i in range(0,n):
        rsl = slice(i * N, (i + 1)*N)
        X_F[rsl] = xf
        
    initial_guess = np.zeros(n * M).reshape(n*M, 1)
    #real_u = real_u.reshape(n*M,1)
    actuator_bounds = [(-2,2)] * n * M
    
    soln = minimize(objFun, initial_guess, args=(M_bar, V, X_F, x0,N), bounds=actuator_bounds)
    
    return soln

if __name__ == '__main__':

    print("test")
    model_path = '/scratch/gpfs/aaronwu/run_results_06_27_21/model-autoencoder_LA-6_27Jun21-12-25_Scenario-6.h5'
    scenario_path = '/scratch/gpfs/aaronwu/run_results_06_27_21/model-autoencoder_LA-6_27Jun21-12-25_Scenario-6_params.pkl'
    print("test")
    K.clear_session()
    # Load Model
    model = keras.models.load_model(model_path)
    with open(scenario_path, 'rb') as f:
        scenario = pickle.load(f, encoding='latin1')
    print("test")

    A,B = get_AB(model)
    print("A: " + str(A.shape))
    print("B: " + str(B.shape))
    print("actuator_names : {}".format(scenario['actuator_names']))
    print("profile_names : {}".format(scenario['profile_names']))
    state_encoder, state_decoder, control_encoder = get_submodels(model)



    datapath = '/scratch/gpfs/jabbate/full_data_with_error/train_data.pkl'
    with open(datapath,'rb') as f:
        rawdata = pickle.load(f,encoding='latin1')

    traindata, valdata, normalization_dict = process_data(rawdata,
                                                                  scenario['sig_names'],
                                                                  scenario['normalization_method'],
                                                                  scenario['window_length'],
                                                                  scenario['window_overlap'],
                                                                  scenario['lookback'],
                                                                  scenario['lookahead'],
                                                                  scenario['sample_step'],
                                                                  scenario['uniform_normalization'],
                                                                  1,
                                                                  0,
                                                                  scenario['nshots'],
                                                                  2,
                                                                  scenario['flattop_only'],
                                                                  pruning_functions=scenario['pruning_functions'],
                                                                  invert_q = scenario['invert_q'], #scenario.get('invert_q'),
                                                                  val_idx = 0,
                                                                  excluded_shots=scenario['excluded_shots'],
                                                                randomize=False)
    valdata = denormalize(valdata, normalization_dict)
    valdata = renormalize(valdata, scenario['normalization_dict'])
    generator = AutoEncoderDataGenerator(valdata,
                                                   1,  
                                                   scenario['profile_names'],
                                                   scenario['actuator_names'],
                                                   scenario['scalar_names'],
                                                   scenario['lookback'],
                                                   scenario['lookahead'],
                                                   scenario['profile_downsample'],
                                                   scenario['state_latent_dim'],
                                                   scenario['discount_factor'],
                                                   scenario['x_weight'],
                                                   scenario['u_weight'],                                            
                                                   False) #scenario['shuffle_generators'])



    x = []
    XF = []
    enc_x = []
    enc_XF = []
    state_inputs_arr = []
    inputs = []
    enc_inputs = []

    for i in range(len(generator)):
        #print("Completed {} out of {}".format(i+1,len(generator)),end='\n')

        # Get state and input from generators
        temp_x, temp_con_inputs, state_inp = get_state_and_inputs(scenario,generator[i])
        x.append(temp_x)
        state_inputs_arr.append(state_inp)

        # Append control inputs
        inputs.append(temp_con_inputs)

        # Encode state and inputs
        temp_x, temp_con_inputs = encode_state_and_inputs(state_encoder,control_encoder,scenario,temp_x,temp_con_inputs)
        enc_x.append(temp_x)
        enc_inputs.append(temp_con_inputs)

        # Get final encoded state for MPC
        exf, exf_enc = get_final_state(state_encoder,scenario,generator[i])
        XF.append(exf)
        enc_XF.append(exf_enc)


    enc_XF = np.array(enc_XF)
    enc_inputs = np.array(enc_inputs)
    enc_x = np.array(enc_x)
    print("Enc Input: {}".format(enc_inputs.shape))
    print("Enc_x: {}".format(enc_x.shape))

#     with open('/home/aaronwu/plasma-profile-predictor/Controls/finalized_data.pkl', 'rb') as f:
#         enc_XF, enc_inputs, enc_x, state_inputs_arr = pickle.load(f)
    
    n = scenario['lookahead']

    mpc_prof_errors = {sig:[] for sig in scenario['profile_names']}
    state_prof_errors = {sig:[] for sig in scenario['profile_names']}
    mpc_enc_prof_errors = []
    state_enc_prof_errors = []
    mpc_out = []


    for j in range(0, enc_x.shape[0]):
        print("Completed {} of {}".format(j,enc_x.shape[0]),end='\n')
        state_pred_profiles, mpc_pred_profiles, state_encoded_profiles, mpc_encoded_profiles, u_mpc = get_pred(j, n, plot = False)

        #enc_x.shape[0]
        for profile in scenario['profile_names']:

            # True profile
            true = state_inputs_arr[j][profile][-1].squeeze()
            true_enc = enc_XF[j]
            #true = helpers.normalization.denormalize_arr(true_norm,scenario['normalization_dict'][profile])

            # MPC Predictions
            mpc_pred = mpc_pred_profiles[-1][profile]; # Keep only final timestep
            mpc_enc_pred = mpc_encoded_profiles[-1]
            #mpc_pred = helpers.normalization.denormalize_arr(mpc_pred[profile],scenario['normalization_dict'][profile])

            # Autoencoder predictions
            state_pred = state_pred_profiles[-1][profile]; # Keep only final timestep
            state_enc_pred = state_encoded_profiles[-1]
            #state_pred = helpers.normalization.denormalize_arr(state_pred[profile],scenario['normalization_dict'][profile])

            # Autoencoder predictions error
            state_pred_err = np.abs(true-state_pred)
            state_enc_err = np.abs(true_enc - state_enc_pred)

            #MPC predictions error 
            mpc_pred_err = np.abs(true-mpc_pred)
            mpc_enc_err = np.abs(true_enc-mpc_enc_pred)

            mpc_prof_errors[profile].append(mpc_pred_err)
            state_prof_errors[profile].append(state_pred_err)
            mpc_enc_prof_errors.append(mpc_enc_err)
            state_enc_prof_errors.append(state_enc_err)
            mpc_out.append(u_mpc)


    with open('/home/aaronwu/plasma-profile-predictor/Controls/mpcDataDump/BruteForceLatentDim65.pkl', 'wb') as f:
        pickle.dump([mpc_prof_errors, state_prof_errors, mpc_enc_prof_errors, state_enc_prof_errors, mpc_out], f)
