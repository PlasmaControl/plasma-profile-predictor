import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.models import Model
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle
import keras
import tensorflow as tf
from keras import backend as K
import sys
import itertools
from cvxopt import matrix
from cvxopt import solvers

import time
sys.path.append(os.path.abspath('../'))
import helpers
from helpers.data_generator import process_data, AutoEncoderDataGenerator, DataGenerator
from helpers.custom_losses import normed_mse, mean_diff_sum_2, max_diff_sum_2, mean_diff2_sum2, max_diff2_sum2, denorm_loss, hinge_mse_loss, percent_correct_sign, baseline_MAE
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from time import strftime, localtime
import matplotlib
from matplotlib import pyplot as plt
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
    
def get_AB(model):
    # Extract A, B from Autoencoder model
    A = model.get_layer('AB_matrices').get_weights()[1].T
    B = model.get_layer('AB_matrices').get_weights()[0].T
    return A,B

def get_submodels(model):
    # Get all relevant layers
    
    from keras.models import Model
    state_encoder = model.get_layer('state_encoder_time_dist').layer.layers[-1]
    control_encoder = model.get_layer('ctrl_encoder_time_dist').layer.layers[-1]
    state_decoder = model.get_layer('state_decoder_time_dist').layer
    return state_encoder, state_decoder, control_encoder


def encode_state_and_inputs(state_encoder,control_encoder,scenario,x0,control_inputs,**kwargs):
    # encode control
    T = scenario['lookahead']
    u = []
    for i in range(T):
        temp_input = {k:v[:,i].reshape((1,1,1)) for k,v in control_inputs.items()}
        u.append(np.squeeze(control_encoder.predict(temp_input)))
        
    # encode state and propogate
    x0 = np.squeeze(state_encoder.predict(x0))
    return x0, u

def encode_state(state_encoder, x0):
    return np.squeeze(state_encoder.predict(x0))

def decode_state(state_decoder,x):
    return state_decoder.predict(x[np.newaxis,:])


def decode_inputs(control_decoder, inputs):
    return control_decoder.predict(inputs)

def get_state_predictions(scenario,x_dec):
    state_predictions = {}
    for i, sig in enumerate(scenario['target_profile_names']):
        state_predictions[sig] = np.squeeze(x_dec[0][i*33:(i+1)*33])
        #     print("x_dec: {}".format(x_dec))
        #     print("state_pred: {}".format(state_predictions))
    return state_predictions


# should be correct now
def predict_model(i, umpc = None): 
    # Predicts on the i-th dataset of the data_generator_batch variable
    # set umpc to the mpc inputs, otherwise set to null to default to real inputs from the data generator
    
    inp = []

    for sig in scenario['target_profile_names']:
        inp.append((data_generator_batch[int(i/128)][0]['input_' + sig][int(i % 128)].reshape((1,1,scenario['profile_length']))))

    for sig in scenario['actuator_names']:
        inp.append((np.array(([np.transpose(data_generator_batch[int(i/128)][0]['input_past_' + sig][int(i % 128)])]))))

    if umpc is not None:
        umpc_transposed = np.transpose(umpc)
        for i in range(0, umpc_transposed.shape[0]):
            inp.append(np.array([umpc_transposed[i]]))
    else:
        for sig in scenario['actuator_names']:
            inp.append(np.array(([np.transpose(data_generator_batch[int(i/128)][0]['input_future_' + sig][int(i % 128)])])))

    for sig in scenario['scalar_input_names']:
        inp.append((data_generator_batch[int(i/128)][0]['input_' + sig][int(i % 128)].reshape((1,7))))
    return np.array(model.predict(inp))


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_discrete_are
    return solve_discrete_are(A, B, Q, R) 

def solve_Neo_MPC_system(Q,R,A,B,x0,xf,n, lim = 1e15):
    
    
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
        
    # Generate L
    L = np.zeros((N * n, N * n))
    rsl = slice(0, N)
    L[rsl, :N] = A #Make first line

    for i in range(1, n):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        L[rsl, :N] = L[rsl_p, :N]
        L[rsl, N : (i + 1) * N] = A @ L[rsl_p, : i * N]
    
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
    sol = solvers.qp(P,q, G, h);

    return sol

def get_model_prediction(x0, control_inputs, u_mpc, A, B, n, state_decoder):
    # Helper function for get_conv
    # returns real and mpc predictions for some initial state x0, given some A,B, control_inputs, mpc_inputs and number of timesteps n
    # u_mpc and control_inputs can be set to none to avoid predictions respectively
    
    
    state_pred_arr = []
    mpc_pred_arr = []
    x_mpc = x0
    # Propagate through model
    for i in range(0, n):
        if (control_inputs is not None):
            x0 = (x0 @ A.T  + control_inputs[i] @ B.T)
            state_pred_arr.append(get_state_predictions(scenario,decode_state(state_decoder,x0)))
        if (u_mpc is not None):
            x_mpc = (x_mpc @ A.T  + u_mpc[i] @ B.T)
            mpc_pred_arr.append(get_state_predictions(scenario,decode_state(state_decoder,x_mpc)))

    # print("State_predictions: {}".format(state_predictions))
    return state_pred_arr, mpc_pred_arr

### Set up system
def get_conv(index, X_0_enc_tot, u_enc_tot, X_F_enc_tot, scenario, state_decoder):
    # Contains only convolutional model predictions and autoencoder predictions with mpc inputs
    # Gets all predictions and true values for dataset at index
    # Plotting and filters are optional
    
    
    n = scenario['lookahead']
    profiles = scenario['target_profile_names']
    temp_x = X_0_enc_tot[index]
    temp_u = u_enc_tot[index]
    temp_xf = X_F_enc_tot[index]

    ### Build Conv Model Predictions
    conv_pred_profiles = {}
    for sig in scenario['target_profile_names']:
        conv_pred_profiles[sig] = predictions[sig][index]

    #### Get MPC Solution

    N = temp_x.shape[0]
    M = u_enc_tot.shape[2]

    # Generate Q,R
    Q = np.eye(N)*1e-5
    R = np.eye(M)*1e3

    sol_mpc = solve_Neo_MPC_system(Q,R,A,B,temp_x,temp_xf,n)
    sol_mpc = np.array(sol_mpc['x'])
    u_mpc = sol_mpc.reshape(n, M)
    temp_u_mpc = u_mpc
    state_pred, mpc_pred = get_model_prediction(temp_x, temp_u, temp_u_mpc, A, B, n, state_decoder)
    
    return state_pred, mpc_pred, conv_pred_profiles


def gen_conv_mpc(index, X_0_enc_tot, x_enc_tot, X_F_enc_tot, scenario, state_decoder, Q_scale, R_scale):
    # Contains convolutional model predictions with mpc inputs instead of just convolutional model predictions
    # Gets all predictions and true values for dataset at index
    # Plotting and filters are optional
    
    ### Set up system
    n = scenario['lookahead']
    profiles = scenario['target_profile_names']
    temp_x = X_0_enc_tot[index]
    temp_u = u_enc_tot[index]
    temp_xf = X_F_enc_tot[index]


    #### Get MPC Solution

    N = temp_x.shape[0]
    M = u_enc_tot.shape[2]

    # Generate Q,R    
    Q = np.eye(N)
    R = np.eye(M)
    Q = np.eye(N)*1e10
    R = np.eye(M)*1e-15

    n_prime = n

    sol_mpc = solve_Neo_MPC_system(Q,R,A,B,temp_x,temp_xf,n_prime, lim = 2)
    sol_mpc = np.array(sol_mpc['x'])
    u_mpc = sol_mpc.reshape(n, M)
    temp_u_mpc = u_mpc

    ### Build Conv Model Predictions
    conv_pred_profiles = {}
    for j, sig in enumerate(scenario['target_profile_names']):
        conv_pred_profiles[sig] = np.squeeze(predict_model(index, umpc = temp_u_mpc)[j])

    state_pred, mpc_pred = get_model_prediction(temp_x, temp_u, temp_u_mpc, A, B, n, state_decoder)
    
    return state_pred, mpc_pred, conv_pred_profiles


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    num_cores = 1
    config = tf.ConfigProto(intra_op_parallelism_threads=8,
                            inter_op_parallelism_threads=8, 
                            allow_soft_placement=True,
                            device_count = {'CPU' : 1,
                                            'GPU' : 0})
    
    session = tf.Session(config=config)
    K.set_session(session)

    
#     os.chdir(os.path.expanduser('/projects/EKOLEMEN/profile_predictor/run_results_03_10/'))
#     files = [foo for foo in os.listdir() if 'Scenario-265.h5' in foo]
    
    model_path = ('/scratch/gpfs/aaronwu/run_results_06_27_21/model-autoencoder_LA-6_27Jun21-12-24_Scenario-0.h5')
    scenario_path = ('/scratch/gpfs/aaronwu/run_results_06_27_21/model-autoencoder_LA-6_27Jun21-12-24_Scenario-0_params.pkl')
    
    #scenario_path = ('/home/aiqtidar/run_results_04_15_Aaron_Best/final_model-autoencoder_SET-dense_SDT-dense_CET-dense_CDT-dense_profiles-'
    #                 'dens-temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_LB-0_LA-4_15Apr21-13-36_Scenario-0_params.pkl')

    #model_path = ('/home/aiqtidar/run_results_04_15_Aaron_Best/final_model-autoencoder_SET-dense_SDT-dense_CET-dense_CDT-dense_profiles-dens-'
    #              'temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_LB-0_LA-4_15Apr21-13-36_Scenario-0.h5')

    # Load Model
    model = keras.models.load_model(model_path, compile=False)
    with open(scenario_path, 'rb') as f:
        scenario_auto = pickle.load(f, encoding='latin1')

    
    A,B = get_AB(model)
    state_encoder, state_decoder, control_encoder = get_submodels(model)

    # Ideal Convolutional Model
#     model = keras.models.load_model('/projects/EKOLEMEN/profile_predictor/run_results_03_10/model-conv2d_profiles-dens-temp-'
#                                     +'q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_15Mar20-03-48_Scenario-265.h5', compile=False)
#     with open('/projects/EKOLEMEN/profile_predictor/run_results_03_10/model-conv2d_profiles-dens-temp-q_EFIT01-'
#               +'rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_15Mar20-03-48_Scenario-265_params.pkl', 'rb') as f:
#         scenario = pickle.load(f, encoding='latin1')

    model = keras.models.load_model('/scratch/gpfs/aaronwu/run_results_07_07_21/run_results_07_07_21model-conv2d_profiles-temp-dens-rotation-'+
                                   'press_EFIT02-q_EFIT02_act-pinj-curr-tinj-gasA-a_EFIT02-drsep_EFIT02-kappa_EFIT02-rmagx_EFIT02'+
                                   '-triangularity_top_EFIT02-triangularity_bot_EFIT02_07Jul21-14-40.h5', compile = False)

    with open('/scratch/gpfs/aaronwu/run_results_07_07_21/run_results_07_07_21model-conv2d_profiles-temp-dens-rotation'+
             '-press_EFIT02-q_EFIT02_act-pinj-curr-tinj-gasA-a_EFIT02-drsep_EFIT02-kappa_EFIT02-'+
             'rmagx_EFIT02-triangularity_top_EFIT02-triangularity_bot_EFIT02_07Jul21-14-40_params.pkl', 'rb') as f:
           scenario = pickle.load(f, encoding='latin1')
        



    

    orig_data_path = '/scratch/gpfs/jabbate/full_data_with_error/train_data.pkl'
    test_data_path = '/scratch/gpfs/jabbate/full_data_with_error/test_data.pkl' 
    traindata, valdata, normalization_dict = helpers.data_generator.process_data(orig_data_path,
                                                                                 scenario['sig_names'],
                                                                                 scenario['normalization_method'],
                                                                                 scenario['window_length'],
                                                                                 scenario['window_overlap'],
                                                                                 scenario['lookbacks'],
                                                                                 scenario['lookahead'],
                                                                                 scenario['sample_step'],
                                                                                 scenario['uniform_normalization'],
                                                                                 1, #scenario['train_frac'],
                                                                                 0, #scenario['val_frac'],
                                                                                 scenario['nshots'],
                                                                                 2, #scenario['verbose']
                                                                                 scenario['flattop_only'],
                                                                                 randomize=False,
                                                                                 pruning_functions=scenario['pruning_functions'],
                                                                                 excluded_shots = scenario['excluded_shots'],
                                                                                 delta_sigs = [],
                                                                                 invert_q= scenario['invert_q'],
                                                                                 val_idx = 0,
                                                                                 uncertainties=True)
    traindata = helpers.normalization.renormalize(helpers.normalization.denormalize(traindata.copy(),normalization_dict),scenario['normalization_dict'])
    valdata = helpers.normalization.renormalize(helpers.normalization.denormalize(valdata.copy(),normalization_dict),scenario['normalization_dict'])

    data_generator_batch = DataGenerator(valdata,
                                         128, #scenario['batch_size'],
                                         scenario['input_profile_names'],
                                         scenario['actuator_names'],
                                         scenario['target_profile_names'],
                                         scenario['scalar_input_names'],
                                         scenario['lookbacks'],
                                         scenario['lookahead'],
                                         scenario['predict_deltas'],
                                         scenario['profile_downsample'],
                                         shuffle=False,
                                         sample_weights = None)
    print("starting predictions")
    predictions_arr = np.array(model.predict_generator(data_generator_batch, verbose=2))
    predictions = {sig: arr for sig, arr in zip(scenario['target_profile_names'],predictions_arr)}
    

    ###################### Get inital state and actuator inputs ready ######################

    X_F_conv_tot = [] # Final state predicted by conv model
    X_F_real_tot = [] # Final real state
    X_F_enc_tot = [] # Final real state encoded
    X_0_tot = [] # Initial state
    u_tot = [] # control inputs
    X_0_enc_tot = [] # Initial state encoded
    u_enc_tot = [] # Encoded control inputs
    state_inputs_arr = [] # Contains profiles, for both initial and final state
    
    for i in range(0, predictions_arr.shape[1]):
        print("Completed {} of {}".format(i,predictions_arr.shape[1]),end='\n')
        
        # Get initial profile array
        x0 = {}
        state_inputs = {}
        for sig in scenario['target_profile_names']:
            x0['input_' + sig] = data_generator_batch[int(i/128)][0]['input_' + sig][int(i % 128)].reshape((1,1,scenario['profile_length']))
            xf_real = data_generator_batch[int(i/128)][1]['target_' + sig][int(i % 128)].reshape((1,1,scenario['profile_length']))
            state_inputs['input_' + sig] = np.array([x0['input_' + sig], xf_real])
            
        for sig in scenario['scalar_input_names']:
            x0['input_' + sig] = data_generator_batch[int(i/128)][0]['input_' + sig][int(i % 128)][0].reshape((1,1,1))
    
            # Get control inputs array
        control_inputs = {}
        for sig in scenario['actuator_names']:
            control_inputs['input_'+sig] = np.array(([np.transpose(data_generator_batch[int(i/128)][0]['input_future_' + sig][int(i % 128)] [np.newaxis])]))
    
        # Get target state
        X_F = {}
        X_F_real = {}
        for profile in scenario['target_profile_names']:
            X_F['input_' + profile] = np.array([[predictions[profile][i]]])
            X_F_real['input_' + profile] = state_inputs['input_' + profile][-1]

        for sig in scenario['scalar_input_names']:
            X_F_real['input_' + sig] = data_generator_batch[int(i/128)][0]['input_' + sig][int(i % 128)][-1].reshape((1,1,1))
    
    
    
        x_enc, u_enc = encode_state_and_inputs(state_encoder,control_encoder,scenario,x0,control_inputs)
    
        X_F_real_tot.append(X_F_real)
        X_F_enc_tot.append(encode_state(state_encoder, X_F_real))
        X_F_conv_tot.append(X_F)
        X_0_tot.append(x0)
        u_tot.append(control_inputs)    
        X_0_enc_tot.append(x_enc)
        u_enc_tot.append(u_enc)
        state_inputs_arr.append(state_inputs)
    

        # Convert to Numpy Arrays
    X_F_real_tot = np.array(X_F_real_tot)
    X_F_enc_tot = np.array(X_F_enc_tot)
    X_F_conv_tot = np.array(X_F_conv_tot)
    X_0_tot = np.array(X_0_tot)
    u_tot = np.array(u_tot)
    X_0_enc_tot = np.array(X_0_enc_tot)
    u_enc_tot = np.array(u_enc_tot)
        

    for pair in itertools.product([1e5], [1e-15]):
        #autoencoder_mpc = {sig:[] for sig in scenario['target_profile_names']}
        conv_no_mpc_errors = {sig:[] for sig in scenario['target_profile_names']}
        conv_with_mpc_errors = {sig:[] for sig in scenario['target_profile_names']}

        for i in range(0, len(X_0_tot)): #len(X_0_tot)
            print("Completed {} of {}".format(i,len(X_0_tot)),end='\n')
            
            state_pred, mpc_pred, conv_pred_profiles_mpc = gen_conv_mpc(i, X_0_enc_tot, u_enc_tot, X_F_enc_tot, scenario, state_decoder, pair[0], pair[1])
            _, _, conv_pred_profiles = get_conv(i, X_0_enc_tot, u_enc_tot, X_F_enc_tot, scenario, state_decoder)
    
    
            for profile in scenario['target_profile_names']:
                # True profile
                true_norm = state_inputs_arr[i]['input_' + profile][0].squeeze() + state_inputs_arr[i]['input_' + profile][-1].squeeze()
                #true = helpers.normalization.denormalize_arr(true_norm,scenario['normalization_dict'][profile])
            

                # Convolutional Model Predictions
                # With MPC inputs
                conv_pred_mpc = state_inputs_arr[i]['input_' + profile][0].squeeze()  + (conv_pred_profiles_mpc[profile].squeeze())
                #conv_pred_mpc = helpers.normalization.denormalize_arr(conv_pred_mpc,scenario['normalization_dict'][profile])
            
                # Without MPC inputs
                conv_pred = state_inputs_arr[i]['input_' + profile][0].squeeze()  + (conv_pred_profiles[profile].squeeze())
                #conv_pred = helpers.normalization.denormalize_arr(conv_pred,scenario['normalization_dict'][profile])
                
                conv_pred_mpc_error = np.abs(true_norm - conv_pred_mpc)
                conv_pred_error = np.abs(true_norm - conv_pred)
                #auto_pred_err = np.abs(true_norm - mpc_pred[-1][profile])
        
            
                conv_with_mpc_errors[profile].append(conv_pred_mpc_error)
                conv_no_mpc_errors[profile].append(conv_pred_error)
                #autoencoder_mpc[profile].append(auto_pred_err)
                
        with open('/home/aaronwu/plasma-profile-predictor/Controls/datadump/Q5Rneg15_noControlEncode_noLQR.pkl', 'wb') as f:
            pickle.dump([conv_no_mpc_errors, conv_with_mpc_errors], f)

        
