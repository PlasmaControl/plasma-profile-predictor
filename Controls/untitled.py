import os
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


import random
import itertools
from helpers.hyperparam_helpers import make_bash_scripts
from helpers.callbacks import CyclicLR, TensorBoardWrapper, TimingCallback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping




###################
# set session
###################
num_cores = 8
req_mem = 48 # gb
ngpu = 1

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

if ngpu > 1:
    parallel_model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    
############### Helpers ########################
# Extract A, B from Autoencoder model

def get_AB(model):
    A = model.get_layer('AB_matrices').get_weights()[1].T
    B = model.get_layer('AB_matrices').get_weights()[0].T
    return A,B
def get_submodels(model):
    from keras.models import Model
    state_encoder = model.get_layer('state_encoder_time_dist').layer.layers[-1]
    control_encoder = model.get_layer('ctrl_encoder_time_dist').layer.layers[-1]
    state_decoder = model.get_layer('state_decoder_time_dist').layer
    return state_encoder, state_decoder, control_encoder

def get_control_decoder_matrices(control_encoder):
    A1 = control_encoder.layers[5].get_weights()[0]
    A2 = control_encoder.layers[6].get_weights()[0]
    B1 = control_encoder.layers[5].get_weights()[1]
    B2 = control_encoder.layers[6].get_weights()[1]
    
    A1 = np.linalg.inv(A1)
    A2 = np.linalg.inv(A2)
    
    return A1, A2, B1, B2

def decode_control(A1, A2, B1, B2, u_enc):
    u_dec = []
    for elem in u_enc:
        u = A2 @ (elem - B2)
        u = A1 @ (u - B1)      
        u_dec.append(u)
    return np.array(u_dec)
        

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

def get_final_state(state_encoder,scenario,inputs,**kwargs):
    state_inputs = {}
    xf = {}
    for sig in scenario['target_profile_names']+scenario['scalar_names']:
        state_inputs[sig] = np.squeeze(inputs[0]['input_'+sig])
        if sig in scenario['profile_names']:
            xf['input_'+sig] = inputs[0]['input_'+sig][0][3].reshape((1,1,scenario['profile_length']))
        else:
            xf['input_'+sig] = inputs[0]['input_'+sig][0][3].reshape((1,1,1))
    
    xf_enc = np.squeeze(state_encoder.predict(xf))
    return xf, xf_enc

def get_state_predictions(scenario,x_dec):
    state_predictions = {}
    for i, sig in enumerate(scenario['target_profile_names']):
        state_predictions[sig] = np.squeeze(x_dec[0][i*33:(i+1)*33])
    return state_predictions

def mean_squared_error(true,pred):
    return np.mean((true-pred)**2)

def mean_absolute_error(true,pred):
    return np.mean(np.abs(true-pred))

def median_absolute_error(true,pred):
    return np.median(np.abs(true-pred))

def percentile25_absolute_error(true,pred):
    return np.percentile(np.abs(true-pred),25)

def percentile75_absolute_error(true,pred):
    return np.percentile(np.abs(true-pred),75)

def median_squared_error(true,pred):
    return np.median((true-pred)**2)

def percentile25_squared_error(true,pred):
    return np.percentile((true-pred)**2,25)

def percentile75_squared_error(true,pred):
    return np.percentile((true-pred)**2,75)

def huber_error(true,pred):
    return np.mean(np.where(np.abs(pred-true) < 0.7, 0.5*(pred - true)**2, 0.7*(np.abs(pred-true)-0.5*0.7)))

def scalarize_mean(arr, **kwargs):
    return np.mean(arr)

def scalarize_std(arr, **kwargs):
    return np.std(arr)

def scalarize_pca_1(arr, **kwargs):
    fitter = kwargs.get('fitter')
    ret = fitter.transform(arr).squeeze()[0]
    return ret

def scalarize_pca_2(arr, **kwargs):
    fitter = kwargs.get('fitter')
    ret = fitter.transform(arr).squeeze()[1]
    return ret

def scalarize_pca_3(arr, **kwargs):
    fitter = kwargs.get('fitter')
    ret = fitter.transform(arr).squeeze()[2]
    return ret

def scalarize_pca_4(arr, **kwargs):
    fitter = kwargs.get('fitter')
    ret = fitter.transform(arr).squeeze()[3]
    return ret

def scalarize_pca_5(arr, **kwargs):
    fitter = kwargs.get('fitter')
    ret = fitter.transform(arr).squeeze()[4]
    return ret

def scalarize_pca_6(arr, **kwargs):
    fitter = kwargs.get('fitter')
    ret = fitter.transform(arr).squeeze()[5]
    return ret

def find_bounds(true,pred,percentile=90):
    arr = np.concatenate([true,pred]).flatten()
  
    true_bounds=(np.percentile(true, 50-percentile/2),np.percentile(true, 50+percentile/2))
    pred_bounds=(np.percentile(pred, 50-percentile/2),np.percentile(pred, 50+percentile/2))
    return (np.maximum(true_bounds[0],pred_bounds[0]),np.minimum(true_bounds[1],pred_bounds[1]))

def get_conv_mpc_train(i):
    # Build MPC inputs
    n = scenario['lookahead']
    temp_x_real = {}
    temp_xf_real = {}
    temp_u_mpc = 0

    for sig in scenario['target_profile_names']:
        temp_x_real['input_' + sig] = train_generator[i][0]['input_' + sig].reshape((1,1,scenario['profile_length']))
        temp_xf_real['input_' + sig] = temp_x_real['input_' + sig] + train_generator[i][1]['target_' + sig].reshape((1,1,scenario['profile_length']))

    for sig in scenario['scalar_input_names']:
        temp_x_real['input_' + sig] = train_generator[i][0]['input_' + sig][-1][0].reshape((1,1,1))
        temp_xf_real['input_' + sig] = train_generator[i][0]['input_' + sig][-1][-1].reshape((1,1,1))


    temp_x = encode_state(state_encoder, temp_x_real)
    temp_xf = encode_state(state_encoder, temp_xf_real)

    #### Get MPC Solution

    N = temp_x.shape[0]
    M = B.shape[1]

    # Generate Q,R
    Q_scale = 1e5
    R_scale = 1e-10

    Q = np.eye(N)
    R = np.eye(M)
    Q = np.eye(N)*Q_scale
    R = np.eye(M)*R_scale

    n_prime = n

    sol_mpc = solve_Neo_MPC_system(Q,R,A,B,temp_x,temp_xf,n_prime, lim = 2)
    sol_mpc = np.array(sol_mpc['x'])
    u_mpc = sol_mpc.reshape(n_prime, M)
    temp_u_mpc = u_mpc

    ### Build Conv Model Predictions
    conv_pred_profiles = {}
    temp_prof = predict_model_train(i, umpc = temp_u_mpc)
    for j, sig in enumerate(scenario['target_profile_names']):
        conv_pred_profiles[sig] = np.squeeze(temp_prof[j])
    return conv_pred_profiles


def predict_model_train(i, umpc = None):
    i = 0; umpc = None;

    inp = []

    for sig in scenario['target_profile_names']:
        inp.append((train_generator[i][0]['input_' + sig].reshape((1,1,scenario['profile_length']))))



    for sig in scenario['actuator_names']:
        inp.append(np.array([np.squeeze(train_generator[i][0]['input_past_' + sig])]))

    for sig in scenario['actuator_names']:
        if umpc is not None:
            inp.append(umpc)
        else:
            inp.append(np.array([np.squeeze(train_generator[i][0]['input_future_' + sig])]))

    for sig in scenario['scalar_input_names']:
        inp.append(np.array([np.squeeze(train_generator[i][0]['input_' + sig])]))

    return np.array(model.predict(inp))

def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_discrete_are
    return solve_discrete_are(A, B, Q, R) 

def solve_Neo_MPC_system(Q,R,A,B,x0,xf,n, lim = 2):
    
    # Imports
    import numpy
    from cvxopt import matrix
    from cvxopt import solvers
    
    # Define parameters
    N = A.shape[0]
    M = B.shape[1]

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
    
    # Generate Q_bar
    Q_bar = np.zeros((N * n, N * n))
    rsl = slice(0, N)
    Q_bar[rsl, :N] = Q

    for i in range(1, n-1):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        Q_bar[rsl, N : (i + 1) * N] = Q_bar[rsl_p, : i * N]
    
    Q_N = lqr(A,B,Q,R)
    rsl = slice((n-1) * N, n * N)
    Q_bar[rsl, rsl] = Q_N


    # Generate R_bar
    R_bar = np.kron(np.eye(n),R)


    # Generate V
    V = np.zeros((N * n, n * M))
    rsl = slice(0, N)
    V[rsl, :M] = B #Make first line

    for i in range(1, n):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        V[rsl, :M] = A @ V[rsl_p, :M] # A^(N-1)*B
        V[rsl, M : (i + 1) * M] = V[rsl_p, : i * M]
        
    # Generate L
    L = np.zeros((N * n, N * n))
    rsl = slice(0, N)
    L[rsl, :N] = A #Make first line

    for i in range(1, n):
        rsl_p, rsl = rsl, slice(i * N, (i + 1) * N)
        L[rsl, :N] = L[rsl_p, :N]
        L[rsl, N : (i + 1) * N] = A @ L[rsl_p, : i * N]
    
    #Generate X_F from x_f
    X_F = np.zeros((N*n,1))
    
    for i in range(0,n):
        rsl = slice(i * N, (i + 1)*N)
        X_F[rsl] = xf
    
    # Generate D_bar, d
    D_bar = np.zeros((2*M*n,M*n))
    rsl = slice(0,M*n)
    D_bar[rsl, rsl] = np.eye(M*n)
    D_bar[slice(M*n,2*M*n), rsl] = -np.eye(M*n)

    d = np.zeros((2*M*n,1))
    
    # Limits on U
    d[rsl,:] = np.ones((M*n,1))*lim
    d[slice(M*n,2*M*n),:] = np.ones((M*n,1))*lim


    # Generate F and H matrices   
    temp = np.transpose(V) @ (Q_bar)
    F = temp @ (M_bar.dot(x0) - X_F) # + L @ (X_F)
    
    temp = np.transpose(V) @ (Q_bar)
    H = temp @ (V) + R_bar

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

#####################################################################################################################


# Load Both Models
scenario_path = '/home/aiqtidar/run_results_04_15_Aaron_Best/final_model-autoencoder_SET-dense_SDT-dense_CET-dense_CDT-dense_profiles-dens-temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_LB-0_LA-4_15Apr21-13-36_Scenario-0_params.pkl'
model_path = '/home/aiqtidar/run_results_04_15_Aaron_Best/final_model-autoencoder_SET-dense_SDT-dense_CET-dense_CDT-dense_profiles-dens-temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_LB-0_LA-4_15Apr21-13-36_Scenario-0.h5'


# Load Model
model = keras.models.load_model(model_path, compile=False)
with open(scenario_path, 'rb') as f:
    scenario_auto = pickle.load(f, encoding='latin1')

    
A,B = get_AB(model)
state_encoder, state_decoder, control_encoder = get_submodels(model)
A1_inv, A2_inv, B1, B2 = get_control_decoder_matrices(control_encoder)

print("A: " + str(A.shape))
print("B: " + str(B.shape))

# Ideal Convolutional Model
model = keras.models.load_model('/projects/EKOLEMEN/profile_predictor/run_results_03_10/model-conv2d_profiles-dens-temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_15Mar20-03-48_Scenario-265.h5', compile=False)
with open('/projects/EKOLEMEN/profile_predictor/run_results_03_10/model-conv2d_profiles-dens-temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_15Mar20-03-48_Scenario-265_params.pkl', 'rb') as f:
    scenario = pickle.load(f, encoding='latin1')
    
################################################ LABELS AND MATPLOTLIB STUFF ########################################################################

font={'family': 'DejaVu Serif',
      'size': 18}
plt.rc('font', **font)
matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)

matplotlib_colors = [(0.1215, 0.4667, 0.7058), # blue
                     (1.0000, 0.4980, 0.0549), # orange
                     (0.1725, 0.6275, 0.1725), # green
                     (0.8392, 0.1529, 0.1568), # red
                     (0.5804, 0.4039, 0.7412), # violet
                     (0.4980, 0.4980, 0.4980), # grey
                     (0.0902, 0.7450, 0.8117)] # cyan

matlab_colors=[(0.0000, 0.4470, 0.7410), # blue
               (0.8500, 0.3250, 0.0980), # reddish orange
               (0.9290, 0.6940, 0.1250), # yellow
               (0.4940, 0.1840, 0.5560), # purple
               (0.4660, 0.6740, 0.1880), # light green
               (0.3010, 0.7450, 0.9330), # cyan
               (0.6350, 0.0780, 0.1840)] # dark red

colorblind_colors = [(0.0000, 0.4500, 0.7000), # blue
                     (0.8359, 0.3682, 0.0000), # vermillion
                     (0.0000, 0.6000, 0.5000), # bluish green
                     (0.9500, 0.9000, 0.2500), # yellow
                     (0.3500, 0.7000, 0.9000), # sky blue
                     (0.8000, 0.6000, 0.7000), # reddish purple
                     (0.9000, 0.6000, 0.0000)] # orange

dashes = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), # solid
          (3.7, 1.6, 0.0, 0.0, 0.0, 0.0), # dashed
          (1.0, 1.6, 0.0, 0.0, 0.0, 0.0), # dotted
          (6.4, 1.6, 1.0, 1.6, 0.0, 0.0), # dot dash
          (3.0, 1.6, 1.0, 1.6, 1.0, 1.6), # dot dot dash
          (6.0, 4.0, 0.0, 0.0, 0.0, 0.0), # long dash
          (1.0, 1.6, 3.0, 1.6, 3.0, 1.6)] # dash dash dot

from matplotlib import rcParams, cycler
matplotlib.rcdefaults()
rcParams['font.family'] = 'DejaVu Serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 12
rcParams['figure.facecolor'] = (1,1,1,1)
rcParams['figure.figsize'] = (16,8)
rcParams['figure.dpi'] = 141
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.labelsize'] =  'large'
rcParams['axes.titlesize'] = 'x-large'
rcParams['lines.linewidth'] = 2.5
rcParams['lines.solid_capstyle'] = 'round'
rcParams['lines.dash_capstyle'] = 'round'
rcParams['lines.dash_joinstyle'] = 'round'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'
# rcParams['text.usetex']=True
color_cycle = cycler(color=colorblind_colors)
dash_cycle = cycler(dashes=dashes)
rcParams['axes.prop_cycle'] =  color_cycle

labelsize=10
ticksize=8
# for i,c in enumerate(colorblind_colors):
#     plt.plot((i)*np.ones(5),c=c)

eq_sigs = {'temp':'etemp',
         'thomson_temp_EFITRT1':'etemp',
         'thomson_temp_EFITRT2':'etemp',
         'dens':'edens',
         'thomson_dens_EFITRT1':'edens',
         'thomson_dens_EFITRT2':'edens',
         'itemp':'itemp',
         'cerquick_temp_EFITRT1':'itemp',
         'cerquick_temp_EFITRT2':'itemp',
         'rotation':'rotation',
         'cerquick_rotation_EFITRT1':'rotation',
         'cerquick_rotation_EFITRT2':'rotation',
         'press_EFITRT1':'press',
         'press_EFITRT2':'press',
         'press_EFIT01':'press',
         'press_EFIT02':'press',
         'ffprime_EFITRT1':'ffprime',
         'ffprime_EFITRT2':'ffprime',
         'ffprime_EFIT01':'ffprime',
         'ffprime_EFIT02':'ffprime',
         'q':'q',
         'q_EFITRT1':'q',
         'q_EFITRT2':'q',
         'q_EFIT01':'q',
         'q_EFIT02':'q'}

labels = {'edens': '$n_e$ ($10^{19}/m^3$)',
          'etemp': '$T_e$ (keV)',
          'itemp': '$T_i$ (keV)',
          'rotation':'$\Omega$ (kHz)',
          'q':'$\iota$',
          'press':'$P$ (Pa)',
         'ffprime':"$FF'$"}

labels = {key:labels[val] for key, val in eq_sigs.items()}

scatter_titles = {'mean':'Mean',
                  'std':'Std Dev.',
                  'pca_1':'PCA Mode 1',
                  'pca_2':'PCA Mode 2',
                  'pca_3':'PCA Mode 3',
                  'pca_4':'PCA Mode 4',
                  'pca_5':'PCA Mode 5',
                  'pca_6':'PCA Mode 6',
                  'pca_2':'PCA Mode 2'}

####################################### DATA ##############################################


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
                                                      invert_q=scenario.setdefault('invert_q',False),
                                                      val_idx = 0,
                                                      uncertainties=True)
traindata = helpers.normalization.renormalize(helpers.normalization.denormalize(traindata.copy(),normalization_dict),scenario['normalization_dict'])

###################################### PCA ANALYSIS ###################################################################
profiles = scenario['target_profile_names']

train_generator = DataGenerator(traindata,
                                1,
                                profiles,
                                scenario['actuator_names'],
                                profiles,
                                scenario['scalar_input_names'],
                                scenario['lookbacks'],
                                scenario['lookahead'],
                                scenario['predict_deltas'],
                                scenario['profile_downsample'],
                                False,
                                sample_weights=None)

from sklearn import preprocessing
from sklearn import decomposition

length = int(len(train_generator)/20)
num_components=10
full_pca_fitters = {}
delta_pca_fitters = {}
std_dict ={}
std_deltas_dict = {}
iqr_dict = {}
iqr_deltas_dict = {}

for profile in profiles:
    full = np.array([(train_generator[i][0]['input_' + profile]+train_generator[i][1]['target_' + profile]) for i in range(length)]).squeeze()
    delta = np.array([train_generator[i][1]['target_' + profile] for i in range(length)]).squeeze()
    std_dict[profile] = np.std(full)
    std_deltas_dict[profile] = np.std(delta)
    iqr_dict[profile] = scipy.stats.iqr(full)
    iqr_deltas_dict[profile] = scipy.stats.iqr(delta)
    
    print(profile, ' made arrays')
    full_pca_fitters[profile] = decomposition.IncrementalPCA(n_components=num_components).fit(full)
    print(profile, ' done full')
    delta_pca_fitters[profile] = decomposition.IncrementalPCA(n_components=num_components).fit(delta)
    print(profile, ' done deltas')


    
conv_pred_profiles_arr = []
with open('/home/aiqtidar/plasma-profile-predictor/Controls/Saved_stats/batch_run/train_conv_pred.pkl', 'rb') as f:
    conv_pred_profiles_arr = pickle.load(f)


fitter = full_pca_fitters
scalarize_functions = [scalarize_mean, scalarize_pca_1,scalarize_pca_2]
scalarize_function_names = [fun.__name__[10:] for fun in scalarize_functions]
profiles = scenario['target_profile_names']

num_samples = length

all_true_delta = {sig:{metric:np.zeros(num_samples) for metric in scalarize_function_names} for sig in profiles}
all_predicted_delta = {sig:{metric:np.zeros(num_samples) for metric in scalarize_function_names} for sig in profiles}

n=1
nmax = len(profiles)*num_samples*len(scalarize_functions)
for j,profile in enumerate(profiles):
    for k in range(length):
        conv_pred_profiles = get_conv_mpc_train(k)
        target = np.squeeze(train_generator[k][0]['input_' + profile] + train_generator[k][1]['target_' + profile])[np.newaxis,:]
        pred = np.squeeze(conv_pred_profiles_arr[0][k][profile] + train_generator[k][0]['input_' + profile])[np.newaxis,:]
#         target = np.squeeze(train_generator[k][1]['target_' + profile])[np.newaxis,:]
#         pred = np.squeeze(conv_pred_profiles_arr[k][profile])[np.newaxis,:]
        
        for i,scalarize in enumerate(scalarize_functions):
            all_true_delta[profile][scalarize_function_names[i]][k] = scalarize(target, fitter=fitter[profile])
            all_predicted_delta[profile][scalarize_function_names[i]][k] = scalarize(pred, fitter=fitter[profile])  
            print('{}/{}'.format(n,nmax),end='\r')
            n += 1


# Saving the statistics to file:
with open('/home/aiqtidar/plasma-profile-predictor/Controls/Saved_stats/batch_run/pca_stats_20.pkl', 'wb') as f:
    pickle.dump([all_true_delta, all_predicted_delta], f)