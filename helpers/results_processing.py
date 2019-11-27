import gspread
from oauth2client.service_account import ServiceAccountCredentials
from keras.models import Model
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import numpy as np



def write_results_autoencoder(model,analysis_params):
    """opens a google sheet and writes results"""
    
    scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.expanduser('~/plasma-profile-predictor/drive-credentials.json'), scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1GbU2FaC_Kz3QafGi5ch97mHbziqGz9hkH5wFjiWpIRc/edit#gid=0").sheet1
    write_scenario(analysis_params,sheet)
    
def write_scenario(scenario,sheet):
    """writes a scenario to google sheets"""
    
    sheet_keys = sheet.row_values(1)
    row = [None]*len(sheet_keys)
    for i,key in enumerate(sheet_keys):
        if key in scenario.keys():
            row[i] = str(scenario[key])
        elif key in scenario.get('history',{}):
            row[i] = str(scenario['history'][key][-1])
    sheet.append_row(row)
    
def get_AB(model):
    """gets A,B matrices from autoencoder model"""
    
    A = model.get_layer('AB_matrices').get_weights()[1].T
    B = model.get_layer('AB_matrices').get_weights()[0].T
    return A,B

def get_submodels(model):
    """gets encoder/decoder submodels for state and control variables from trained model"""
    
    state_encoder = model.get_layer('state_encoder_time_dist').layer.layers[-1]
    control_encoder = model.get_layer('ctrl_encoder_time_dist').layer.layers[-1]
    state_decoder = Model(model.get_layer('state_decoder_time_dist').layer.layers[0].input,
                          model.get_layer('state_decoder_time_dist').layer.layers[-2].get_output_at(1),
                         name='state_decoder')    
    control_decoder = Model(model.get_layer('ctrl_decoder_time_dist').layer.layers[0].input,
                            model.get_layer('ctrl_decoder_time_dist').layer.layers[-2].get_output_at(1),
                           name='control_decoder')
 
    return state_encoder, state_decoder, control_encoder, control_decoder

def plot_AB(model,analysis_params, filename=None, **kwargs):
    """plots heatmaps of A,B matrices"""
    
    
    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)

    A,B = get_AB(model)
    f, axes = plt.subplots(1, 2, figsize=(28, 14),
                           gridspec_kw={'width_ratios': [analysis_params['state_latent_dim'], 
                                                         analysis_params['control_latent_dim']]})
    sns.heatmap(A, 
                cmap=kwargs.get('cmap','Spectral'),
                annot=kwargs.get('annot',False), 
                square=kwargs.get('square',True), 
                robust=kwargs.get('robust',False), 
                ax=axes[0]).set_title('A')
    sns.heatmap(B,
                cmap=kwargs.get('cmap','Spectral'), 
                annot=kwargs.get('annot',False), 
                square=kwargs.get('square',True), 
                robust=kwargs.get('robust',False), 
                ax=axes[1]).set_title('B')
    if filename:
        f.savefig(filename,bbox_inches='tight')
        
def plot_spectrum(model,analysis_params, filename=None, **kwargs):
    """plots spectrum of eigenvales of A, both regular and log"""
    
    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)
    
    dt = kwargs.get('dt',50/1000)
    A,B = get_AB(model)
    eigvals, eigvecs = np.linalg.eig(A)
    logeigvals = np.log(eigvals)
    for i, elem in enumerate(logeigvals):
        if abs(np.imag(elem)-np.pi)<np.finfo(np.float32).resolution:
            logeigvals[i] = np.real(elem) + 0j
    logeigvals = logeigvals/dt

    f, axes = plt.subplots(1, 2, figsize=(28, 14))
    axes[0].scatter(np.real(eigvals),np.imag(eigvals))
    t = np.linspace(0,2*np.pi,1000)
    axes[0].plot(np.cos(t),np.sin(t))

    axes[0].set_title('Eigenvalues of A')
    axes[0].grid(color='gray')
    axes[0].set_xlabel('Re($\lambda$)')
    axes[0].set_ylabel('Im($\lambda$)')


    axes[1].scatter(np.real(logeigvals),np.imag(logeigvals))
    axes[1].set_title('Eigenvalues of A')
    axes[1].grid(color='gray')
    axes[1].set_xlabel('Growth Rate (1/s)')
    axes[1].set_ylabel('$\omega$ (rad/s)')
    axes[1].set_xlim((1.1*np.min(np.real(logeigvals)),np.maximum(1.1*np.max(np.real(logeigvals)),0)))
    
    if filename:
        f.savefig(filename, bbox_inches='tight')
    return f

def plot_autoencoder_training(model,analysis_params,filename=None,**kwargs):
    """plots training/validation history (loss, residuals from outputs)"""
    
    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)

    f, axes = plt.subplots(2, 2, figsize=(28, 28))
    axes[0,0].semilogy(analysis_params['history']['loss'],label='train')
    axes[0,0].semilogy(analysis_params['history']['val_loss'],label='val')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()

    axes[0,1].semilogy(analysis_params['history']['x_residual_mean_squared_error'],label='train')
    axes[0,1].semilogy(analysis_params['history']['val_x_residual_mean_squared_error'],label='val')
    axes[0,1].set_title('X residual MSE')
    axes[0,1].legend()

    axes[1,0].semilogy(analysis_params['history']['u_residual_mean_squared_error'],label='train')
    axes[1,0].semilogy(analysis_params['history']['val_u_residual_mean_squared_error'],label='val')
    axes[1,0].set_title('U residual MSE')
    axes[1,0].legend()

    axes[1,1].semilogy(analysis_params['history']['linear_system_residual_mean_squared_error'],label='train')
    axes[1,1].semilogy(analysis_params['history']['val_linear_system_residual_mean_squared_error'],label='val')
    axes[1,1].set_title('Linear Model MSE')
    axes[1,1].legend()
    
    if filename:
        f.savefig(filename,bbox_inches='tight')
    return f