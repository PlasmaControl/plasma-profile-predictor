import gspread
from oauth2client.service_account import ServiceAccountCredentials
from keras.models import Model
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle
import keras
import sys
import os
sys.path.append(os.path.abspath('../'))
from helpers.data_generator import process_data, AutoEncoderDataGenerator, DataGenerator
import copy
from helpers.normalization import normalize, denormalize, renormalize
import scipy
from tqdm import tqdm


def clean_dir(dir_path):
    all_files = os.listdir(os.path.abspath(dir_path))
    model_files = [f for f in all_files if f.endswith('.h5')]
    scenario_files = [f for f in all_files if f.endswith('params.pkl')]
    drivers = [f for f in all_files if f.endswith('.sh')]
    curdir = os.getcwd()
    os.chdir(os.path.abspath(dir_path))
    for f in drivers:
        os.remove(f)
    for f in model_files:
        scenario_path = f[:-3] + '_params.pkl'
        if not os.path.exists(scenario_path):
            print('No scenario found for {}'.format(f))
            os.remove(f)
    for f in scenario_files:
        model_path = f[:-11] + '.h5'
        if not os.path.exists(model_path):
            print('No model found for {}'.format(f))
            os.remove(f)
    os.chdir(os.path.abspath(curdir))


def process_results_folder(dir_path):
    clean_dir(dir_path)
    all_files = os.listdir(os.path.abspath(dir_path))
    model_files = [os.path.abspath(dir_path) +'/'+ f for f in all_files if f.endswith('.h5')]
    for model_path in tqdm(model_files):
        model = keras.models.load_model(model_path, compile=False)
        scenario_path = model_path[:-3] + '_params.pkl'
        try:
            with open(scenario_path, 'rb') as fo:
                scenario = pickle.load(fo, encoding='latin1')
        except:
            print('No scenario file found for model {}'.format(str(model_path)))
            os.remove(model_path)

        if 'autoencoder' in scenario['runname']:
            try:
                write_autoencoder_results(model, scenario)
            except KeyError as key:
                print('missing key {} for run {}'.format(key.args[0],str(model_path)))
                
        else:
            try:
                write_conv_results(model,scenario)
            except KeyError as key:
                print('missing key {} for run {}'.format(key.args[0],str(model_path)))

def write_autoencoder_results(model, scenario):
    """opens a google sheet and writes results, and generates images and html"""

    if 'image_path' not in scenario.keys():
        scenario['image_path'] = 'https://jabbate7.github.io/plasma-profile-predictor/results/' + scenario['runname']

    base_sheet_path = "https://docs.google.com/spreadsheets/d/1GbU2FaC_Kz3QafGi5ch97mHbziqGz9hkH5wFjiWpIRc/edit#gid=0"
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.expanduser('~/plasma-profile-predictor/drive-credentials.json'), scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(base_sheet_path).sheet1

    col = sheet.find('runname').col
    runs = sheet.col_values(col)
    if scenario['runname'] not in runs:
        write_scenario_to_sheets(scenario,sheet)
    rowid = sheet.find(scenario['runname']).row
    scenario['sheet_path'] = base_sheet_path + "&range={}:{}".format(rowid,rowid)

    curr_dir = os.getcwd()
    results_dir =os.path.expanduser('~/plasma-profile-predictor/results/'+scenario['runname'])  
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    os.chdir(results_dir)
    f = open('index.html','w+')
    f.write('<html><head></head><body>')
    
    html = scenario_to_html(scenario)
    f.write(html + '<p>\n')
    
    html = plot_autoencoder_training(model,scenario, filename='training.png')
    f.write(html + '<p>\n')
    
    datapath = '/scratch/gpfs/jabbate/mixed_data/final_data_batch_80.pkl'
    with open(datapath,'rb') as fo:
        rawdata = pickle.load(fo,encoding='latin1')
    data = {163303: rawdata[163303]}
    traindata, valdata, normalization_dict = process_data(data,
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
                                                          0,
                                                          scenario['flattop_only'],
                                                          randomize=False)
    traindata = denormalize(traindata, normalization_dict,verbose=0)
    traindata = renormalize(traindata, scenario['normalization_dict'],verbose=0)
    generator = AutoEncoderDataGenerator(traindata,
                                                   scenario['batch_size'],
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
                                                   scenario['shuffle_generators'])
    times = [2000, 2480, 3080, 4040, 4820, 5840]
    shots = [163303]*len(times)
    html = plot_autoencoder_profiles(model,scenario, generator,shots,times)
    f.write(html + '<p>\n')

    html = plot_autoencoder_control_encoding(model,scenario,generator,shots,times,filename='control_encoding.png')
    f.write(html + '<p>\n')
    
    html = plot_autoencoder_AB(model,scenario, filename='AB.png')
    f.write(html + '<p>\n')
    
    html = plot_autoencoder_spectrum(model,scenario, filename='spectrum.png')
    f.write(html + '<p>\n')

    f.write('</body></html>')
    f.close()
    os.chdir(curr_dir)
    return scenario


def write_conv_results(model,scenario):
    """opens a google sheet and writes results, and generates images and html"""

    if 'image_path' not in scenario.keys():
        scenario['image_path'] = 'https://jabbate7.github.io/plasma-profile-predictor/results/' + scenario['runname']


    base_sheet_path = "https://docs.google.com/spreadsheets/d/10ImJmFpVGYwE-3AsJxiqt0SyTDBCimcOh35au6qsh_k/edit#gid=0"
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.expanduser('~/plasma-profile-predictor/drive-credentials.json'), scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(base_sheet_path).sheet1

    col = sheet.find('runname').col
    runs = sheet.col_values(col)
    if scenario['runname'] not in runs:
        write_scenario_to_sheets(scenario,sheet)
    rowid = sheet.find(scenario['runname']).row
    scenario['sheet_path'] = base_sheet_path + "&range={}:{}".format(rowid,rowid)

    curr_dir = os.getcwd()
    results_dir =os.path.expanduser('~/plasma-profile-predictor/results/'+scenario['runname'])  
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    os.chdir(results_dir)
    f = open('index.html','w+')
    f.write('<html><head></head><body>')

    html = scenario_to_html(scenario)
    f.write(html + '<p>\n')
    
    html = plot_conv_training(model,scenario, filename='training.png')
    f.write(html + '<p>\n')

    datapath = '/scratch/gpfs/jabbate/mixed_data/final_data_batch_80.pkl'
    with open(datapath,'rb') as fo:
        rawdata = pickle.load(fo,encoding='latin1')
    data = {163303: rawdata[163303]}
    traindata, valdata, normalization_dict = process_data(data,
                                                          scenario['sig_names'],
                                                          scenario['normalization_method'],
                                                          scenario['window_length'],
                                                          scenario['window_overlap'],
                                                          scenario['lookbacks'],
                                                          scenario['lookahead'],
                                                          scenario['sample_step'],
                                                          scenario['uniform_normalization'],
                                                          1,
                                                          0,
                                                          scenario['nshots'],
                                                          0,
                                                          scenario['flattop_only'],
                                                          randomize=False)
    traindata = denormalize(traindata, normalization_dict, verbose=0)
    traindata = renormalize(traindata, scenario['normalization_dict'],verbose=0)
    generator = DataGenerator(traindata,
                              scenario['batch_size'],
                              scenario['input_profile_names'],
                              scenario['actuator_names'],
                              scenario['target_profile_names'],
                              scenario['scalar_input_names'],
                              scenario['lookbacks'],
                              scenario['lookahead'],
                              scenario['predict_deltas'],
                              scenario['profile_downsample'],
                              shuffle=False)
    times = [2000, 2480, 3080, 4040, 4820, 5840]
    shots = [163303]*len(times)
    html = plot_conv_profiles(model,scenario,generator,shots,times)
    f.write(html + '<p>\n')
    
    f.write('</body></html>')
    f.close()
    os.chdir(curr_dir)
    return scenario
    
def write_scenario_to_sheets(scenario,sheet):
    """writes a scenario to google sheets"""

    sheet_keys = sheet.row_values(1)
    row = [None]*len(sheet_keys)
    for i,key in enumerate(sheet_keys):
        if key in scenario.keys():
            row[i] = str(scenario[key])
        elif key in scenario.get('history',{}):
            row[i] = str(scenario['history'][key][-1])
    sheet.append_row(row)

def scenario_to_html(scenario):
    """converts scenario dict to html"""
    
    foo = {k:v for k,v in scenario.items() if k not in ['history','normalization_dict','history_params','mse_weight_vector']}
    def printitems(dictObj, indent=0):
        p=[]
        p.append('<ul>\n')
        for k,v in dictObj.items():
            if isinstance(v, dict):
                p.append('<li><b>'+ str(k)+ '</b>: ')
                p.append(printitems(v))
                p.append('</li>\n')
            elif k in ['image_path','sheet_path']:
                p.append("<a href=\"" + str(v) + "\">" + str(k) + "</a>\n")          
            else:
                p.append('<li><b>'+ str(k)+ '</b>: '+ str(v)+ '</li>\n')
        p.append('</ul>\n')
        return ''.join(p)
    return printitems(foo)

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

def plot_autoencoder_AB(model,scenario, filename=None, **kwargs):
    
    if filename:
        matplotlib.use('Agg')
        plt.ioff()
        
    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)
    
    A,B = get_AB(model)
    f, axes = plt.subplots(1, 2, figsize=(28, 14),
                           gridspec_kw={'width_ratios': [scenario['state_latent_dim'], 
                                                         scenario['control_latent_dim']]})
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
        f.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html
    return f
        
def plot_autoencoder_spectrum(model,scenario, filename=None, **kwargs):
    
    if filename:
        matplotlib.use('Agg')
        plt.ioff()
        
    font={'family': 'DejaVu Serif',
          'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)
    
    dt = scenario['dt']
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
        f.savefig(filename,bbox_inches='tight')
        f.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html
    return f

def plot_autoencoder_training(model,scenario,filename=None,**kwargs):
    
    if filename:
        matplotlib.use('Agg')
        plt.ioff()
        
    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)
    
    f, axes = plt.subplots(2, 2, figsize=(28, 28))
    axes[0,0].semilogy(scenario['history']['loss'],label='train')
    axes[0,0].semilogy(scenario['history']['val_loss'],label='val')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()

    axes[0,1].semilogy(scenario['history']['x_residual_mean_squared_error'],label='train')
    axes[0,1].semilogy(scenario['history']['val_x_residual_mean_squared_error'],label='val')
    axes[0,1].set_title('X residual MSE')
    axes[0,1].legend()

    axes[1,0].semilogy(scenario['history']['u_residual_mean_squared_error'],label='train')
    axes[1,0].semilogy(scenario['history']['val_u_residual_mean_squared_error'],label='val')
    axes[1,0].set_title('U residual MSE')
    axes[1,0].legend()

    axes[1,1].semilogy(scenario['history']['linear_system_residual_mean_squared_error'],label='train')
    axes[1,1].semilogy(scenario['history']['val_linear_system_residual_mean_squared_error'],label='val')
    axes[1,1].set_title('Linear Model MSE')
    axes[1,1].legend()
    
    if filename:
        f.savefig(filename,bbox_inches='tight')
        f.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html
    return f

def get_autoencoder_predictions(state_encoder,state_decoder,control_encoder,A,B,scenario,inputs,shot,timestep,**kwargs):

    state_inputs = {}
    x0 = {}
    for sig in scenario['profile_names']+scenario['scalar_names']:
        state_inputs[sig] = np.squeeze(inputs['input_'+sig])
        x0['input_'+sig] = inputs['input_'+sig][:,0,:].reshape((1,1,scenario['profile_length']))
    control_inputs = {}
    for sig in scenario['actuator_names']:
        control_inputs['input_'+sig] = inputs['input_'+sig]
    # encode control    
    T = scenario['lookback'] + scenario['lookahead'] +1
    u = []
    for i in range(T):
        temp_input = {k:v[:,i].reshape((1,1,1)) for k,v in control_inputs.items()}
        u.append(np.squeeze(control_encoder.predict(temp_input)))
    # encode state and propogate
    x0 = np.squeeze(state_encoder.predict(x0))
    x = [x0]
    for i in range(scenario['lookahead']):
        x.append(A.dot(x[i]+B.dot(u[i])))
    # decode state and organize
    x_decoded = []
    for elem in x:
        x_decoded.append(state_decoder.predict(elem[np.newaxis,:]))
    state_predictions = {}
    residuals = {}
    for i, sig in enumerate(scenario['profile_names']):
        state_predictions[sig] = np.squeeze(np.array([x_decoded[j][i] for j in range(len(x_decoded))]))
        residuals[sig] = state_inputs[sig] - state_predictions[sig]
    return state_inputs, state_predictions, residuals
    
def plot_autoencoder_residuals(residuals,scenario,shot,timestep,filename=None, **kwargs):
  
    if filename:
        matplotlib.use('Agg')
        plt.ioff()

    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)
    
    psi = np.linspace(0,1,scenario['profile_length'])
    nsteps = scenario['lookahead']+1

    figsize = (30,10)
    fig, ax = plt.subplots(1,nsteps, figsize=figsize,sharey=True)
    fig.suptitle('Shot ' + str(int(shot)) + '   Time ' + str(int(timestep)),y=.95)
    for j in range(nsteps):
        for i, sig in enumerate(scenario['profile_names']):
            ax[j].plot(psi,residuals[sig][j].reshape((scenario['profile_length'],)),label=sig)
            ax[j].hlines(0,0,1)
            ax[j].legend()
            ax[j].tick_params(reset=True)
            ax[j].title.set_text('State Residuals t+' + str(int(j*scenario['dt']*1000)))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)     

    if filename:
        fig.savefig(filename,bbox_inches='tight')
        fig.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html
    return fig 
    
def plot_autoencoder_predictions_timestep(state_inputs, state_predictions, scenario, shot, timestep, filename=None, **kwargs):

    if filename:
        matplotlib.use('Agg')
        plt.ioff()

    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)    

    baseline = {k:v[0].reshape((scenario['profile_length'],)) for k,v in state_inputs.items()}
    true = {k:v[-1].reshape((scenario['profile_length'],)) for k,v in state_inputs.items()}
    pred = {k:v[-1].reshape((scenario['profile_length'],)) for k,v in state_predictions.items()}

    psi = np.linspace(0,1,scenario['profile_length'])
    ncols = len(scenario['profile_names'])
    nrows = 2
    figsize = (30,12)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle('Shot ' + str(int(shot)) + '   Time ' + str(int(timestep)) + '   Prediction Window ' 
                 + str(int(scenario['lookahead']*scenario['dt']*1000)),y=.95)
    for i, sig in enumerate(scenario['profile_names']):
        
        ax[0,i].plot(psi,pred[sig]-baseline[sig],psi,true[sig]-baseline[sig])
        ax[0,i].title.set_text(sig + ' (deltas)')
        ax[0,i].hlines(0,0,1)
        ax[0,i].legend(['predicted delta','true delta'])
        
        ax[1,i].plot(psi,pred[sig],psi,true[sig],psi, baseline[sig])
        ax[1,i].title.set_text(sig + ' (full)')
        ax[1,i].legend(['predicted','true','baseline'])
        
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)    
    if filename:
        fig.savefig(filename,bbox_inches='tight')
        fig.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html
    return fig        

def plot_autoencoder_profiles(model,scenario,generator,shots,times):
    html = ''
    A,B = get_AB(model)
    state_encoder, state_decoder, control_encoder, control_decoder = get_submodels(model)
    inputs, targets, actual = generator.get_data_by_shot_time(shots,times)
    for i, (shot,time) in enumerate(zip(actual['shots'],actual['times'])):
        inp = {sig:arr[np.newaxis,i] for sig, arr in inputs.items()}
        state_inputs, state_predictions, residuals = get_autoencoder_predictions(
            state_encoder,state_decoder,control_encoder,A,B,scenario,inp,shot,time)
        filename = 'residuals_shot' + str(int(shot)) + 'time' + str(int(time)) + '.png'
        newhtml = plot_autoencoder_residuals(residuals,scenario,shot,time,filename)
        html += newhtml + '\n'
        
        filename = 'profiles_shot' + str(int(shot)) + 'time' + str(int(time)) + '.png'
        newhtml = plot_autoencoder_predictions_timestep(state_inputs, state_predictions, scenario,shot,time,filename)
        html += newhtml + '\n'
    
    return html


def plot_autoencoder_control_encoding(model,scenario,generator,shots,times,filename=None,**kwargs):

    if filename:
        matplotlib.use('Agg')
        plt.ioff()

    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)
    
    state_encoder, state_decoder, control_encoder, control_decoder = get_submodels(model)
    inputs, targets, actual = generator.get_data_by_shot_time(shots,times)
    control_inputs = {}
    for sig in scenario['actuator_names']:
        control_inputs['input_'+sig] = inputs['input_'+sig]
    figsize = (20,10)
    nrows = int(np.ceil(len(shots)/3))
    ncols = min(len(shots),3)
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize,squeeze=False,sharey=True)
    fig.suptitle('Control Residuals',y=.95)
    
    for j, (shot,time) in enumerate(zip(shots,times)):
        inp = {sig:arr[j] for sig, arr in control_inputs.items() }
        residuals = []
        T = scenario['lookback'] + scenario['lookahead'] +1
        for i in range(T):
            temp_input = {k:v[i].reshape((1,1,1)) for k,v in inp.items()}
            encoded_control = control_encoder.predict(temp_input)
            residuals.append(np.squeeze(control_decoder.predict(encoded_control)))
        residuals = {sig:np.squeeze(inp['input_'+sig]-np.array(residuals)[:,i]) for i, sig in enumerate(scenario['actuator_names'])}

        t = np.arange(time,time+(T)*scenario['dt']*1000,scenario['dt']*1000)
        for i, sig in enumerate(scenario['actuator_names']):
            ax[np.unravel_index(j,(nrows,ncols))].plot(t,residuals[sig], label=sig)
            ax[np.unravel_index(j,(nrows,ncols))].hlines(0,min(t),max(t))
            ax[np.unravel_index(j,(nrows,ncols))].tick_params(reset=True)
            ax[np.unravel_index(j,(nrows,ncols))].legend()
            ax[np.unravel_index(j,(nrows,ncols))].title.set_text('Shot ' + str(int(shot)) + '   Time ' + str(int(time)))
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)     

    if filename:
        fig.savefig(filename,bbox_inches='tight')
        fig.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html    
    return fig


def plot_conv_training(model,scenario,filename=None,**kwargs):
    
    if filename:
        matplotlib.use('Agg')
        plt.ioff()

    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)

    targets = scenario['target_profile_names']
    nout = len(targets)
    nrows = int(np.ceil((nout+1)/2))
    f, axes = plt.subplots(nrows, 2, figsize=(28, 14*nrows))
    axes[0,0].semilogy(scenario['history']['loss'],label='train')
    axes[0,0].semilogy(scenario['history']['val_loss'],label='val')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
    i=1
    for i,targ in enumerate(targets):
        idx = np.unravel_index(i+1,(nrows,2))
        if 'target_' + targ + '_mean_squared_error' in scenario['history'].keys():
            axes[idx].semilogy(scenario['history']['target_' + targ + '_mean_squared_error'],label='train')
            axes[idx].semilogy(scenario['history']['val_target_' + targ + '_mean_squared_error'],label='val')
            axes[idx].set_title(targ + ' MSE')
            axes[idx].legend()
        else:
            axes[idx].semilogy(scenario['history']['target_' + targ + '_loss'],label='train')
            axes[idx].semilogy(scenario['history']['val_target_' + targ + '_loss'],label='val')
            axes[idx].set_title(targ + ' loss')
            axes[idx].legend()
    
    if filename:
        f.savefig(filename,bbox_inches='tight',quality=25)
        fig.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html
    return f

def plot_conv_profiles(model,scenario,generator,shots,times):
    html = ''
    inputs, targets, actual = generator.get_data_by_shot_time(shots,times)
    for i, (shot,time) in enumerate(zip(actual['shots'],actual['times'])):
        inp = {sig:arr[np.newaxis,i] for sig, arr in inputs.items()}
        targ = {sig:arr[np.newaxis,i] for sig, arr in targets.items()}

        filename = 'shot' + str(int(shot)) + 'time' + str(int(time)) + '.png'
        newhtml = plot_conv_profiles_timestep(model,scenario,inp,targ,shot,time,filename)
        html += newhtml + '\n'
    return html
    
def plot_conv_profiles_timestep(model,scenario,inputs,targets,shot,timestep,filename=None, **kwargs):
    
    if filename:
        matplotlib.use('Agg')
        plt.ioff()

    font={'family': 'DejaVu Serif',
      'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)

    predictions = model.predict(inputs)
    perturbed_input_names = ['input_future_' + sig for sig in scenario['actuator_names']]
    perturbed_inputs=copy.copy(inputs)
    for sig in perturbed_input_names:
        perturbed_inputs[sig] +=0.5 
    predictions_perturbed = model.predict(perturbed_inputs)
    if scenario['predict_deltas']:
        baseline = {sig: inputs['input_'+sig].reshape((scenario['profile_length'],)) 
                    for sig in scenario['target_profile_names']}
        true = {sig: targets['target_'+sig].reshape((scenario['profile_length'],)) 
                + baseline[sig] for sig in scenario['target_profile_names']}
        pred = {sig: predictions[i].reshape((scenario['profile_length'],)) 
                + baseline[sig] for i,sig in enumerate(scenario['target_profile_names'])}
        pred_pert = {sig: predictions_perturbed[i].reshape((scenario['profile_length'],)) 
                     + baseline[sig] for i,sig in enumerate(scenario['target_profile_names'])}
    else:
        baseline = {sig: np.zeros(scenario['profile_length']) 
                    for sig in scenario['target_profile_names']}
        true = {sig: targets['target_'+sig].reshape((scenario['profile_length'],)) 
                for sig in scenario['target_profile_names']}
        pred = {sig: predictions[i].reshape((scenario['profile_length'],)) 
                for i,sig in enumerate(scenario['target_profile_names'])}
        pred_pert = {sig: predictions_perturbed[i].reshape((scenario['profile_length'],)) 
                     for i,sig in enumerate(scenario['target_profile_names'])}
    psi = np.linspace(0,1,scenario['profile_length'])
    nrows = len(scenario['target_profile_names'])
    ncols = 2
    figsize = (ncols*8,nrows*8)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle('Shot ' + str(int(shot)) + '   Time ' + str(int(timestep)),y=.93)
    for i, sig in enumerate(scenario['target_profile_names']):
        ax[np.unravel_index(2*i,(nrows,ncols))].plot(psi,pred[sig]-baseline[sig],
                                             psi,true[sig]-baseline[sig],psi,pred_pert[sig]-baseline[sig])
        ax[np.unravel_index(2*i,(nrows,ncols))].title.set_text(sig + ' (deltas)')
        ax[np.unravel_index(2*i,(nrows,ncols))].hlines(0,0,1)
        ax[np.unravel_index(2*i,(nrows,ncols))].legend(['predicted delta','true delta','pred with perturbed input'])
        
        ax[np.unravel_index(2*i+1,(nrows,ncols))].plot(psi,pred[sig],psi,true[sig],
                                               psi,pred_pert[sig],psi, baseline[sig])
        ax[np.unravel_index(2*i+1,(nrows,ncols))].title.set_text(sig + ' (full)')
        ax[np.unravel_index(2*i+1,(nrows,ncols))].legend(['predicted','true','pred with perturbed input','baseline'])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)    
    if filename:
        fig.savefig(filename,bbox_inches='tight')
        fig.clear()
        plt.close('all')
        html = """<img src=\"""" + filename + """\"><p>"""
        return html
    return fig    


if __name__ == '__main__':
    process_results_folder(sys.argv[1])
