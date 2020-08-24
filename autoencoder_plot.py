import pickle
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import sys
import os
from helpers.data_generator import process_data, AutoEncoderDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from time import strftime, localtime
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from helpers.normalization import normalize, denormalize, renormalize
from keras.utils.vis_utils import model_to_dot
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from helpers.custom_init import downsample
from helpers.custom_reg import groupLasso
import seaborn as sns

def get_AB(model):
    A = model.get_layer('AB_matrices').get_weights()[1].T
    B = model.get_layer('AB_matrices').get_weights()[0].T
    return A,B

def get_submodels(model):
    from keras.models import Model
    state_encoder = model.get_layer('state_encoder_time_dist').layer.layers[-1]
    control_encoder = model.get_layer('ctrl_encoder_time_dist').layer.layers[-1]
    state_decoder = model.get_layer('state_decoder_time_dist').layer
    #state_decoder = Model(model.get_layer('state_decoder_time_dist').layer.layers[0].input,
    #                      model.get_layer('state_decoder_time_dist').layer.layers[-2].get_output_at(1),
    #                     name='state_decoder')    
    #control_decoder = Model(model.get_layer('ctrl_decoder_time_dist').layer.layers[0].input,
    #                        model.get_layer('ctrl_decoder_time_dist').layer.layers[-2].get_output_at(1),
    #                        name='control_decoder')
 
    return state_encoder, state_decoder, control_encoder #, control_decoder

def plot_autoencoder_AB(model,scenario, filename=None, **kwargs):
    
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
        html = """<img src=\"""" + filename + """\"><p>"""
        plt.close()
        return html
    return f


def plot_autoencoder_spectrum(model,scenario, filename=None, **kwargs):

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
        html = """<img src=\"""" + filename + """\"><p>"""
        plt.close()
        return html
    return f

def plot_autoencoder_training(model,scenario,filename=None,**kwargs):

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
        html = """<img src=\"""" + filename + """\"><p>"""
        plt.close()
        return html
    return f
        

def get_autoencoder_predictions(state_encoder,state_decoder,control_encoder,A,B,scenario,inputs,**kwargs):
    import numpy as np
    state_inputs = {}
    x0 = {}
    for sig in scenario['profile_names']+scenario['scalar_names']:
        state_inputs[sig] = np.squeeze(inputs['input_'+sig])
        if sig in scenario['profile_names']:
            x0['input_'+sig] = inputs['input_'+sig][:,0,:].reshape((1,1,scenario['profile_length']))
        else:
            x0['input_'+sig] = inputs['input_'+sig][:,0].reshape((1,1,1))
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
        x.append(A.dot(x[i])+B.dot(u[i]))
    # decode state and organize
    x_decoded = []
    for elem in x:
        x_decoded.append(state_decoder.predict(elem[np.newaxis,:]))
    state_predictions = {}
    residuals = {}
    for i, sig in enumerate(scenario['profile_names']):
        state_predictions[sig] = np.squeeze(np.dsplit((np.array([x_decoded[j] for j in range(len(x_decoded))])),5)[i])
        residuals[sig] = state_inputs[sig] - state_predictions[sig]

    return state_inputs, state_predictions, residuals

def plot_autoencoder_residuals(model,scenario,generator,shots,times, filename = None,**kwargs):
    A, B = get_AB(model)
    state_encoder, state_decoder, control_encoder = get_submodels(model)
    inputs, targets, actual = generator.get_data_by_shot_time(shots,times)
    psi = np.linspace(0,1,scenario['profile_length'])
    nsteps = scenario['lookahead']
    
    fig = plt.figure(figsize=(40, 60))
    outer_grid = fig.add_gridspec(len(times),1)
    
    for k, (shot,time) in enumerate(zip(actual['shots'],actual['times'])):
        inp = {sig:arr[np.newaxis,k] for sig, arr in inputs.items()}
        state_inputs, state_predictions, residuals = get_autoencoder_predictions(
            state_encoder,state_decoder,control_encoder,A,B,scenario,inp)
        
        outerax = fig.add_subplot(outer_grid[k])
        outerax.set_title(label='Shot ' + str(int(shot)) + '   Time ' + str(int(time)),pad = 30)
        outerax.axis('off')
        
        inner_grid = outer_grid[k].subgridspec(1, nsteps)
        for j in range(nsteps):
            ax = fig.add_subplot(inner_grid[j])
            for i, sig in enumerate(scenario['profile_names']):
                ax.plot(psi,residuals[sig][j].reshape((scenario['profile_length'],)),label=sig)
                ax.hlines(0,0,1)
                ax.legend()
                ax.tick_params(reset=True)
                ax.set_title(label = 'State Residuals t+' + str(int(j*scenario['dt']*1000)))
            fig.add_subplot(ax)
      
    if filename:
        fig.savefig(filename, bbox='tight')
        html = """<img src=\"""" + filename + """\"><p>"""
        plt.close()
        return  html
    return fig 
    
def plot_autoencoder_predictions_timestep(model,scenario,generator,shots,times, filename = None,**kwargs):
    A,B = get_AB(model)
    state_encoder, state_decoder, control_encoder = get_submodels(model)
    inputs, targets, actual = generator.get_data_by_shot_time(shots,times)
    
    fig = plt.figure(figsize=(40,90))
    outer_grid = fig.add_gridspec(len(times),1)
    
    for i, (shot,time) in enumerate(zip(actual['shots'],actual['times'])):
        inp = {sig:arr[np.newaxis,i] for sig, arr in inputs.items()}
        state_inputs, state_predictions, residuals = get_autoencoder_predictions(
            state_encoder,state_decoder,control_encoder,A,B,scenario,inp)
        baseline = {k:v[0].reshape((scenario['profile_length'],)) for k,v in state_inputs.items() if k in scenario['profile_names']}
        true = {k:v[-1].reshape((scenario['profile_length'],)) for k,v in state_inputs.items() if k in scenario['profile_names']}
        pred = {k:v[-1].reshape((scenario['profile_length'],)) for k,v in state_predictions.items() if k in scenario['profile_names']}
        
        outerax = fig.add_subplot(outer_grid[i])
        outerax.set_title(label='Shot ' + str(int(shot)) + '   Time ' + str(int(time)) + '   Prediction Window ' 
                     + str(int(scenario['lookahead']*scenario['dt']*1000)),pad = 70)
        outerax.axis('off')
        
        ncols = len(scenario['profile_names'])
        nrows = 2
        psi = np.linspace(0,1,scenario['profile_length'])
        inner_grid = outer_grid[i].subgridspec(nrows, ncols)
        
        for j, sig in enumerate(scenario['profile_names']):
            ax = fig.add_subplot(inner_grid[j]) 
            ax.plot(psi,pred[sig]-baseline[sig],psi,true[sig]-baseline[sig])
            ax.set_title(label = sig + ' (deltas)')
            ax.hlines(0,0,1)
            ax.legend(['predicted delta','true delta'])
            
            ax1 = fig.add_subplot(inner_grid[j+ncols]) 
            ax1.plot(psi,pred[sig],psi,true[sig],psi, baseline[sig])
            ax1.set_title(label = sig + ' (full)')
            ax1.legend(['predicted','true','baseline'])
            
            fig.add_subplot(ax)
            fig.add_subplot(ax1)
            
    if filename:
        fig.savefig(filename, bbox='tight')
        html = """<img src=\"""" + filename + """\"><p>"""
        plt.close()
        return  html
    
    return fig        

def plot_autoencoder_control_encoding(model,scenario,generator,shots,times,filename=None,**kwargs):
    state_encoder, state_decoder, control_encoder = get_submodels(model)
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
        html = """<img src=\"""" + filename + """\"><p>"""
        plt.close()
        return  html    
    return fig


# state_inputs, state_predictions, residuals = get_autoencoder_predictions(state_encoder,state_decoder,control_encoder,A,B,scenario,inputs,175702,1400)
# f = plot_autoencoder_residuals(residuals,scenario,175702,1400)
# f = plot_autoencoder_predictions_timestep(state_inputs, state_predictions, scenario,175702,1400)


def write_results(model, scenario, worksheet):
    
    if 'image_path' not in scenario.keys():
        scenario['image_path'] = 'https://jabbate7.github.io/plasma-profile-predictor/results/' + scenario['runname']
    
    base_sheet_path = "https://docs.google.com/spreadsheets/d/1h2jm3PWuck-7t_WcHi3Zm0OT35fRfLr4RrrUjHrm1dA/edit#gid=0"
    scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.expanduser('~/plasma-profile-predictor/drive-credentials.json'), scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(base_sheet_path).get_worksheet(worksheet)
    
    write_scenario_to_sheets(scenario,sheet)
    rowid = sheet.find(scenario['runname']).row
    scenario['sheet_path'] = base_sheet_path + "&range={}:{}".format(rowid,rowid)
    
    results_dir = os.path.expanduser('~/results/'+scenario['runname'])  
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    os.chdir(results_dir)
    f = open('index.html','w+')
    f.write('<html><head></head><body>')
    html = scenario_to_html(scenario)
    f.write(html + '<p>\n')
    html = plot_autoencoder_training(model,scenario, filename='training.png')
    f.write(html + '<p>\n')
    
    html = plot_autoencoder_AB(model,scenario, filename='AB.png')
    f.write(html + '<p>\n')
    html = plot_autoencoder_spectrum(model,scenario, filename='spectrum.png')
    f.write(html + '<p>\n')
    
    html = plot_autoencoder_residuals(model,scenario,generator,shots,times, filename = 'state_residuals.png')
    f.write(html + '<p>\n')
    
    html = plot_autoencoder_predictions_timestep(model,scenario,generator,shots,times, filename = 'predictions.png')
    f.write(html + '<p>\n')
    '''
    _, html =  plot_autoencoder_control_encoding(model,scenario,generator,shots,times,filename='control_residuals.png')
    f.write(html + '<p>\n')
    '''
    f.write('</body></html>')
    f.close()
    
    
def write_scenario_to_sheets(scenario,sheet):
    sheet_keys = sheet.row_values(1)
    row = [None]*len(sheet_keys)
    for i,key in enumerate(sheet_keys):
        if key in scenario.keys():
            row[i] = str(scenario[key])
        elif key in scenario.get('history',{}):
            row[i] = str(scenario['history'][key][-1])
    sheet.append_row(row)

def scenario_to_html(scenario):
    foo = {k:v for k,v in scenario.items() if k not in ['history','normalization_dict','history_params']}
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


if __name__=='__main__':
    
    num_cores = int(sys.argv[1])
    compute_node_flag = int(sys.argv[2])
    run_path=sys.argv[3]
    sheet_idx = int(sys.argv[4])

    # Environment and plotting setup. 
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    num_cores = 1
    config = tf.ConfigProto(intra_op_parallelism_threads=4*num_cores,
                            inter_op_parallelism_threads=4*num_cores, 
                            allow_soft_placement=True,
                            device_count = {'CPU' : 1,
                                            'GPU' : 0})
                        
    session = tf.Session(config=config)
    K.set_session(session)
    font={'family': 'DejaVu Serif',
          'size': 18}
    plt.rc('font', **font)
    matplotlib.rcParams['figure.facecolor'] = (1,1,1,1)
    sys.path.append(os.path.abspath('../'))
    
    scenarios = []
    models = []
    for file in os.listdir(run_path):
        if file.endswith(".h5"):
            model_path = run_path+'/'+file
            if compute_node_flag == 0: 
                model = keras.models.load_model(model_path, compile=False)
                models.append(model)
                print('loaded model: ' + model_path.split('/')[-1])
            params_path = model_path[:-3]+'_params.pkl'
            with open(params_path, 'rb') as f:
                scenario = pickle.load(f, encoding='latin1')
                scenario['dt'] = 0.05
                scenarios.append(scenario)
            print('loaded dict: ' + params_path.split('/')[-1])
    
    datapath = '/scratch/gpfs/jabbate/full_data_with_error/train_data.pkl'
    with open(datapath,'rb') as f:
        rawdata = pickle.load(f,encoding='latin1')
        data = {163303: rawdata[163303]}
        times = [2000, 2480, 3080, 4040, 4820, 5840]
        shots = [163303]*len(times)
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
                                                              1,
                                                              scenario['flattop_only'],
                                                              invert_q = scenario['invert_q'],
                                                              randomize=False)
        traindata = denormalize(traindata, normalization_dict)
        traindata = renormalize(traindata, scenario['normalization_dict'])
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

        for model, scenario in zip(models, scenarios):
            write_results(model, scenario, sheet_idx)

        for model,scenario in zip(models, scenarios):
            dot = model_to_dot(model,show_shapes=True,show_layer_names=True,rankdir='TB')
            dot.write_png('/home/aaronwu/results/'+scenario['runname']+'/architecture.png')
