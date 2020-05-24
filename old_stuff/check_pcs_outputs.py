import keras
import numpy as np
from matplotlib import pyplot as plt
import pickle

datapath = './etemp_data_936928.pkl'
modelpath = './etemp_predictor.h5'
time = 160000

with open(datapath, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

idx = np.nonzero(data['time'] > time)[0][0]
idx_past = np.nonzero(data['time'] > time-60000)[0][0]
idx_future = np.nonzero(data['time'] > time+60000)[0][0]

model = keras.models.load_model(modelpath, compile=False)

proposal = {'input_future_pinj': np.array([0, 0, 0]) + data['in']['input_past_pinj'][0, idx],
            'input_future_curr': np.array([0, 0, 0]) + data['in']['input_past_curr'][0, idx],
            'input_future_tinj': np.array([0, 0, 0]) + data['in']['input_past_tinj'][0, idx]}
testin = {sig: val[:, idx] for sig, val in data['in'].items()}
testin['input_thomson_temp_EFITRT1'] = np.expand_dims(
    testin['input_thomson_temp_EFITRT1'], 0)
testin['input_thomson_dens_EFITRT1'] = np.expand_dims(
    testin['input_thomson_dens_EFITRT1'], 0)
testin['input_press_EFITRT1'] = np.expand_dims(
    testin['input_press_EFITRT1'], 0)
testin['input_q_EFIT01'] = np.expand_dims(testin['input_q_EFIT01'], 0)
testin['input_ffprime_EFITRT1'] = np.expand_dims(
    testin['input_ffprime_EFITRT1'], 0)
pcsout = {sig: val[:, idx] for sig, val in data['out'].items()}


testin.update(proposal)

testin = {key: np.expand_dims(item, 0) for key, item in testin.items()}
testout = model.predict(testin)

print(data['time'][idx_past])
print(data['in']['input_past_pinj'][:, idx_past])
print(data['time'][idx])
print(data['in']['input_past_pinj'][:, idx])
print(data['time'][idx_future])
print(data['in']['input_past_pinj'][:, idx_future])

psi = np.linspace(0, 1, 33)
fig, axes = plt.subplots(2, 1)
axes[0].plot(psi, pcsout['target_thomson_temp_EFITRT1'],
             psi, np.squeeze(testout[0]))
axes[0].legend(['pcs', 'true'])
axes[1].plot(psi, pcsout['target_thomson_dens_EFITRT1'],
             psi, np.squeeze(testout[1]))
axes[1].legend(['pcs', 'true'])
plt.show()
