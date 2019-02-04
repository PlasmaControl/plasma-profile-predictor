from keras import models
from keras import layers

# Still need to add the proper input_shape to the train.py
#input_shape: input_shape=(None, len(train_data[0][0]))
#output_shape: output_shape=len(train_data[0][0])-num_sigs
def build_model(num_sigs_0d, num_sigs_1d, rho_length, lookback):
    
    model = models.Sequential()
    model.add(layers.LSTM(rho_length*num_sigs_1d, 
                          input_shape=(lookback,rho_length*num_sigs_1d+num_sigs_0d)))
    model.add(layers.Dense(rho_length*num_sigs_1d))

    return model