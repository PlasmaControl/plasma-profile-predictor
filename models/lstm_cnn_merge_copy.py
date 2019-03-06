from keras import models
from keras import layers

def build_model(num_sigs_0d, num_sigs_1d, rho_length, lookback, dense_1_size, lstm_1_size, kernel_size, max_pool_size=2):
    my_input = layers.Input(shape=(num_sigs_0d+num_sigs_1d*rho_length,))
    input_0d = layers.Lambda(lambda x: x[:,:num_sigs_0d],output_shape=(num_sigs_0d,))(my_input)
    input_1d = layers.Lambda(lambda x: x[:,num_sigs_0d:],output_shape=(num_sigs_1d*rho_length,))(my_input)
    input_1d = layers.Reshape((num_sigs_1d*rho_length,1))(input_1d)
    #input_1d = layers.Permute((2,1))(input_1d)
    input_1d=layers.Conv1D(filters=2, kernel_size=kernel_size, activation='relu', padding='same')(input_1d)
    input_1d=layers.MaxPooling1D(pool_size=2)(input_1d)
    input_1d=layers.Flatten()(input_1d)
    input_1d=layers.Dense(dense_1_size, activation='relu')(input_1d)
    final_input=layers.Concatenate()([input_0d,input_1d])
    pre_model=models.Model(inputs=my_input, outputs=final_input)
    
    full_input = layers.Input(shape=(lookback,num_sigs_1d*rho_length+num_sigs_0d,))
    output = layers.TimeDistributed(pre_model)(full_input)
    output = layers.LSTM(lstm_1_size)(output)
    output = layers.Dense(num_sigs_1d*rho_length, activation='relu')(output)
    model = models.Model(inputs=full_input, outputs=output)
    
    return model
