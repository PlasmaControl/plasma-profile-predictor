from keras import models
from keras import layers

# Still need to add the proper input_shape to the train.py
#input_shape: input_shape=(None, len(train_data[0][0]))
#output_shape: output_shape=len(train_data[0][0])-num_sigs
def build_model(num_sigs_0d, num_sigs_1d, rho_length, lookback, 
                include_dense=False, dense_1_size, lstm_1_size, 
                rnn_type='LSTM', include_dropout=False, dropout=.2):
    
    model = models.Sequential()
    if (rnn_type=='LSTM'):
        rnn_layer = layers.LSTM
    elif (rnn_type=='GRU'):
        rnn_layer=layers.GRU
    else:
        raise ValueError('rnn_type in conf must be GRU or LSTM')
    
    if (include_dropout):
        model.add(layers.Dropout(dropout, 
                                 input_shape=(lookback,rho_length*num_sigs_1d+num_sigs_0d)))
    if (include_dense):
        model.add(layers.Dense(dense_1_size, 
                           input_shape=(lookback,rho_length*num_sigs_1d+num_sigs_0d)))
        model.add(rnn_layer(lstm_1_size))
    else:
        model.add(rnn_layer(lstm_1_size, input_shape=((lookback,rho_length*num_sigs_1d+num_sigs_0d)))
    
    model.add(layers.Dense(rho_length*num_sigs_1d))

    print(model.summary())

    return model
