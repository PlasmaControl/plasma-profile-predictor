from keras import models
from keras import layers
from keras.utils import plot_model

def build_model(sigs_1d, sigs_0d, sigs_predict, rho_length_in, rho_length_out, lookback, delay,
                rnn_type, rnn_size, rnn_activation, 
                dense_0d_size, dense_0d_activation):

    if (rnn_type=='LSTM'):
        rnn_layer = layers.LSTM
    elif (rnn_type=='GRU'):
        rnn_layer=layers.GRU
    else:
        raise ValueError('rnn_type in conf must be GRU or LSTM')    

    input_0d=layers.Input(shape=(delay+1,len(sigs_0d),))
    input_1d=layers.Input(shape=(lookback+1,rho_length_in*len(sigs_1d),))
    
    permuted_0d=layers.Permute((2,1,))(input_0d)
    compressed_0d=layers.Dense(dense_0d_size)(permuted_0d)
    final_input_0d = layers.Flatten()(compressed_0d)

    trend_1d=rnn_layer(rnn_size, activation=rnn_activation)(input_1d)

    concat=layers.Concatenate()([final_input_0d, trend_1d])

    output=layers.Dense(rho_length_out)(concat)

    model = models.Model(inputs=[input_0d,input_1d], outputs=output)

    print(model.summary())
    return model

