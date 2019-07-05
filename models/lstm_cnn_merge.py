from keras import models
from keras import layers
from keras.utils import plot_model

def build_model(num_sigs_0d, num_sigs_1d, num_sigs_predict, rho_length_in, rho_length_out, lookback, delay,
                include_cnn, cnn_activation, cnn_padding,
                kernel_size, max_pool_size, num_filters,
                dense_cnn_size, dense_cnn_activation, num_dense_cnn_layers,
                dense_pre_size, dense_pre_activation, num_pre_layers,
                rnn_type, rnn_size, rnn_activation,
                dense_final_size, dense_final_activation, num_final_layers, recurrent_dropout=0, rnn_dropout=0, dense_dropout=0):

    if (rnn_type=='LSTM'):
        rnn_layer = layers.LSTM
    elif (rnn_type=='GRU'):
        rnn_layer=layers.GRU
    else:
        raise ValueError('rnn_type in conf must be GRU or LSTM')
    my_input = layers.Input(shape=(lookback+1+delay,num_sigs_0d+num_sigs_1d*rho_length_in,))

    # CNN
    if include_cnn:
        input_0d = layers.Lambda(lambda x: x[:,:,:num_sigs_0d],output_shape=(lookback+1+delay,num_sigs_0d,))(my_input)
        input_1d = layers.Lambda(lambda x: x[:,:,num_sigs_0d:],output_shape=(lookback+1+delay,num_sigs_1d*rho_length_in,))(my_input)

        input_1d = layers.Reshape((lookback+1+delay,rho_length_in,1))(input_1d)

        input_1d = layers.TimeDistributed(layers.Conv1D(filters=num_filters, kernel_size=kernel_size, 
                                                        activation=cnn_activation, padding=cnn_padding))(input_1d)
        input_1d = layers.TimeDistributed(layers.MaxPooling1D(max_pool_size))(input_1d)

        converted_rho_length=int(input_1d.shape[2])

        input_1d = layers.Reshape((lookback+1+delay,converted_rho_length*num_filters))(input_1d)

        for i in range(num_dense_cnn_layers):
            input_1d = layers.TimeDistributed(layers.Dense(dense_cnn_size, activation=dense_cnn_activation))(input_1d)

        final_input = layers.Concatenate()([input_0d,input_1d])
    else: 
        final_input = my_input

    # Pre-RNN dense layer
    for i in range(num_pre_layers):
        final_input = layers.Dense(dense_pre_size, activation=dense_pre_activation)(final_input)
        final_input = layers.Dropout(dense_dropout)(final_input)

    # RNN layer
    output = rnn_layer(rnn_size, activation=rnn_activation, recurrent_dropout = recurrent_dropout, dropout = rnn_dropout)(final_input)

    # Post-RNN layer
    for i in range(num_final_layers):
        output = layers.Dense(dense_final_size, activation=dense_final_activation)(output)
        output = layers.Dropout(dense_dropout)(output)

    output = layers.Dense(num_sigs_predict*rho_length_out)(output)
    model = models.Model(inputs=my_input, outputs=output)

    print(model.summary())
    return model
