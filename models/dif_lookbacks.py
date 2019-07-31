from keras import models
from keras import layers
from keras.utils import plot_model

def build_model(sigs_1d, sigs_0d, sigs_predict, rho_length_in, rho_length_out, lookbacks, delay,
                rnn_type, rnn_size, rnn_activation, num_rnn_layers,
                dense_0d_size, dense_0d_activation,
                dense_final_size, dense_final_activation, num_final_layers):

    if (rnn_type=='LSTM'):
        rnn_layer = layers.LSTM
    elif (rnn_type=='GRU'):
        rnn_layer=layers.GRU
    else:
        raise ValueError('rnn_type in conf must be GRU or LSTM')

    num_actuators = len(sigs_0d)
    num_sigs_1d = len(sigs_1d)
    num_sigs_predict = len(sigs_predict)

    previous_actuators=layers.Input(shape=(lookbacks[sigs_0d[0]]+1,num_actuators), name="previous_actuators")
    future_actuators=layers.Input(shape=(delay,num_actuators),name="future_actuators")

    actuator_effect = rnn_layer(rho_length_in, activation=rnn_activation)(previous_actuators)
    actuator_effect = layers.Reshape(target_shape=(rho_length_in,1))(actuator_effect)

    future_actuator_effect = rnn_layer(rho_length_in, activation=rnn_activation)(future_actuators)
    future_actuator_effect = layers.Reshape(target_shape=(rho_length_in,1))(future_actuator_effect)
    
    current_profiles=layers.Input(shape=(rho_length_in,num_sigs_1d), name="previous_profiles")
    #take out for the other version

    current_profiles_processed_0=layers.Concatenate()([current_profiles,actuator_effect,future_actuator_effect])
    current_profiles_processed_1=layers.Conv1D(filters=8, kernel_size=2,
                                             padding='same', activation='relu')(current_profiles_processed_0)
    current_profiles_processed_2=layers.Conv1D(filters=8, kernel_size=4,
                                             padding='same', activation='relu')(current_profiles_processed_1)
    current_profiles_processed_3=layers.Conv1D(filters=8, kernel_size=8,
                                             padding='same', activation='relu', 
                                             name='processed_profiles')(current_profiles_processed_2)
    
    final_output=layers.Concatenate()([current_profiles_processed_1,current_profiles_processed_2,current_profiles_processed_3])
    final_output=layers.Conv1D(filters=10, kernel_size=4,
                               padding='same', activation='tanh')(final_output)
    final_output=layers.Conv1D(filters=1, kernel_size=4,
                               padding='same', activation='linear')(final_output)
    
    
#     current_profiles,current_profiles_processed={},{}
#     for sig in sigs_1d:
#         current_profiles[sig]=layers.Input(shape=(rho_length_in,1),"current_profile_{}".format(sig))
#         current_profiles_processed[sig]=layers.Conv1D(filters=3,kernel_size=4)(current_profiles[sig])
    model=models.Model(inputs=[previous_actuators,current_profiles,future_actuators], 
                       outputs=[final_output])

#    model=models.Model(inputs=[previous_actuators]+list(current_profiles.values()), 
#                       outputs=list(current_profiles_processed.values()))

    print(model.summary())
    return model
    #######################
#     num_sigs_1d = len(sigs_1d)
#     num_actuators = len(sigs_0d)

#     num_all_past_sigs = rho_length_in*num_sigs_1d + num_actuators

#     lookback=lookbacks[sigs_0d[0]]
#     # profile work
#     all_past_sigs_input = layers.Input(shape=(lookback+1, num_all_past_sigs,), name = "all_previous_sigs")
#     all_past_sigs_response = layers.Dense(num_all_past_sigs+5, activation='relu')(all_past_sigs_input)
#     all_past_sigs_response = layers.Dense(num_all_past_sigs+10, activation='relu')(all_past_sigs_response)
#     all_past_sigs_response = layers.LSTM(
#         num_all_past_sigs+15, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True)(all_past_sigs_response)
#     all_past_sigs_response = layers.Dense(num_all_past_sigs+20, activation='relu')(all_past_sigs_response)



#     # input layers for future actuators
#     future_actuators_input = layers.Input((delay, num_actuators,), name = "future_actuators")
#     future_actuators_response = layers.Dense(
#         num_actuators+50, activation='relu')(future_actuators_input)
#     future_actuators_response = layers.Dense(
#         num_actuators+70, activation='relu')(future_actuators_response)
#     future_actuators_response = layers.LSTM(
#         num_actuators+90, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True)(future_actuators_response)

#     future_actuators_response = layers.Permute((2,1))(future_actuators_response)
#     future_actuators_response = layers.Dense(lookback+1, activation = "relu")(future_actuators_response)
#     future_actuators_response = layers.Permute((2,1))(future_actuators_response)




#     future_actuators_response = layers.Dense(
#         num_all_past_sigs+20, activation='relu')(future_actuators_response)


#     total_response = layers.Multiply()([all_past_sigs_response, future_actuators_response])

#     total_response = layers.Dense(rho_length_in+40, activation='relu')(total_response)


#     total_response = layers.LSTM(
#         rho_length_in+10, activation='relu', recurrent_activation='hard_sigmoid')(total_response)
# #     total_response = layers.Dense(rho_length_in+10, activation='relu')(total_response)
#     total_response = layers.Dense(rho_length_out, name = "target_{}".format(sigs_predict[0]))(total_response)

#     model = models.Model(
#         inputs=[future_actuators_input, all_past_sigs_input], outputs=total_response)

#     print(model.summary())   

#     return model
    
