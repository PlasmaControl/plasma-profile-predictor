import keras

oldtimes = 5
profile_length = 65
future_actuators = 3
num_actuators = 4

profile_input = keras.layers.Input((oldtimes, profile_length))
profile_response = keras.layers.Dense(30, activation='relu')(profile_input)
profile_response = keras.layers.Dense(30, activation='relu')(profile_response)
profile_response = keras.layers.LSTM(
    30, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True)(profile_response)
profile_response = keras.layers.Dense(30, activation='relu')(profile_response)

actuator_input = keras.layers.Input((oldtimes+future_actuators, num_actuators))
actuator_response = keras.layers.Dense(
    num_actuators, activation='relu')(actuator_input)
actuator_response = keras.layers.Dense(
    num_actuators, activation='relu')(actuator_response)
actuator_response = keras.layers.LSTM(
    num_actuators, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True)(actuator_response)
actuator_response = keras.layers.Permute((2, 1))(actuator_response)
actuator_response = keras.layers.Dense(5, activation='relu')(actuator_response)
actuator_response = keras.layers.Permute((2, 1))(actuator_response)
actuator_response = keras.layers.Dense(
    30, activation='relu')(actuator_response)
actuator_response = keras.layers.Dense(
    30, activation='relu')(actuator_response)

total_response = keras.layers.Multiply()([actuator_response, profile_response])
total_response = keras.layers.LSTM(
    30, activation='relu', recurrent_activation='hard_sigmoid')(total_response)
total_response = keras.layers.Dense(45, activation='relu')(total_response)
total_response = keras.layers.Dense(65)(total_response)

model = keras.models.Model(
    inputs=[actuator_input, profile_input], outputs=total_response)
