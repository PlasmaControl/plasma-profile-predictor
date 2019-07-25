oldtimes = 5
profile_length = 30
future_actuators = 3
num_actuators = 4

etemp_input = keras.layers.Input((oldtimes, profile_length))
etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_input)
etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_response)
etemp_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(etemp_response)
etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_response)

pressure_input = keras.layers.Input((oldtimes, profile_length))
pressure_response = keras.layers.Dense(
    profile_length, activation='relu')(pressure_input)
pressure_response = keras.layers.Dense(
    profile_length, activation='relu')(pressure_response)
pressure_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(pressure_response)
pressure_response = keras.layers.Dense(
    profile_length, activation='relu')(pressure_response)

edens_input = keras.layers.Input((oldtimes, profile_length))
edens_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_input)
edens_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_response)
edens_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(edens_response)
edens_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_response)

itemp_input = keras.layers.Input((oldtimes, profile_length))
itemp_response = keras.layers.Dense(
    profile_length, activation='relu')(itemp_input)
itemp_response = keras.layers.Dense(
    profile_length, activation='relu')(itemp_response)
itemp_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(itemp_response)
itemp_response = keras.layers.Dense(
    profile_length, activation='relu')(itemp_response)

rotation_input = keras.layers.Input((oldtimes, profile_length))
rotation_response = keras.layers.Dense(
    profile_length, activation='relu')(rotation_input)
rotation_response = keras.layers.Dense(
    profile_length, activation='relu')(rotation_response)
rotation_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(rotation_response)
rotation_response = keras.layers.Dense(
    profile_length, activation='relu')(rotation_response)

ffprime_input = keras.layers.Input((oldtimes, profile_length))
ffprime_response = keras.layers.Dense(
    profile_length, activation='relu')(ffprime_input)
ffprime_response = keras.layers.Dense(
    profile_length, activation='relu')(ffprime_response)
ffprime_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(ffprime_response)
ffprime_response = keras.layers.Dense(
    profile_length, activation='relu')(ffprime_response)

actuator_input = keras.layers.Input((oldtimes+future_actuators, num_actuators))
actuator_response1 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_input)
actuator_response1 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_response1)
actuator_response1 = keras.layers.LSTM(
    oldtimes, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(actuator_response1)
actuator_response2 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_input)
actuator_response2 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_response2)
actuator_response2 = keras.layers.LSTM(
    oldtimes, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(actuator_response2)
actuator_response3 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_input)
actuator_response3 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_response3)
actuator_response3 = keras.layers.LSTM(
    oldtimes, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(actuator_response3)
actuator_response4 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_input)
actuator_response4 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_response4)
actuator_response4 = keras.layers.LSTM(
    oldtimes, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(actuator_response4)
actuator_response5 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_input)
actuator_response5 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_response5)
actuator_response5 = keras.layers.LSTM(
    oldtimes, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(actuator_response5)
actuator_response6 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_input)
actuator_response6 = keras.layers.Dense(
    oldtimes, activation='relu')(actuator_response6)
actuator_response6 = keras.layers.LSTM(
    oldtimes, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(actuator_response6)

etemp_actuator_response = keras.layers.Dot(axes=(2, 1))(
    [actuator_response1, etemp_response])
etemp_actuator_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(etemp_actuator_response)
etemp_actuator_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_actuator_response)
etemp_actuator_response = keras.layers.Dense(
    profile_length)(etemp_actuator_response)


itemp_actuator_response = keras.layers.Dot(axes=(2, 1))(
    [actuator_response2, itemp_response])
itemp_actuator_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(itemp_actuator_response)
itemp_actuator_response = keras.layers.Dense(
    profile_length, activation='relu')(itemp_actuator_response)
itemp_actuator_response = keras.layers.Dense(
    profile_length)(itemp_actuator_response)


edens_actuator_response = keras.layers.Dot(axes=(2, 1))(
    [actuator_response3, edens_response])
edens_actuator_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(edens_actuator_response)
edens_actuator_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_actuator_response)
edens_actuator_response = keras.layers.Dense(
    profile_length)(edens_actuator_response)


pressure_actuator_response = keras.layers.Dot(axes=(2, 1))(
    [actuator_response4, pressure_response])
pressure_actuator_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(pressure_actuator_response)
pressure_actuator_response = keras.layers.Dense(
    profile_length, activation='relu')(pressure_actuator_response)
pressure_actuator_response = keras.layers.Dense(
    profile_length)(pressure_actuator_response)


ffprime_actuator_response = keras.layers.Dot(axes=(2, 1))(
    [actuator_response5, ffprime_response])
ffprime_actuator_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(ffprime_actuator_response)
ffprime_actuator_response = keras.layers.Dense(
    profile_length, activation='relu')(ffprime_actuator_response)
ffprime_actuator_response = keras.layers.Dense(
    profile_length)(ffprime_actuator_response)


rotation_actuator_response = keras.layers.Dot(axes=(2, 1))(
    [actuator_response6, rotation_response])
rotation_actuator_response = keras.layers.LSTM(
    profile_length, activation='relu', recurrent_activation='hard_sigmoid',
    return_sequences=True)(rotation_actuator_response)
rotation_actuator_response = keras.layers.Dense(
    profile_length, activation='relu')(rotation_actuator_response)
rotation_actuator_response = keras.layers.Dense(
    profile_length)(rotation_actuator_response)


etemp_etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_actuator_response)
itemp_etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(itemp_actuator_response)
edens_etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_actuator_response)
ffprime_etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(ffprime_actuator_response)
pressure_etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(pressure_actuator_response)
rotation_etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(rotation_actuator_response)

edens_etemp_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_actuator_response)
itemp_edens_response = keras.layers.Dense(
    profile_length, activation='relu')(itemp_actuator_response)
edens_edens_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_actuator_response)
ffprime_edens_response = keras.layers.Dense(
    profile_length, activation='relu')(ffprime_actuator_response)
pressure_edens_response = keras.layers.Dense(
    profile_length, activation='relu')(pressure_actuator_response)
rotation_edens_response = keras.layers.Dense(
    profile_length, activation='relu')(rotation_actuator_response)


etemp_total_response = keras.layers.Multiply()([etemp_etemp_response,
                                                itemp_etemp_response,
                                                edens_etemp_response,
                                                pressure_etemp_response,
                                                ffprime_etemp_response,
                                                rotation_etemp_response])
etemp_total_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_total_response)
etemp_total_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_total_response)
etemp_total_response = keras.layers.LSTM(
    profile_length, activation='relu')(etemp_total_response)
etemp_total_response = keras.layers.Dense(
    profile_length, activation='relu')(etemp_total_response)
etemp_total_response = keras.layers.Dense(profile_length)(etemp_total_response)

edens_total_response = keras.layers.Multiply()([edens_edens_response,
                                                itemp_edens_response,
                                                edens_etemp_response,
                                                pressure_edens_response,
                                                ffprime_edens_response,
                                                rotation_edens_response])
edens_total_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_total_response)
edens_total_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_total_response)
edens_total_response = keras.layers.LSTM(
    profile_length, activation='relu')(edens_total_response)
edens_total_response = keras.layers.Dense(
    profile_length, activation='relu')(edens_total_response)
edens_total_response = keras.layers.Dense(profile_length)(edens_total_response)

model = keras.models.Model(
    inputs=[actuator_input, etemp_input, edens_input,
            itemp_input, ffprime_input, pressure_input, rotation_input],
    outputs=[etemp_total_response, edens_total_response])
