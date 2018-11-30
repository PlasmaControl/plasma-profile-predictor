from keras import models
from keras import layers

# Still need to add the proper input_shape to the train.py
#input_shape: input_shape=(None, len(train_data[0][0]))
#output_shape: output_shape=len(train_data[0][0])-num_sigs
def build_model(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.GRU(16, input_shape=input_shape))
    model.add(layers.Dense(output_shape))
    model.compile(optimizer=optimizers.RMSprop(lr=.001),
                  metrics=['mae'], 
                  loss='mse')
    return model
