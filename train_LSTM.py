# Make sure the training is reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
from preprocessing_and_loading_data.DataLoader import DataLoader
max_words=40
# Create DataLoader object to get the training, validation and testing data
dl = DataLoader(glove_dimension=200, max_words=40, full=True)

# Load the data, including the embedding matrix for our vocabulary
X_train, X_val, Y_train, Y_val = dl.get_train_test_split()
embedding_matrix = dl.get_embedding_matrix()
print(X_train.shape)
print(Y_train.shape)


# Define the parameters of the model
params = {
    'LSTM_num_neurons': 100,
    'LSTM_dropout': 0.0,
    'LSTM_recurrent_dropout': 0.0,
    'epochs': 5,
    'batch_size': 512,
    'DENSE_activation': 'sigmoid',
    'loss': 'binary_crossentropy',
    'optimizer': 'RMSprop'
}

# Import the model
from models.LSTM_model import LSTM_model

# The model_name is used to create checkpoint files in the "models_checkpoints" folder
model_name = "LSTM"
lstm = LSTM_model(model_name)

# Build the model and print its summary
lstm.build_model(embedding_matrix, max_words, params)
print(lstm.model.summary())
# Train the model, saves the weights which give the best validation loss
lstm.train(X_train, Y_train, epochs=params["epochs"], batch_size=params["batch_size"], validation_data=(X_val, Y_val))

print('_________________________________')
print(model_name)
print('_________________________________')
