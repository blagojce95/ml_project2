# Make sure the training is reproducible
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(3)

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
    'CNN_filters': 80,
    'CNN_kernel_size': 10,
    'CNN_activation': "relu",
    'MP_pool_size': 2,
    'epochs': 5,
    'batch_size': 1024,
    'DENSE_activation': 'sigmoid',
    'loss': 'binary_crossentropy',
    'optimizer': 'RMSprop'
}

# Import the model
from models.CNN_model import CNN_model

# The model_name is used to create checkpoint files in the "models_checkpoints" folder
model_name = "CNN"
cnn = CNN_model(model_name)

# Build the model and print its summary
cnn.build_model(embedding_matrix, max_words, params)
print(cnn.model.summary())
# Train the model, saves the weights which give the best validation loss
cnn.train(X_train, Y_train, epochs=params["epochs"], batch_size=params["batch_size"], validation_data=(X_val, Y_val))

print('_________________________________')
print(model_name)
print('_________________________________')
