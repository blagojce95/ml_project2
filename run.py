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

# Import the model
from models.LSTM_model import LSTM_model

# The model_name is used to load a model from checkpoint files in the "models_checkpoints" dictionary
model_name = "LSTM"
lstm = LSTM_model(model_name)

# If the model already exists, then lstm.load() will load the preexisting model including its architecture and weights
lstm.load()
print(lstm.model.summary())

# Get the testing data for submitting on crowdAI
X_crowdAI_test = dl.get_test_data()
# Save the predictions in the "predictions" folder with "model_name" name of  the csv file
lstm.predict_and_save_predictions(X_crowdAI_test, model_name, batch_size=params["batch_size"])

print('_________________________________')
print(model_name)
print('_________________________________')
