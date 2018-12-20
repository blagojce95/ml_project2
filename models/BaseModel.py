import keras
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
import json
import os


class BaseModel():
    """This class is base class for the models and all the modes should extend this class.
    
    The base class implements all the generic methods and functionalities for the models such as training, saiving the models during training, evaluation of the models, loading of previously saved models and generating csv file with predictions for submitting on crowdAI platform. This way, the functionalities are implemented only once and all the models reuse these functionalities.
    
    To create a new model, the user first must create a class in the `models` dictionary, inherit this `BaseModel` class and implement the `build()` function which sets the `model` parameter. After that, the user can use all the functions which are implemented here in the `BaseModel` class.

    Parameters
    ----------
    model_name : str
        The `model_name` parameter is used as a name for a checkpoint dictionary inside models checkpoints dictionary, during the training of the model in this dictionary we save the model with its weights, the history of the training and validation loss/accuracy and the tensorboard.
    model : object
        This is object representing the actual keras model.
    model_dir : str
        The path to the models dictionary obtained with concatenating the models checkpoint dictionary with the models name.
    checkpoint_path : str
        The path to the hdf5 file where the models weights are stored.
    tensorboard_path : str
        The path to the dictionary where the tensorboard files are stored.
    model_json_path : str
        The path to the json file where the model architecture is stored.
    history_path : str
        The path to the history dictionary where the training and validation loss/accuracy are stored.

    Attributes
    ----------
    model_name : str
        The name of the model given by the user.

    """
    def __init__(self, model_name):
        # Read the constants from the json file
        with open("config/constants.json") as f:
            self.CONS = json.load(f)
        self.model_name = model_name
        self.model = None
        self.model_dir = os.path.join(self.CONS["save_dir"], model_name)
        # Check if a model with the given name exists
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            # If a model exists, than we can either load it and use it for prediction or continue training it
            print("MODEL WITH THIS NAME EXISTS! Delete the save_dir or use load() function to load the existing model!")
        # Initialize all the paths
        self.checkpoint_path = os.path.join(self.model_dir, "best_checkpoint.hdf5")
        self.tensorboard_path = os.path.join(self.model_dir, "tensorboard")
        self.model_json_path = os.path.join(self.model_dir, "model.json")
        self.history_path = os.path.join(self.model_dir, "history")
        
    def build_model(self, embedding_matrix, seq_len, params):
        raise NotImplementedError
        
        
    def train(self, X, y, epochs, batch_size, validation_data=None):
        """This method is used for training the model.
        
        After the `model` parameter is set, this function can be used to train the model. Before training the model, the model arhitecture is saved, tensorboard is created for monitoring the training process and while training the model after each epoch the models weights and the training and validation loss/accuracy are stored. All these files can be found in `model_dir` dictionary.

        Parameters
        ----------
        X : ndarray
            Numpay array containing the training data.
        y : ndarray
            Numpay array containing the training labels.
        epochs : int
            Number of epochs to train the model.
        batch_size : int
            Number of samples per gradient update.
        validation_data : (ndarray, ndarray)
            Tuple on which to evaluate the loss and the accuracy at the end of each epoch.

        """
        # For reproducibility
        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)
        # Create checkpoint callback object
        checkpointer = ModelCheckpoint(filepath=self.checkpoint_path, save_best_only=False, save_weights_only=True)
        # Create tensorboard callback object
        tensorboard = TensorBoard(log_dir=self.tensorboard_path, batch_size=batch_size)
        # Save the model architecture
        model_json = self.model.to_json()
        json.dump(model_json, open(self.model_json_path, "w"))
        # Create history (defined at the end of this file) callback object
        hist = LossHistory()
        hist.set_history_path(self.history_path)
        # Train model
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=[checkpointer, tensorboard, hist])
        
        
    def evaluate(self, X, y, batch_size):
        """This method is used for evaluating the model.
        
        This model is a simple wraper for the keras evaluate method.

        Parameters
        ----------
        X : ndarray
            Numpay array containing the evaluation data.
        y : ndarray
            Numpay array containing the evaluation labels.
        batch_size : int
            Number of samples per evaluation step.
            
        Returns
        -------
        float
            Accuracy of the model on the evaluation data.

        """
        return self.model.evaluate(X, y, batch_size=batch_size)
        
    def load(self):
        """This method is used for loading the model.
        
        If a model exists with the initialized name, this function can be used to load its architecture, its weights and comple the model. After loading the model, it can be used for predicting new data or it can be retrained on new data.

        """
        model_json = json.load(open(self.model_json_path, "r"))
        self.model = model_from_json(model_json)
        self.model.load_weights(self.checkpoint_path)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Loading done!")
        
    def predict_and_save_predictions(self, X_test, file_name, batch_size=1024):
        """This method is used for predicting the labels of the `X_test` data and saving them in csv format ready for submitting on crowdAI.
        
        After the predictions are obtained, the zeros are replaced by `-1`, and an Id is assigned to every prediction. Everything is stored in a pandas DataFrame and saved in csv format in the `results` dictionary.

        Parameters
        ----------
        X_test : ndarray
            Numpay array containing the testing data.
        file_name : string
            Name of the file where the predictions will be stored.            
        batch_size : int
            Number of samples per evaluation step.

        """
        result = pd.DataFrame()
        predicted_classes = self.model.predict_classes(X_test, batch_size=batch_size)
        result["Id"]= range(1, predicted_classes.shape[0]+1)
        result["Prediction"] = predicted_classes
        result["Prediction"].replace(0, -1, inplace=True)
        result.to_csv(os.path.join(self.CONS["results_dir"], file_name + ".csv"))

class LossHistory(keras.callbacks.Callback):
    """This is a helper `Callback` class for saving the train and validation accuracy/loss after every epoch.

    """
    def on_train_begin(self, logs={}):
        self.losses = []
    
    def set_history_path(self, path):
        self.history_path = path
    
    def on_epoch_end(self, epoch, logs={}):
        # save the history of the model
        if os.path.isfile(self.history_path + ".npy"):
            h1 = np.load(self.history_path + ".npy").item()
            merged_history = {}
            for key in h1.keys():
                merged_history[key] = h1[key] + [logs[key]]
            np.save(self.history_path, merged_history)
        else:
            merged_history = {}
            for key in logs.keys():
                merged_history[key] = [logs[key]]
            np.save(self.history_path, merged_history)