from models.BaseModel import BaseModel
from keras.models import Sequential
from keras.layers import *

class LSTM_CNN_model(BaseModel):
    """This is the LSTM-CNN model. It inherits from the `BaseModel` class.
    
    This class only implements the `build_model` function used to define and compile the keras model.

    Attributes
    ----------
    model_name : str
        The `model_name` parameter is used as a name for a checkpoint dictionary inside models checkpoints dictionary, for more details see `BaseModel` class.

    """
    def __init__(self, model_name):
        super().__init__(model_name)


    def build_model(self, embedding_matrix, seq_len, params):
        self.model = Sequential()

        self.model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=seq_len, \
                                 weights=[embedding_matrix], name='emb'))

        self.model.add(LSTM(units=params["LSTM_num_neurons"], dropout=params["LSTM_dropout"], recurrent_dropout=params["LSTM_recurrent_dropout"], return_sequences=True))

        self.model.add(Conv1D(filters=params["CNN_filters"], kernel_size=params["CNN_kernel_size"], activation=params["CNN_activation"]))
        
        self.model.add(MaxPooling1D(pool_size=params["CNN_pool_size"]))

        self.model.add(Flatten())
        self.model.add(Dense(1, activation=params["DENSE_activation"]))
        self.model.compile(loss=params["loss"], optimizer=params["optimizer"], metrics=['accuracy'])
