from models.BaseModel import BaseModel
from keras.models import Sequential
from keras.layers import *

class LSTM_model(BaseModel):
    def __init__(self, model_name):
    """This is the LSTM model. It inherits from the `BaseModel` class.
    
    This class only implements the `build_model` function used to define and compile the keras model.

    Attributes
    ----------
    model_name : str
        The `model_name` parameter is used as a name for a checkpoint dictionary inside models checkpoints dictionary, for more details see `BaseModel` class.

    """
        super().__init__(model_name)


    def build_model(self, embedding_matrix, seq_len, params):
        """This method is used to define and compile the keras model.
        
        The first layer allways is the Embedding layer used to embed the word sequences to vector sequences which than are send to the LSTM layer. After the LSTM layer there is a Dense layer which outputs the probability for the tweet being of the first class.

        Parameters
        ----------
        embedding_matrix : ndarray
            Numpay matrix containing the GloVe embeddings for our training vocabulary.
        seq_len : ndarray
            The length of the tweet, in our case the number of word vectors every tweet should contain.
        params : dic
            Dictionary containing all the parameters for the LSTM model like "LSTM_num_neurons" etc...

        """
        self.model = Sequential()
        
        self.model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=seq_len, \
                                 weights=[embedding_matrix], name='emb'))
        
        self.model.add(LSTM(units=params["LSTM_num_neurons"], dropout=params["LSTM_dropout"], recurrent_dropout=params["LSTM_recurrent_dropout"]))
        
        self.model.add(Dense(1, activation=params['DENSE_activation']))
        
        self.model.compile(loss=params["loss"], optimizer=params["optimizer"], metrics=['accuracy'])
