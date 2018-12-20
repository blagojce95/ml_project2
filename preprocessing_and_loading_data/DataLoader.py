import os
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing_and_loading_data.generate_data import generate_data


class DataLoader():
    """This class is used for loading the data.
    
    Instead of preprocessing the tweets, rebulding the vocabulary and generating the embedding matrix 
    everytime we run a model, we implemented this class which upon first creation of a DataLoader object 
    generates everything and saves the generated files. Every time the object is cerated with the same 
    parameters (glove_dimension, max_words, full) the object reads the saved data, instead of regenerating 
    it again. This way we save on time when training the models.

    Parameters
    ----------
    glove_dimension : int
        Integer number representing which GloVe embedding to be used, must be 25, 50, 100 or 200.
    max_words : int
        The length of the tweet, in our case the number of word vectors every tweet should contain.
    full : boolean
        Whether to use full or sample dataset.
    path : string
        The path where the generated data is saved.

    """
    def __init__(self, glove_dimension=200, max_words=40, full=True):
        self.glove_dimension = glove_dimension
        self.max_words = max_words
        self.full = full
        self.path = "data/glove_%s_words_%s_full_%s" % (glove_dimension, max_words, full)
        self.load_data()
    
    def load_data(self):
        """Helper function for loading the data.
        
        First, it is checked if the data has already been generated. If it has been, then it is simply 
        loaded. If not, than the `generate_data` function from `generate_data.py` file is called to generate 
        the data.

        """
        if not os.path.isdir(self.path):
            print("Data for these parameters not generated!!!")
            print("Generating data, this can take a while...")
            generate_data(self.glove_dimension, self.max_words, self.full)
            print("Generating data done!")
        self.embedding_matrix = np.load(os.path.join(self.path, "embedding_matrix.npy"))
        self.X_train = np.load(os.path.join(self.path, "X_train.npy"))
        self.Y_train = np.load(os.path.join(self.path, "Y_train.npy"))
        self.X_test = np.load(os.path.join(self.path, "X_test.npy"))
    
    def get_train_test_split(self):
        """Returns 9:1 train/teset split of the data.

        """
        return train_test_split(self.X_train, self.Y_train, shuffle=True, test_size=0.1, random_state=0)
    
    def get_test_data(self):
        """Returns the test data which is used for generating the csv file for submitting to crowdAI.

        """
        return self.X_test
    
    def get_embedding_matrix(self):
        """Returns the generated embedding matrix for our vocabulary.

        """
        return self.embedding_matrix
