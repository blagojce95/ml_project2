# Project 1 : Detecting Higgs Boson

This project was created as part of the Machine Learning course [ **CS-433** ] at EPFL. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
The project was created and tested with the following dependencies:

### Libraries
```
- Anaconda Python 3.6.5
- numpy
- pandas
- h5py
- scikit-learn
- keras-gpu=2.0.8
- tensorflow-gpu=1.4.1
```

### Datasets + additional resources
blabla

## Installing

```
git clone https://github.com/mmilenkoski/ml_project1.git
```

## Project Structure
The project is organized as follows:

    .
    ├── config                          # Configure json files for the project
    ├── data                            # Directory containing the Twitter dataset and GloVe dataset
    ├── models                          # Directory containing the definition of the models
    ├── models_checkpoints              # Directory containing the training checkpoints of the models
    ├── preprocessing_and_loading_data  # Utilities for generating and reading the data
    ├── results                         # Prediction files for submission on CrowdAI
    ├── Baseline_models.ipynb           # Jupyter notebook used for training and testing the baseline models
    ├── README.md                       # README file
    ├── run.py                          # Script for running the optimal model, and creating a file with final predictions
    ├── train_CNN.py                    # Script for training the CNN model
    ├── train_CNN_LSTM.py               # Script for training the CNN-LSTM model
    ├── train_LSTM.py                   # Script for training the LSTM model
    └── train_LSTM_CNN.py               # Script for training the LSTM-CNN model
    
## Running

Before training the model, please unzip the files `data/train.zip` and `data/test.zip` in the folder `data`. You can also unzip the file `data/sample-submission.zip` in order to see the format of the submissions for Kaggle. To reproduce our results run the following command:

``` 
python run.py
```

After running the script `run.py` you can find the generated predictions in the file `predictions/predictions.csv`. Our final predictions are in the file `predictions/predictions_final.csv` for comparison.

## Tune hyperparameters

We obtained the final hyperparameter using holdout validation and grid search. To train the model with your own hyperparameters, change their values in the file `utils/hyperparameters.py`, and run the script `run.py`

## Authors

* Martin Milenkoski     martin.milenkoski@epfl.ch
* Blagoj Mitrevski      blagoj.mitrevski@epfl.ch
* Samuel Bosch          samuel.bosch@epfl.ch
