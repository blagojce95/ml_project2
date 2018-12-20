# Project 2 : Twitter Sentiment Classification

This project was created as part of the Machine Learning course [ **CS-433** ] at EPFL. ABSTRACT HERE

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
The pretrained GloVe embeddings and the Twitter datasets can be downloaded from the following links:

link1: https://drive.google.com/open?id=1pqY5LHdB4R101G9MUaVygXeUYRgXX8k6
link2: mac dodaj link tuka za redundunacy

Just extract the zip file into the `data` folder.

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

Before running the model, please unzip the `data.zip` file in the folder `data`. To reproduce our results run the following command:

``` 
python run.py
```

After running the script `run.py` you can find the generated predictions in the file `results/LSTM.csv`. Our final predictions are in the file `results/final_LSTM.csv` for comparison.

## Training models

If you want to train some of the model for example the CNN model, just run:

```
python train_CNN.py
```

The model parameters can be changed by editing the `train_CNN.py` file and the model architecture by editing the `models/CNN_model.py` file.

## Authors

* Blagoj Mitrevski      blagoj.mitrevski@epfl.ch
* Martin Milenkoski     martin.milenkoski@epfl.ch
* Samuel Bosch          samuel.bosch@epfl.ch
