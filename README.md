# Project 2 : Twitter Sentiment Classification

This project was created as part of the Machine Learning course [ **CS-433** ] at EPFL. We developed a tweet sentiment classification pipeline including preprocessing steps, sequential representation of tweets, and some state-of-the-art neural architectures for tweet classification. The best and final model was obtained with the LSTM architecture. 

## crowdAI submission

- Name: Martin Milenkoski
- ID: 25120
- Link: https://www.crowdai.org/challenges/48/submissions/25120

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
The project was created and tested with the following dependencies:

### Libraries
```
- Anaconda Python 3.6.5
- numpy=1.15.4
- pandas=0.23.4
- h5py=2.8.0
- scikit-learn=0.20.1
- keras-gpu=2.0.8
- tensorflow-gpu=1.4.1
```

### Datasets
The pretrained GloVe embeddings and the Twitter datasets can be downloaded from the following links:

link 1: https://drive.google.com/open?id=1pqY5LHdB4R101G9MUaVygXeUYRgXX8k6

mirror (in case of problems): https://drive.google.com/file/d/1S3BD_hBa16i9mozQ3y75j5OU0B15sDl6/view?usp=sharing

Just extract the zip file into the `data` folder. The data is the same as provided on the GloVe website and crowdAI competition, but named and organized as used in our code. Please make sure that the data folder contains two subdirectories 'twitter-dataset' and 'glove_embeddings' instead of a another single 'data' subdirectory in order for the code to work. The twitter dataset should be in the directory 'data/twitter-dataset' and the glove embeddings should be in 'data/glove_embeddings'.

### Model weights
Our final trained model can be downloaded from the following link:

link 1: https://drive.google.com/open?id=1wbDzVXizOOwCdPRN3rQIWvKN-EvoHgO5

mirror (in case of problems): https://drive.google.com/open?id=1JRSItMMt_yJmc7AMzpZUj_ynru38wwze

Exrtact the zip file into the `models_checkpoints` folder.


## Installing

This project does not require any installation. To use, simply clone the repository to your local machine using the following command:

```
git clone https://github.com/blagojce95/ml_project2.git
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
    ├── run.py                          # Script for running the final model, and creating a file with final predictions
    ├── train_CNN.py                    # Script for training the CNN model
    ├── train_CNN_LSTM.py               # Script for training the CNN-LSTM model
    ├── train_LSTM.py                   # Script for training the LSTM model
    └── train_LSTM_CNN.py               # Script for training the LSTM-CNN model
    
## Reproducing our results

In order to reproduce our results, please unzip the `data.zip` file in the `data` folder and the `LSTM.zip` file in the `models_checkpoints` folder. Then, to obtain the same predictions as us run the following command:

``` 
python run.py
```

After running the script `run.py` you can find the generated predictions in the file `results/LSTM.csv`. Our final predictions are in the file `results/final_LSTM.csv` for comparison.

## Training models

In this project, we have implemented 4 models:

- CNN
- LSTM
- CNN-LSTM
- LSTM-CNN

If you want to train some of the model with your own parameters, you can use the train_\*.py files. For example, for training CNN, you can use the following command. 

```
python train_CNN.py
```

The model parameters can be changed by editing the params dictionary in the `train_CNN.py` file, and the model architecture can be changed by editing the build_model() method in the `models/CNN_model.py` file.

## Authors

* Blagoj Mitrevski      blagoj.mitrevski@epfl.ch
* Martin Milenkoski     martin.milenkoski@epfl.ch
* Samuel Bosch          samuel.bosch@epfl.ch
