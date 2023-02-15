# House prices prediction

## Objective 

This project establishes a data processing and modeling pipeline for predicting house prices in England. 
The data is publicly available at the following Kaggle link. 

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Repository structure

palmeros_tarea05
├── config.yaml
├── data
│   ├── predictions
│   │   └── predictions.csv
│   ├── raw
│   │   ├── test.csv
│   │   └── train.csv
│   └── transformer
│       └── preprocessor.pkl
├── __init__.py
├── logs
│   └── results.log
├── models
│   ├── best_model_params.pkl
│   └── gs_bestmodel.pkl
├── notebooks
│   └── Data_Exploration.ipynb
├── README.md
├── requirements.txt
├── src
│   ├── predictions.py
│   ├── training_processing.py
│   └── utility.py
└── tests
    └── test_training.py




## Details
config.yaml: Defines paths and variables that are employed in training or predicition process
data: Folder contaning unprocessed data, predictions generated in the pipeline and a pickle file with the processing steps of sklearn
Logs: Folder containing log filles for debugging the project
notebooks: Jupyter notebooks containing basic exploratory data analysis
src: python scripts that implement the whole data processing and modeling pipeline
src/predictions.py: predicts price of houses based on best parameters of bets model obtained in training stage.
src/training_processing.py: performs preprocessing of data and training of models, returns the best parameters from cross validation.
tests: contains test script of the functions defined in the utility.py script. 



