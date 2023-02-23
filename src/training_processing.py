"""
Module that integrates the fitting process of a linear model and a regularized linear model.
"""
import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import yaml
from utility import column_names_refact,impute_missing_drop_columns,clip_outliers, drop_columns, load_config
import argparse
import logging

# Load config file calling load_config function
config_f = load_config("config.yaml")

# Add parser for cross-fold validation
parser = argparse.ArgumentParser(
    prog="training_preprocessing.py", 
    usage="%(prog)s [options] folds (number of cross validation folds) ",
    description="Training script with cross fold validation")
parser.add_argument('folds', type=int)
args = parser.parse_args()

#configuration of loggins
logging.basicConfig(
    filename=os.path.join(config_f["log_directory"], config_f["log_file"]),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


################ PREPROCESSING TRAIN DATA
# 1) Load clean 

df=pd.read_csv(os.path.join(config_f["data_directory"]+config_f["clean_data"],config_f["data_clean_train"]))

# Define X (independent variables) and y (target variable)
y = df[config_f["target_name"]]
# Drop target variable from df to be processed
df = df.drop(config_f["target_name"], axis=1)

#Load modified numerical names after cleaning process
with open(os.path.join(config_f["data_directory"]+config_f["clean_data"],'numerical.txt'),'r') as numericas:
    nums = numericas.readlines()
numerical=[]
for l in nums:
    n_list= l.split(",")
    numerical.append(n_list)
numerical=numerical[0]
numerical = numerical[:-1]

#Load modified nominal names after cleaning process
with open(os.path.join(config_f["data_directory"]+config_f["clean_data"],'nominal.txt'),'r') as nominales:
    noms = nominales.readlines()
nominal=[]
for l in noms:
    n_list= l.split(",")
    nominal.append(n_list)
nominal=nominal[0]
nominal = nominal[:-1]



#3) transformers for nominal attributes
nominal_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

#4) Transformes for numeric attributes
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
    )

#5) Transformers for ordinal attributes
ordinal_transformer = Pipeline(
    steps=[("ord_encoder",OrdinalEncoder(categories=[['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex']]))])

#6) Complete preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical),
        ("nom", nominal_transformer, nominal)
        ],
    remainder='drop'
    )

#### Apply preprocessing on train data and save preprocessing params in picklle file

preprocessor.fit(df)

# Transform the training dataset
X_train_preprocessed = preprocessor.transform(df)


# save the processor to disk
filename_process = config_f["preprocessor_name"]
#Write the scaler to folder
with open(config_f["preprocessor_directory"]+filename_process, 'wb') as file_process:
    pickle.dump(preprocessor, file_process)


######### MODELING ON TRAIN SET

###GRID SEARCH PIPELINE
# Initialze the regressors
clf1 = LinearRegression()
clf2 = ElasticNet()
clf3 = RandomForestRegressor()

# Initiaze the hyperparameters for each dictionary
param1 = {}
param1['regressor'] = [clf1]

# Parameters for Elastic Net
param2 = {}
param2['regressor__alpha'] = [.5, 1]
param2['regressor__l1_ratio'] = [0,.5, 1]
param2['regressor'] = [clf2]

### Parameters for RandomForest
param3 = {}
param3['regressor__n_estimators'] = [50, 100, 200]
param3['regressor__criterion'] = ['squared_error','absolute_error']
param3['regressor__min_samples_leaf'] = [10,20,50]
param3['regressor'] = [clf3]


# Set the pipeline whit all the considered regressors or classifiers
pipeline = Pipeline([('regressor', clf1)])
params = [param1, param2, param3]


# Train the grid search model
gs = GridSearchCV(
    pipeline, params, cv=args.folds,
    scoring='neg_mean_absolute_error').fit(X_train_preprocessed, y)

# Best performing model and its corresponding hyperparameters
parameters_best_model=gs.best_params_


# save  best model params to disk
filename_bm_params = config_f["best_model_parameters"]

#Write the best model params to folder
with open(config_f["models_directory"]+filename_bm_params, 'wb') as file_params:
    pickle.dump(parameters_best_model, file_params)

# save  best GS model to disk
filename_bm= config_f["gs_best_model_name"]

#Write the best model to folder
with open(config_f["models_directory"]+filename_bm, 'wb') as file_bm:
    pickle.dump(gs, file_bm)
