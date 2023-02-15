import os
import logging
import pickle
import time
import yaml
import pandas as pd
from utility import column_names_refact,impute_missing_drop_columns,clip_outliers, drop_columns, load_config

# Load config file calling load_config function
config_f = load_config("config.yaml")

### TESTING FASE
# Load data
#Read the new data file
try:
    df=pd.read_csv(os.path.join(config_f["data_directory"]+config_f["raw_data"],config_f["data_test"]))
except FileNotFoundError:
    try:
        print("There is no such file in the data directory, you have 5 minutes to load")
        time.sleep(60)
        print("Time elapsed, checking for the availability of data")
        df=pd.read_csv(os.path.join(config_f["data_directory"],config_f["data_test"]))
        time.sleep(5)
        print("File found proceding with the prediction flow")
    except FileNotFoundError:
        logging.info("There is no such file in the data directory")
        

#### DEAL with the target variable in the new set to make predictions
#If there exists a target variable (in case of test sets) separte the target variable from input variables
try:
    # Define X (independent variables) and y (target variable)
    y = df[config_f["target_name"]]
    df = df.drop(config_f["target_name"], axis=1)
except KeyError:
    logging.info(f"This data set has not a target variable or the name is different from the train data")

#Preprocessing
#1) Performs column name refactoring (lowercases cols name and eliminates spaces in the col names).
df,numerical,nominal,to_drop=column_names_refact(
    df,config_f["numerical"],config_f["nominal"],config_f["to_drop"])

#2) Drop unwanted columns
df=drop_columns(df,to_drop)

#2)Drops cols with more than .3 of Nans and replaces with median or mode Nans
#of columns with less than .3 of Nans.
df,numerical,nominal=impute_missing_drop_columns(df, numerical,nominal)
df=clip_outliers(df,numerical)


#Load the pickle object to transform new data
preprocessor = pickle.load( open( config_f["preprocessor_directory"]+config_f['preprocessor_name'], "rb" ) )

# Transform the data
X_test_processed = preprocessor.transform(df)

# Load best model object
best_model= pickle.load( open( config_f["models_directory"]+config_f['gs_best_model_name'], "rb" ) )


try:
    #Make predictions in processed data frame
    predictions=best_model.predict(X_test_processed)
except ValueError:
    logging.info("Format of some variables in the predictor could not be converted")

#### Modify the data directory in config file it send predictions directly to raw
predictions.tofile(config_f["data_directory"]+config_f["predictions_data"]+'/predictions.csv', sep = ',')
