"""
Cleaning module:
This module performs basic data cleaning and processing tasks not related to training or prediction processes. 
"""
import os
import pandas as pd
import yaml
import csv
from utility import column_names_refact,impute_missing_drop_columns,clip_outliers, drop_columns, load_config

#CLEANING PROCESS

# Load config file calling load_config function
config_f = load_config("config.yaml")

# Load data
df=pd.read_csv(os.path.join(config_f["data_directory"]+config_f["raw_data"],config_f["data_train"]))

#Preprocessing
#1) Performs column name refactoring (lowercases cols name and eliminates spaces in the col names).
df,numerical,nominal,to_drop=column_names_refact(
    df,config_f["numerical"],config_f["nominal"],config_f["to_drop"])

#2) Drop unwanted columns=
df=drop_columns(df,to_drop)

#2)Drops cols with more than .3 of Nans and replaces with median or mode Nans
#of columns with less than .3 of Nans.
df,numerical,nominal=impute_missing_drop_columns(df, numerical,nominal)
df=clip_outliers(df,numerical)

#Send clean data before procesing to folder

df.to_csv(os.path.join(config_f["data_directory"]+config_f["clean_data"],config_f["data_clean_train"]),index=False)

# storing modified list of numerical variables
file_n = open(os.path.join(config_f["data_directory"]+config_f["clean_data"],'numerical.txt'),'w')
for var in numerical:
    file_n.write(var+",")
file_n.close()

# storing modified list of nominal variables
file_n = open(os.path.join(config_f["data_directory"]+config_f["clean_data"],'nominal.txt'),'w')
for var in nominal:
    file_n.write(var+",")
file_n.close()
