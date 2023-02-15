"""
Module that defines functions used in the preprocessing steps of a data science
project
"""
import os
import yaml
import logging

# folder to load config file
CONFIG_PATH = "../"

# Function to load yaml configuration file
def load_config(config_name):
    """
    Sets the configuration file path
    Args:
    config_name: Name of the configuration file in the directory
    Returns:
    Configuration file
    """
    with open(os.path.join(CONFIG_PATH, config_name), encoding="utf-8") as conf:
        config = yaml.safe_load(conf)
    return config

def column_names_refact(df, numerical, nominal, to_drop):
    """
    Function to refactor column names into lowercase and names without spaces
    Args:
        df (pandas dataframe): dataframe
        numerical (list):   list of numerical attributes
        ordinal   (list):   list of ordinal attributes
        nominal   (list):   list of nominal attributes
    Returns:
        df (pandas dataframe): dataframe with modified column names
        numerical (list):   modified list of numerical attributes
        ordinal   (list):   modified list of ordinal attributes
        nominal   (list):   modified list of nominal attributes
    """

    df= df.rename(columns=str.lower)
    df= df.rename(columns=str.strip)
    numerical=[x.lower().strip() for x in numerical]
    nominal=[x.lower().strip() for x in nominal]
    to_drop=[x.lower().strip() for x in to_drop]

    return df,numerical,nominal, to_drop

def impute_missing_drop_columns(df,numerical,nominal):
    """
    Function to impute missing values with median or mode to features containing less
    than 50% of missing values and drops the columns with more than 50% of missing values
    Args:
        df (pandas dataframe): dataframe
        numerical (list):   list of numerical attributes
        nominal   (list):   list of nominal attributes
    Returns:
        df (pandas dataframe): dataframe
        numerical (list):   modified list of numerical attributes
        nominal   (list):   modified list of nominal attributes
        """
    # check for null values
    null_val = (df.isnull()
                # sum of null values
                    .sum()
                # sort values in descending order
                    .sort_values(ascending=False)
                # reset index
                    .reset_index())
    #rename columns
    null_val.columns = ['attribute','count']
    #Add percentage of null values column
    null_val['percentage']=(null_val['count']/df.shape[0])


    # iterate through df to impute median or mode
    to_remove=[]
    for index,row in null_val.iterrows():
        # if percentage of nulls between 0 and 30%
        if 0<row["percentage"]<=.3:
            # print attribute and percentage of nulls
            logging.debug(f'{row["attribute"]} has {row["percentage"]*100} percent of null values')
            #deal with numerical attributes
            if row["attribute"] in numerical:
                # impute median
                df.loc[(df[row['attribute']].isnull()==True),
                       row['attribute']]=df[row['attribute']].median()
            # impute mode to nominal and ordinal variables
            elif row["attribute"] not in numerical:
                df.loc[(df[row['attribute']].isnull()==True),
                       row['attribute']]=df[row["attribute"]].mode(dropna=True)[0]
                # if percentage is greater than 30% remove columns
        elif row["percentage"]>.3:
            to_remove.append(row["attribute"])
        else:
            continue

    remove = set(to_remove)
    numerical = [x for x in numerical if x not in remove]
    nominal = [x for x in nominal if x not in remove]
    df.drop(columns=to_remove, inplace=True)


    return df,numerical,nominal


def to_categorical(df, nominal):
    """
    Converts all object type columns into categorical columns
    Args:
    df (pandas dataframe)
    nominal: list of nominal columns

    Returns:
    df (pandas dataframe)
    """

    for str_obj_col in nominal:
        df[str_obj_col] = df[str_obj_col].astype("category")

    return df


def clip_outliers(df,numerical):
    """
    Clips all the numerical columns to the 98% percentile avoiding extreme (upper bound) values
    Args:
    df (pandas dataframe): dataframe
    numerical (list):   list of numerical attributes
    Returns:
    df (pandas dataframe): dataframe without outliers in numerical attributes
    """
    for column in numerical:
        df[column].clip(upper=df[column].quantile(.98), inplace=True)
    return df


def drop_columns(df, to_drop):
    """
    Removes the unnecesary columns for the proyect
    Args:
        df (pandas dataframe): dataframe
        numerical (list):   list of numerical attributes
        ordinal   (list):   list of ordinal attributes
        nominal   (list):   list of nominal attributes
        to_drop   (list):   list of columns to drop
    Returns:
        df (pandas dataframe): dataframe with modified column names
        numerical (list):   modified list of numerical attributes
        ordinal   (list):   modified list of ordinal attributes
        nominal   (list):   modified list of nominal attributes
    """
    df=df.drop(to_drop, axis=1)
    return df



def non_skprepross(df, numerical, nominal, to_drop,target):
    y = df[target]
    # Drop target variable from df to be processed
    df = df.drop(target, axis=1)
    #Preprocessing
    #1) Performs column name refactoring (lowercases cols name and eliminates spaces in the col names).
    df,numerical,nominal,to_drop=column_names_refact(
    df,numerical,nominal,to_drop)
    #2) Drop unwanted columns=
    df=drop_columns(df,to_drop)
    #2)Drops cols with more than .3 of Nans and replaces with median or mode Nans
    #of columns with less than .3 of Nans.
    df,numerical,nominal=impute_missing_drop_columns(df, numerical,nominal)
    df=clip_outliers(df,numerical)
    return df


