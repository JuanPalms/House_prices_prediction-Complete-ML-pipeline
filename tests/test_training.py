"""
Module that integrates the fitting process of a linear model and a regularized linear model.
"""
import os,sys
import pytest
import pandas as pd
import yaml
import logging

#This chunk creates absolute paths so that python can look for in the system paths and retrieve modules
CURRENT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(CURRENT)
ROOT = os.path.dirname(PARENT)
SRC_DIR = os.path.join(PARENT, 'src')
sys.path.append(PARENT)
sys.path.append(SRC_DIR)

# Import function from modules that depend on preceeding sys.append()
from utility import column_names_refact,impute_missing_drop_columns,clip_outliers, drop_columns, load_config, non_skprepross


def definir_config(name):
    config_f = load_config(name)
    return config_f

@pytest.fixture(scope="module")
def name_conf():
    return "config.yaml"

"""
Test method
"""
def test_print_config(name_conf):
    """
    Verificar que el archivo config esta correctamente definido
    """
    try:
        config_f=definir_config(name_conf)
        print(config_f)

    except FileNotFoundError as err:
        print("No se encontro el archivo de configuracion en el test")
        raise err
    
    # Verificar que si esta leyendo bien el archivo de configuracion
    try:
        assert config_f['data_directory']=='../data'

    except AssertionError as err:
        logging.error(
            "Testing leer el archivo de configuracion: algo anda mal")
        raise err

def test_len_df(name_conf):
    """
    Test para verificar que la longitud del df es adecuada, 
    al menos dos variables y al menos 1,000 observaciones para
    la etapa de entrenamiento
    """
    try:
        config_f=definir_config(name_conf)
        df=pd.read_csv(os.path.join(config_f["data_directory"]+config_f["raw_data"],config_f["data_train"]))
    
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err
    # Verificar que no venga vacio.
    try:
        #Verify number of observations to train is greater than 10000
        assert df.shape[0] > 1000
        # Verify we have at least two columns in the df so we can model
        assert df.shape[1] >= 2

    except AssertionError as err:
        logging.error(
            "Testing importar_datos: El archivo no trae columnas o filas")
        raise err

def test_preprocesamiento(name_conf):
    """
    Test para verificar que despues del preprocesamiento los datos aun tienen longitud suficiente para 
    entrenar el modelo y que no existen missing values despues del preprocesamiento. 
    """
    try:
        config_f=definir_config(name_conf)
        df=pd.read_csv(os.path.join(config_f["data_directory"]+config_f["raw_data"],config_f["data_train"]))

    except FileNotFoundError as err:
        logging.error("No se encontro el archivo de configuracion en el test")
        raise err
    try:
        df_preprocesados=non_skprepross(df,config_f["numerical"],config_f["nominal"],config_f["to_drop"],config_f["target_name"])
        assert df.shape[0] > 1000
    except AssertionError as err:
        logging.error(
            "Testing importar_datos: El archivo no trae columnas o filas")
        raise err
        
    
    