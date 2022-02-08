
import numpy as np, pandas as pd, random
import sys, os
import json
import tensorflow as tf
import algorithm.load_config as cfg
from algorithm.preprocessing.preprocess_utils import save_preprocessor, load_preprocessor
from algorithm.elasticnet import save_model, load_model



def set_seeds(seed_value=42):
    if type(seed_value) == int or type(seed_value) == float:          
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
    else: 
        print(f"Invalid seed value: {seed_value}. Cannot set seeds.")




def get_default_hps(hps):
    default_hps = { hp["name"]:hp["default"] for hp in hps }
    return default_hps



def get_data(data_path): 
    try:
        return pd.read_csv(data_path)
    except: 
        raise Exception(f"Error reading data at: {data_path}")


def get_data_schema(data_schema_path): 
    try:
        data_schema = json.load(open(data_schema_path))  
    except: 
        raise Exception(f"Error reading data_schema file at: {data_schema_path}")   
    return data_schema



def get_data_and_schema(data_path, data_schema_path):
    data = get_data(data_path)
    schema = get_data_schema(data_schema_path)
    return data, schema


def save_model_and_preprocessor(model, preprocess_pipe, model_path):    
    # save main model
    save_model(model, model_path) 
    
    # save preprocessor
    preprocessor_fname = cfg.model_cfg['default_fnames']['preprocessor_fname']
    pp_fpath_and_name = os.path.join(model_path, preprocessor_fname)    
    save_preprocessor(preprocess_pipe, pp_fpath_and_name)
    
    return    
    

def load_model_and_preprocessor(model_path):
    # load model
    model = load_model(model_path=model_path)
    
    # load preprocessor
    preprocessor_fname = cfg.model_cfg['default_fnames']['preprocessor_fname']
    pp_file_path_and_name = os.path.join(model_path, preprocessor_fname)    
    preprocess_pipe = load_preprocessor(pp_file_path_and_name)
    
    
    return model, preprocess_pipe
