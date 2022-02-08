#!/usr/bin/env python

import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import logging 
import numpy as np, pandas as pd
import time, pprint
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

import algorithm.preprocessing.preprocess_pipe as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
import algorithm.scoring as scoring
from algorithm.elasticnet import ElasticNet, get_data_based_model_params
import algorithm.load_config as cfg


def run_training(train_data_path, data_schema_path, model_path, logs_path, random_state=42): 

    print("Starting the training process...")
    start = time.time()

    # set seeds 
    utils.set_seeds(seed_value = random_state)        

    # get training data and schema
    orig_train_data, data_schema = utils.get_data_and_schema(train_data_path, data_schema_path)    
    
    # get default hyper-parameters
    hyper_params = utils.get_default_hps(cfg.hps)    
    
    print(f"Training {cfg.model_cfg['MODEL_NAME']} ...")  
    model, _, preprocess_pipe = get_trained_model(orig_train_data, data_schema, hyper_params)
    
    # Save the model and processing pipeline     
    print('Saving model ...')
    utils.save_model_and_preprocessor(model, preprocess_pipe, model_path)    
    
    end = time.time()
    print(f"Total training time: {np.round((end - start)/60.0, 2)} minutes")     
    
    return 0



def get_trained_model(data, data_schema, hyper_params):  
    
    # perform train/valid split 
    train_data, valid_data = train_test_split(data, test_size=cfg.model_cfg['valid_split'])
    # train_data.to_csv("orig_train_data.csv", index=False)
    print('train_data shape:',  train_data.shape, 'valid_data shape:', valid_data.shape) 
       
    
    # preprocess data
    print("Pre-processing data...")
    train_data, valid_data, preprocess_pipe = preprocess_data(train_data, valid_data, data_schema)    
    train_X, train_y = train_data['X'].astype(np.float), train_data['y'].astype(np.float)
    valid_X, valid_y = valid_data['X'].astype(np.float), valid_data['y'].astype(np.float)       
    
    # get model hyper-paameters parameters 
    data_based_params = get_data_based_model_params(train_X)
    model_params = { **data_based_params, **hyper_params }
    # print(model_params); sys.exit()
          
    # Create and train matrix factorization model     
    print('Training model ...')  
          
    model = ElasticNet(  **model_params )  
    # model.summary() ;  sys.exit()  
    history = model.fit(
        train_X=train_X, train_y=train_y, 
        valid_X=valid_X, valid_y=valid_y,
        batch_size = 32, 
        epochs = 1000,
        verbose = 1, 
    )        
   
    # --------------------------------------------------------------------------
    # # testing predictions on validation data
    # preds = model.predict(valid_X)    
    
    # # preds = pp_pipe.get_inverse_transform_on_preds(preprocess_pipe, cfg.model_cfg, preds)
    # # valid_y = pp_pipe.get_inverse_transform_on_preds(preprocess_pipe, cfg.model_cfg, valid_y) 
               
    # loss_types = ['mse', 'rmse', 'mae', 'nmae', 'smape', 'r2']
    # scores_dict = scoring.get_loss_multiple( valid_y, preds, loss_types )    
    # pprint.pprint(scores_dict) #; 
    
    # preds = pd.DataFrame(preds, columns=['pred'])
    # preds['act'] = valid_y
    # print('preds shape:', preds.shape, preds.head())
    # preds.to_csv("preds.csv", index=False) ; 
    # sys.exit()
    # --------------------------------------------------------------------------
    print(f'Finished training {cfg.model_cfg["MODEL_NAME"]} ...')   
    
    return model, history, preprocess_pipe



def preprocess_data(train_data, valid_data, data_schema):
    print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, cfg.model_cfg)    
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, cfg.model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)
    # pd.DataFrame(train_data['X']).to_csv("Processed_train_data_X.csv", index=False) #; sys.exit() 
    # pd.DataFrame(train_data['y']).to_csv("Processed_train_data_y.csv", index=False) #; sys.exit() 
    # sys.exit()
      
    valid_data = preprocess_pipe.transform(valid_data)
    print("Processed valid X/y data shape", valid_data['X'].shape, valid_data['y'].shape)
    # pd.DataFrame(valid_data['X']).to_csv("valid_data.csv", index=False) #; sys.exit() 
    return train_data, valid_data, preprocess_pipe 


