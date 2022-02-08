import numpy as np, pandas as pd, random
import sys, os, time, pprint
import joblib

import algorithm.utils as utils
import algorithm.preprocessing.preprocess_pipe as pp_pipe
import algorithm.load_config as cfg

preprocessor_fname = cfg.model_cfg['default_fnames']['preprocessor_fname'] 
model_params_fname = cfg.model_cfg['default_fnames']['model_params_fname'] 
model_wts_fname = cfg.model_cfg['default_fnames']['model_wts_fname'] 
predictions_fname = cfg.model_cfg['default_fnames']['predictions_fname'] 
MODEL_NAME = cfg.model_cfg['MODEL_NAME']



def run_predictions(data_fpath, model_path, output_path):
    
    # clear previous prediction and score files
    print("Clearing prior predictions (if any) ... ")
    clear_predictions_dir(output_path)

    # get data
    print("Reading prediction input data... ")
    test_data = utils.get_data(data_fpath) 
    # print("test data shape: ", test_data.shape)

    # load the model, and preprocessor 
    print(f"Loading trained {MODEL_NAME}... ")
    model, preprocess_pipe = utils.load_model_and_preprocessor(model_path)    
    # sys.exit()

    # get predictions from model
    print("Making predictions... ")
    preds_df = predict_with_model(test_data, model, preprocess_pipe)
    
    print("Saving predictions... ")
    if preds_df is not None:
        preds_df.to_csv(os.path.join(output_path, predictions_fname), index=False)
    else: 
        print("No predictions saved.")

    print("Done with predictions.")
    return 0




def clear_predictions_dir(output_path):
    for fname in os.listdir(output_path):
        fpath = os.path.join(output_path, fname)
        os.unlink(fpath)


def predict_with_model(predict_data, model, preprocess_pipe):         

    # transform data
    proc_data = preprocess_pipe.transform(predict_data)    
    
    pred_X = proc_data['X'].astype(np.float)
    # pd.DataFrame(pred_X).to_csv("pred_X.csv") #; sys.exit()
    
    if proc_data['y'] is not None: 
        pred_y = proc_data['y'].astype(np.float)
    
    if pred_y is not None: pred_y = pred_y.astype(np.float32)
    

    preds = model.predict( pred_X )
    preds = pp_pipe.get_inverse_transform_on_preds(preprocess_pipe, cfg.model_cfg, preds)
     
    pred_col_name = cfg.model_cfg['model_outputs']['preds_col_name']
    predict_data[pred_col_name] = preds
    
    return predict_data


