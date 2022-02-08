
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os, sys
import algorithm.load_config as cfg
from algorithm.utils import get_data_schema
import algorithm.scoring as scoring


predictions_fname = cfg.model_cfg['default_fnames']['predictions_fname'] 
scoring_fname = cfg.model_cfg['default_fnames']['scoring_fname'] 
pred_col_name = cfg.model_cfg['model_outputs']['preds_col_name']  
MODEL_NAME = cfg.model_cfg['MODEL_NAME']


act_col_schema_key = cfg.model_cfg['data_schema_keys']['target_attr_name']


def get_nmae(Y, Yhat): 
    sum_abs_diff = np.sum(np.abs(Y - Yhat))
    sum_act = Y.sum()
    return sum_abs_diff / sum_act


def get_smape(Y, Yhat):
    return 100./len(Y) * np.sum(2 * np.abs(Yhat - Y) / (np.abs(Y) + np.abs(Yhat)))


def get_wape(Y, Yhat): 
    abs_diff = np.abs(Y - Yhat)
    return 100 * np.sum(abs_diff) / np.sum(Y)


def get_rmse(Y, Yhat):
    return mean_squared_error(Y, Yhat, squared=False) 


def get_loss(Y, Yhat, loss_type): 
    if loss_type == 'mse':  return mean_squared_error(Y, Yhat)
    if loss_type == 'rmse':  return get_rmse(Y, Yhat)
    elif loss_type == 'mae':  return mean_absolute_error(Y, Yhat)
    elif loss_type == 'nmae':  return get_nmae(Y, Yhat)
    elif loss_type == 'smape':  return get_smape(Y, Yhat)
    elif loss_type == 'r2':  return r2_score(Y, Yhat)
    else: raise Exception(f"undefined loss type: {loss_type}")


loss_funcs = {
    'mse': mean_squared_error,
    'rmse': get_rmse,
    'mae': mean_absolute_error,
    'nmae': get_nmae,
    'smape': get_smape,
    'r2': r2_score,
}


def get_loss_multiple(Y, Yhat, loss_types): 
    scores = {}
    for loss in loss_types:
        scores[loss] = loss_funcs[loss](Y, Yhat)
    return scores



def score_predictions(output_path, data_schema_path):  
    
    
    print("Reading predictions data... ")   
    
    pred_file = os.path.join(output_path, predictions_fname)
    if not os.path.exists(pred_file):
        err_msg = f"No predictions file found. Expected to find: {pred_file}. No scores generated."
        print(err_msg)
        return err_msg

    df = pd.read_csv(pred_file)

    cols = df.columns
    
    data_schema = get_data_schema(data_schema_path)
    # print(data_schema) ; sys.exit()
    
    actual_y_col = data_schema[act_col_schema_key]
    # print("actual_y_col", actual_y_col); sys.exit()

    
    print("Generating scores... ") 
    
    if pred_col_name not in cols:
        err_msg = f"Prediction file missing prediction column '{pred_col_name}'. Cannot generate scores."
        print(err_msg)
        return err_msg
    elif actual_y_col not in cols:
        err_msg = f"Prediction file missing predicted value column '{actual_y_col}'. Cannot generate scores."
        print(err_msg)
        return err_msg
    else: 
        loss_types = ['mse', 'rmse', 'mae', 'nmae', 'smape', 'r2']
        scores_dict = scoring.get_loss_multiple(
            df[actual_y_col], df[pred_col_name], loss_types)
        print("scores:", scores_dict)
        with open(os.path.join(output_path, scoring_fname), 'w') as f: 
            f.write("Attribute,Value\n")
            f.write(f"Model_Name,{MODEL_NAME}\n")
            for loss in loss_types:
                f.write( f"{loss},{round(scores_dict[loss], 4)}\n" )

    
    print("Saved scoring data. ") 
    return 0