import os, json


model_cfg_path = os.path.join(os.path.dirname(__file__), 'config', 'model_config.json')
model_cfg = None
try:
    model_cfg = json.load(open(model_cfg_path)) 
except: 
    raise Exception(f"Error reading model config file at: {model_cfg_path}")   
# print(model_cfg); sys.exit()



hp_f_path = os.path.join(os.path.dirname(__file__), 'config', 'hyperparameters.json')
hps = None
try:
    hps = json.load(open(hp_f_path))  
except: 
    raise Exception(f"Error reading hyperparameters file at: {hp_f_path}")   
