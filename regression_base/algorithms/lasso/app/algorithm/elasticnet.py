
import numpy as np, pandas as pd
import os, sys
from sklearn.utils import shuffle
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten, \
    Concatenate, Dense, Activation
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import algorithm.load_config as cfg


model_params_fname = cfg.model_cfg['default_fnames']['model_params_fname']
model_wts_fname = cfg.model_cfg['default_fnames']['model_wts_fname']
MODEL_NAME = cfg.model_cfg['MODEL_NAME']

COST_THRESHOLD = float('inf')



class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("\nCost is inf, so stopping training!!")
            self.model.stop_training = True


class ElasticNet(): 
    
    def __init__(self, M, l1_reg=1e-3, l2_reg=1e-3, lr = 1e-2, **kwargs) -> None:
        super(ElasticNet, self).__init__(**kwargs)
        self.M = M
        self.l1_reg = np.float(l1_reg)
        self.l2_reg = np.float(l2_reg)
        self.lr = lr
        
        self.model = self.build_model()
        self.model.compile(
            loss='mse',
            # optimizer=Adam(learning_rate=self.lr),
            optimizer=SGD(learning_rate=self.lr),
            metrics=['mse'],
        )
        
        
    def build_model(self): 
        input_ = Input(self.M)
        reg = l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        output_ = Dense(1, activity_regularizer=reg )(input_)

        model = Model(input_, output_)
        return model
    
    
    def fit(self, train_X, train_y, valid_X=None, valid_y=None,
            batch_size=256, epochs=100, verbose=0):        
        
        if valid_X is not None and valid_y is not None:
            early_stop_loss = 'val_loss' 
            validation_data = [valid_X, valid_y]
        else: 
            early_stop_loss = 'loss'
            validation_data = None   
            
        
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-3, patience=3)      
        infcost_stop_callback = InfCostStopCallback()
    
        history = self.model.fit(
                x = train_X,
                y = train_y, 
                batch_size = batch_size,
                validation_data=validation_data,
                epochs=epochs,
                verbose=verbose,
                shuffle=True,
                callbacks=[early_stop_callback, infcost_stop_callback]
            )
        return history
    
    
    def predict(self, X): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.summary()

    
    def save(self, model_path): 
        model_params = {
            "M": self.M,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))

        self.model.save_weights(os.path.join(model_path, model_wts_fname))


    @classmethod
    def load(cls, model_path): 
        # print(model_params_fname, model_wts_fname)
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        model = cls(**model_params)
        model.model.load_weights(os.path.join(model_path, model_wts_fname)).expect_partial()
        return model


def save_model(model, model_path):
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = ElasticNet.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model



def get_data_based_model_params(data): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''  
    return {"M": data.shape[1]}