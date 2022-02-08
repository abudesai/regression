#!/usr/bin/env python

import paths
from algorithm.train_test_predict import run_predictions


def predict():
    resp = run_predictions(
        data_fpath = paths.test_ratings_fpath, 
        model_path = paths.model_path, 
        output_path = paths.output_path)
        
    return resp



if __name__ == '__main__': 
    predict() 