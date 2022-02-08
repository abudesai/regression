#!/usr/bin/env python

import paths
from algorithm.train_test_predict import score_predictions


def score():
    resp = score_predictions(paths.output_path)
    return resp



if __name__ == '__main__': 
    score() 