#!/usr/bin/env python

import paths
from algorithm.training import run_training


def train():
    resp = run_training(paths.train_ratings_fpath, paths.model_path, paths.logs_path)
    return resp



if __name__ == '__main__': 
    train() 