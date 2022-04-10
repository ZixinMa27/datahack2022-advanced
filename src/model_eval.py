"""
evaluate model 
"""

# load packages 
import os
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split

# load file 
from src.data_ingestion import *
from src.feature_engineering import * 
from src.model_train import * 

# log path 
LOG_PATH = 'log/records.txt'


# ======== aux ======

def run_single_loop_evaluation(model_name, params, train_data_raw):
    """ run a single simulation """
    train_data, test_data = train_test_split(
        train_data_raw, test_size=0.2, stratify=train_data_raw['Sentiment']
    )

    # make features
    make_features(train_data, test_data)
    X_train, y_train, X_val, y_val = create_X_y()

    # fit and predict
    model = model_name(X_train, y_train) 
    model.fit()
    train_evaluations = model.evaluate(X_train, y_train)
    val_evaluations = model.evaluate(X_val, y_val)
    evaluations = {
        'train': train_evaluations,
        'val': val_evaluations
    }

    # record 
    with open(LOG_PATH, 'a') as f:
        f.write(str(evaluations))
        f.write('\n')


# ======= overall =======

def run_model_evaluation(model, params, n = 10):
    """ run evaluation on a model for n many times, record metrics """
    # read data 
    train_data_raw = read_train()

    # log name 
    with open(LOG_PATH, 'a') as f:
        f.write(model.name)
        f.write('\n')
    
    # run loop 
    for _ in range(n):
        run_single_loop_evaluation(model, params, train_data_raw)

    # log space 
    with open(LOG_PATH, 'a') as f:
        f.write('\n')
