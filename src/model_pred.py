"""
Final predictions 
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

# ======= methods =======
def final_predict(model_name, params):
    # read data 
    train_data = read_train()
    test_data = read_test()

    # make features 
    make_features(train_data, test_data, is_test=True)
    X_train, y_train, X_val = create_X_y(is_test=True)

    # TODO: pass in model and params 
